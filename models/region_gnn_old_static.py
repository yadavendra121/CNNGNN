import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast


# ======================================================
# Graph Convolution (AMP-safe)
# ======================================================

class GraphConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lin = nn.Linear(channels, channels, bias=False)
        self.att = nn.Linear(channels, 1, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, edge_index):
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        row, col = edge_index

        # ---- FORCE FP32 FOR MESSAGE PASSING ----
        with autocast(enabled=False):
            x_fp32 = x.float()
            out = torch.zeros_like(x_fp32)

            for b in range(B):
                src = x_fp32[b][col]              # FP32
                msg = self.lin(src)               # FP32
                att = torch.sigmoid(self.att(src))# FP32
                msg = msg * att                   # FP32
                out[b].index_add_(0, row, msg)    # FP32-safe

            out = self.norm(out + x_fp32)
            out = F.relu(out)

        # ---- CAST BACK FOR AMP COMPATIBILITY ----
        return out.to(x.dtype)




class GNNBlock(nn.Module):
    def __init__(self, channels, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([GraphConv(channels) for _ in range(layers)])
    #@torch._dynamo.disable
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = x + layer(x, edge_index)
        return x


# ======================================================
# Graph builder (local grid)
# ======================================================
def build_grid_graph(D, H, W, device):
    idx = torch.arange(D * H * W, device=device).view(D, H, W)
    edges = []

    for dz, dy, dx in [(1,0,0),(0,1,0),(0,0,1)]:
        src = idx[:-dz or None, :-dy or None, :-dx or None].reshape(-1)
        dst = idx[dz:, dy:, dx:].reshape(-1)
        edges += [(src, dst), (dst, src)]

    row = torch.cat([e[0] for e in edges])
    col = torch.cat([e[1] for e in edges])
    return row, col

# hybrid graph building
def build_hybrid_graph(D, H, W, device):
    idx = torch.arange(D * H * W, device=device).view(D, H, W)

    edges = []

    # --- local 6-neighborhood ---
    for dz, dy, dx in [(1,0,0),(0,1,0),(0,0,1)]:
        src = idx[:-dz or None, :-dy or None, :-dx or None].reshape(-1)
        dst = idx[dz:, dy:, dx:].reshape(-1)
        edges += [(src, dst), (dst, src)]

    # --- regional long-range edges ---
    flat = idx.reshape(-1)
    region_stride = max(1, (D*H*W)//64)
    anchors = flat[::region_stride]

    for a in anchors:
        edges.append((a.repeat(len(flat)), flat))
        edges.append((flat, a.repeat(len(flat))))

    row = torch.cat([e[0] for e in edges])
    col = torch.cat([e[1] for e in edges])
    return row, col

# ============================================================
# Region GNN (Upgraded, Paper-Ready Version)
# ============================================================

class RegionGNN(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        pooled_size,
        use_logits=True,
        hybrid_graph=True,
        gnn_layers=1,
    ):
        super().__init__()

        self.use_logits = use_logits
        self.hybrid_graph = hybrid_graph
        self.pool = nn.AdaptiveAvgPool3d(pooled_size)

        # ---------------------------------------------
        # Class-wise learnable fusion strength
        # ---------------------------------------------
        self.class_alpha = nn.Parameter(
            torch.ones(num_classes) * 0.15
        )

        # Node dimension
        node_dim = in_channels
        if use_logits:
            node_dim += num_classes
        node_dim += 1  # Z positional encoding

        self.node_dim = node_dim

        # ---------------------------------------------
        # GNN block
        # ---------------------------------------------
        self.gnn = GNNBlock(node_dim, layers=gnn_layers)

        self.project = nn.Conv3d(node_dim, in_channels, 1)

        self.gnn_norm = nn.GroupNorm(
            num_groups=8 if in_channels >= 64 else 4,
            num_channels=in_channels,
        )

        # ---------------------------------------------
        # Spatial gate (WHERE)
        # ---------------------------------------------
        self.class_gate = nn.Sequential(
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

        # ---------------------------------------------
        # Auxiliary classifier (organ presence)
        # ---------------------------------------------
        self.gnn_cls = nn.Linear(node_dim, num_classes)

        # ---------------------------------------------
        # Feature attention
        # ---------------------------------------------
        self.att_feat = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

        # ---------------------------------------------
        # Logit attention
        # ---------------------------------------------
        if use_logits:
            self.att_logit = nn.Sequential(
                nn.Conv3d(num_classes, num_classes // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_classes // 2, 1, 1),
                nn.Sigmoid(),
            )

    # ============================================================
    # Attention pooling
    # ============================================================

    def attention_pool(self, x, att):
        att = torch.clamp(att, min=1e-4)
        weighted = self.pool(x * att)
        att_sum = torch.clamp(self.pool(att), min=1e-4)
        return weighted / att_sum

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, feat, logits=None):

        B, C, D, H, W = feat.shape

        # -------------------------------------------------
        # 1️⃣ Feature attention pooling
        # -------------------------------------------------
        feat_att = self.att_feat(feat)
        pooled_feat = self.attention_pool(feat, feat_att)

        Dp, Hp, Wp = pooled_feat.shape[2:]

        nodes = pooled_feat.flatten(2).transpose(1, 2)
        nodes = F.layer_norm(nodes, nodes.shape[-1:])

        # -------------------------------------------------
        # 2️⃣ Add detached logits
        # -------------------------------------------------
        if self.use_logits and logits is not None:

            log_att = self.att_logit(torch.sigmoid(logits.detach()))
            pooled_logits = self.attention_pool(
                logits.detach(), log_att
            )

            log_nodes = pooled_logits.flatten(2).transpose(1, 2)

            nodes = torch.cat([nodes, log_nodes], dim=-1)

        # -------------------------------------------------
        # 3️⃣ Z positional encoding
        # -------------------------------------------------
        z = torch.linspace(0, 1, Dp, device=feat.device)
        z = z.view(1, Dp, 1).expand(B, Dp, Hp * Wp)
        z = z.reshape(B, -1, 1)

        nodes = torch.cat([nodes, z], dim=-1)

        # -------------------------------------------------
        # 4️⃣ Graph building
        # -------------------------------------------------
        if self.hybrid_graph:
            edge_index = build_hybrid_graph(Dp, Hp, Wp, feat.device)
        else:
            edge_index = build_grid_graph(Dp, Hp, Wp, feat.device)

        # -------------------------------------------------
        # 5️⃣ GNN propagation
        # -------------------------------------------------
        nodes = self.gnn(nodes, edge_index)

        # Save node embeddings (for contrastive loss)
        node_features = nodes

        # -------------------------------------------------
        # 6️⃣ Auxiliary presence classifier
        # -------------------------------------------------
        aux_logits = self.gnn_cls(nodes)      # [B, N, C]
        aux_logits = aux_logits.mean(dim=1)   # [B, C]

        # Create pseudo node labels from semantic probs
        if logits is not None:
            with torch.no_grad():
                prob = torch.softmax(logits, dim=1)
                pooled_prob = self.pool(prob)
                node_labels = pooled_prob.flatten(2).transpose(1, 2)
        else:
            node_labels = None

        # -------------------------------------------------
        # 7️⃣ Project back to feature space
        # -------------------------------------------------
        nodes = nodes.transpose(1, 2).reshape(
            B, -1, Dp, Hp, Wp
        )

        nodes = self.project(nodes)

        nodes = F.interpolate(
            nodes,
            size=(D, H, W),
            mode="trilinear",
            align_corners=False,
        )

        nodes = self.gnn_norm(nodes)

        # -------------------------------------------------
        # 8️⃣ Spatial gate
        # -------------------------------------------------
        gate = self.class_gate(nodes)

        # -------------------------------------------------
        # 9️⃣ Uncertainty gate
        # -------------------------------------------------
        if logits is not None:
            with torch.no_grad():
                prob = torch.softmax(logits, dim=1)
                entropy = -torch.sum(
                    prob * torch.log(prob + 1e-6),
                    dim=1,
                    keepdim=True,
                )
                entropy = entropy / math.log(prob.shape[1])

            uncert_gate = torch.sigmoid(entropy)

            if uncert_gate.shape[2:] != gate.shape[2:]:
                uncert_gate = F.interpolate(
                    uncert_gate,
                    size=gate.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
        else:
            uncert_gate = torch.ones_like(gate)

        fusion_gate = torch.clamp(gate * uncert_gate, min=0.05)

        fused = feat * (1 - fusion_gate) + nodes * fusion_gate

        # -------------------------------------------------
        # 🔟 Class-aware alpha
        # -------------------------------------------------
        if logits is not None:
            prob = torch.softmax(logits, dim=1)

            effective_alpha = torch.clamp(
                self.class_alpha, 0.05, 0.5
            )

            alpha_map = torch.sum(
                prob * torch.sigmoid(3 * effective_alpha)
                .view(1, -1, 1, 1, 1),
                dim=1,
                keepdim=True,
            )
        else:
            alpha_map = torch.zeros(
                B, 1, 1, 1, 1,
                device=feat.device
            )

        if alpha_map.shape[2:] != feat.shape[2:]:
            alpha_map = F.interpolate(
                alpha_map,
                size=feat.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        # -------------------------------------------------
        # Final fusion
        # -------------------------------------------------
        out = feat + alpha_map * (fused - feat)

        return out, aux_logits, node_features, node_labels
