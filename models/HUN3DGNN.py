import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math

# ======================================================
# Graph Convolution (AMP-safe)
# ======================================================
from torch.cuda.amp import autocast

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

# ======================================================
# Conv block
# ======================================================
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, layers=2):
        super().__init__()
        ops = []
        for i in range(layers):
            ops += [
                nn.Conv3d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            ]
        self.block = nn.Sequential(*ops)

    def forward(self, x):
        return self.block(x)


# ======================================================
# Region GNN module (core innovation)
# ======================================================
class RegionGNN(nn.Module):
    def __init__(self, in_channels, num_classes, pooled_size,
                 use_logits=True, hybrid_graph=True, gnn_layers=1):
        super().__init__()

        self.use_logits = use_logits
        self.pool = nn.AdaptiveAvgPool3d(pooled_size)
        self.eps = 1e-6

        # --------------------------------------------------
        # Class-wise fusion strength (ONE value per organ)
        # --------------------------------------------------
        self.class_alpha = nn.Parameter(torch.ones(num_classes) * 0.15)

        node_dim = in_channels + (num_classes if use_logits else 0)
        node_dim += 1
        self.gnn = GNNBlock(node_dim, layers=gnn_layers)
        #self.gnn_norm = nn.InstanceNorm3d(in_channels)
        self.gnn_norm = nn.GroupNorm(num_groups=8 if in_channels >= 64 else 4, num_channels=in_channels )

        self.project = nn.Conv3d(node_dim, in_channels, 1)

        # --------------------------------------------------
        # Spatial gate: decides WHERE GNN should act
        # --------------------------------------------------
        self.class_gate = nn.Sequential(
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        # --------------------------------------------------
        # GNN auxiliary classifier (organ presence)
        # --------------------------------------------------
        self.node_dim = in_channels + (num_classes if use_logits else 0) + 1

        self.gnn_cls = nn.Linear(self.node_dim, num_classes)

        # --------------------------------------------------
        # Feature attention (for node pooling)
        # --------------------------------------------------
        self.att_feat = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # --------------------------------------------------
        # Logit attention (class importance)
        # --------------------------------------------------
        if use_logits:
            self.att_logit = nn.Sequential(
                nn.Conv3d(num_classes, num_classes // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_classes // 2, 1, 1),
                nn.Sigmoid()
            )

        self.hybrid_graph = hybrid_graph

    def attention_pool(self, x, att):
        att = torch.clamp(att, min=1e-4)
        weighted_sum = self.pool(x * att)
        att_sum = torch.clamp(self.pool(att), min=1e-4)
        return weighted_sum / att_sum

    @torch._dynamo.disable
    def forward(self, feat, logits=None, warmup_multiplier=None):
        B, C, D, H, W = feat.shape
        # warmup 
        if warmup_multiplier is None:
            warmup_multiplier = torch.ones(
                self.class_alpha.shape,
                device=feat.device
            )

        if logits is None:
            use_logits = False
        else:
            use_logits = self.use_logits

        # --------------------------------------------------
        # 1️⃣ Feature attention pooling → nodes
        # --------------------------------------------------
        feat_att = self.att_feat(feat)
        pooled_feat = self.attention_pool(feat, feat_att)

        nodes = pooled_feat.flatten(2).transpose(1, 2)
        nodes = F.layer_norm(nodes, nodes.shape[-1:])

        # --------------------------------------------------
        # 2️⃣ Optional logits as node features (DETACHED)
        # --------------------------------------------------
        if self.use_logits and logits is not None:
            log_att = self.att_logit(torch.sigmoid(logits.detach()))
            pooled_logits = self.attention_pool(logits.detach(), log_att)
            log_nodes = pooled_logits.flatten(2).transpose(1, 2)
            nodes = torch.cat([nodes, log_nodes], dim=-1)

        Dp, Hp, Wp = pooled_feat.shape[2:]

        edge_index = (
            build_hybrid_graph(Dp, Hp, Wp, feat.device)
            if self.hybrid_graph else
            build_grid_graph(Dp, Hp, Wp, feat.device)
        )

        # --------------------------------------------------
        # 3️⃣ GNN propagation
        # --------------------------------------------------
        # --------------------------------------------------
        # Z positional encoding (critical for anatomy)
        # --------------------------------------------------
        z = torch.linspace(0, 1, Dp, device=feat.device)
        z = z.view(1, Dp, 1).expand(B, Dp, Hp * Wp)
        z = z.reshape(B, -1, 1)

        nodes = torch.cat([nodes, z], dim=-1)

        nodes = self.gnn(nodes, edge_index)
        # --------------------------------------------------
        # GNN auxiliary output (organ presence)
        # --------------------------------------------------
        gnn_logits = self.gnn_cls(nodes)      # [B, N, C]
        gnn_logits = gnn_logits.mean(dim=1)   # [B, C]
        self.last_gnn_logits = gnn_logits
        nodes = nodes.transpose(1, 2).reshape(B, -1, Dp, Hp, Wp)
        nodes = self.project(nodes)

        nodes = F.interpolate(
            nodes, size=(D, H, W),
            mode="trilinear", align_corners=False
        )
        nodes = self.gnn_norm(nodes)

        # --------------------------------------------------
        # 4️⃣ Spatial gate (WHERE to fuse)
        # --------------------------------------------------
        gate = self.class_gate(nodes)

        # --------------------------------------------------
        # 5️⃣ Uncertainty gate (WHEN to trust GNN)
        # --------------------------------------------------
        if logits is not None:
            with torch.no_grad():
                prob = torch.softmax(logits, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-6), dim=1, keepdim=True)
                entropy = entropy / math.log(prob.shape[1])
            uncert_gate = torch.sigmoid(entropy)
            #uncert_gate = torch.sigmoid(2.0 * (entropy - 0.5))
            #uncert_gate = torch.clamp(uncert_gate, 0.1, 0.7)
        else:
            uncert_gate = torch.ones_like(gate)

        if uncert_gate.shape[2:] != gate.shape[2:]:
            uncert_gate = F.interpolate(
                uncert_gate, size=gate.shape[2:],
                mode="trilinear", align_corners=False
            )

        fusion_gate = torch.clamp(gate * uncert_gate, min=0.05)
        fused = feat * (1 - fusion_gate) + nodes * fusion_gate

        # --------------------------------------------------
        # 6️⃣ Class-aware α (WHO needs GNN)
        # --------------------------------------------------
        if logits is not None:
            prob = torch.softmax(logits, dim=1)
        else:
            # uniform probability → no class bias
            prob = torch.full(
                (feat.shape[0], self.class_alpha.numel(), *feat.shape[2:]),
                1.0 / self.class_alpha.numel(),
                device=feat.device
            )
        # -------------------------------------------------
        # SAFE class-aware alpha (works for all nnUNet stages)
        # -------------------------------------------------
        if logits is not None:
            prob = torch.softmax(logits, dim=1)

            effective_alpha = self.class_alpha
            effective_alpha = torch.clamp(effective_alpha, 0.05, 0.5)

            if prob.dim() == 5:
                # Spatial logits: [B, C, D, H, W]
                alpha_map = torch.sum(
                    prob * torch.sigmoid(3 * effective_alpha).view(1, -1, 1, 1, 1),
                    dim=1,
                    keepdim=True
                )  # [B,1,D,H,W]
            else:
                # Class-only logits: [B, C]
                alpha_scalar = torch.sum(
                    prob * torch.sigmoid(3 * effective_alpha).view(1, -1),
                    dim=1,
                    keepdim=True
                )  # [B,1]

                # 🔥 FORCE spatial broadcast
                alpha_map = alpha_scalar[:, :, None, None, None]

        else:
            # No logits → disable GNN effect safely
            alpha_map = torch.zeros(
                feat.shape[0], 1, 1, 1, 1,
                device=feat.device
            )

        # 🔒 FINAL GUARANTEE
        if alpha_map.shape[2:] != feat.shape[2:]:
            alpha_map = F.interpolate(
                alpha_map,
                size=feat.shape[2:],
                mode="trilinear",
                align_corners=False
            )


        # --------------------------------------------------
        # 7️⃣ Final fusion
        # --------------------------------------------------
        out = feat + alpha_map * (fused - feat)

        return out



# ======================================================
# Decoder
# ======================================================
class Decoder3D(nn.Module):
    def __init__(self, base_channels, num_classes, deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.dec5 = ConvBlock3D(base_channels * 32, base_channels * 16)
        self.dec4 = ConvBlock3D(base_channels * 16, base_channels * 8)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        self.out_final = nn.Conv3d(base_channels, num_classes, 1)

        if deep_supervision:
            self.ds4 = nn.Conv3d(base_channels * 16, num_classes, 1)
            self.ds3 = nn.Conv3d(base_channels * 8, num_classes, 1)
            self.ds2 = nn.Conv3d(base_channels * 4, num_classes, 1)
            self.ds1 = nn.Conv3d(base_channels * 2, num_classes, 1)


# ======================================================
# FINAL MODEL
# ======================================================
class HUN3DGNN(nn.Module):
    def __init__(self, in_channels, num_classes,
                 base_channels=48, dropout_rate=0.2, cbam_ratio=8, do_ds=True):
        super().__init__()
        self.do_ds = do_ds

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8)
        self.enc5 = ConvBlock3D(base_channels * 8, base_channels * 16)

        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(base_channels * 16, base_channels * 32)

        # GNN at encoder levels 2 and 3 are not working good so comment this only botelneck GNN
        # Encoder-2: NO logits, local graph
        self.gnn2 = RegionGNN(
            base_channels * 2,
            num_classes,
            pooled_size=(4,16,16),
            use_logits=False,
            hybrid_graph=False,
            gnn_layers=1
        )

        # Encoder-3: logits + hybrid graph
        self.gnn3 = RegionGNN(
            base_channels * 4,
            num_classes,
            pooled_size=(2,8,8),
            use_logits=True,
            hybrid_graph=True,
            gnn_layers=1
        )
        

        # Bottleneck: strong semantic GNN both logistic and hybrid
        self.gnn_bottleneck = RegionGNN(
            base_channels * 32,
            num_classes,
            pooled_size=(1,4,4),
            use_logits=True,
            hybrid_graph=True,
            gnn_layers=1
        )


        self.dropout = nn.Dropout3d(dropout_rate)

        # Upsampling
        self.up5 = nn.ConvTranspose3d(base_channels * 32, base_channels * 16, 2, 2)
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 2, 2)
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)

        self.decoder = Decoder3D(base_channels, num_classes, do_ds)

        # Auxiliary classifiers for GNN semantics
        self.cls2 = nn.Conv3d(base_channels * 2, num_classes, 1)
        self.cls3 = nn.Conv3d(base_channels * 4, num_classes, 1)
        self.cls4 = nn.Conv3d(base_channels * 8, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # ---- GNN at encoder-2 ----
        #logits2 = self.cls2(e2)
        #e2 = self.gnn2(e2)

        # ---- GNN at encoder-3 ----
        #logits3 = self.cls3(e3)
        #e3 = self.gnn3(e3, torch.softmax(logits3, dim=1))

        # ----Botteleneck
        b = self.dropout(self.bottleneck(self.pool(e5)))
        b = self.gnn_bottleneck(b, torch.softmax(self.cls4(e4), dim=1))

        #b = self.dropout(self.bottleneck(self.pool(e4)))

        d5 = self.decoder.dec5(torch.cat([self.up5(b), e5], 1))
        d4 = self.decoder.dec4(torch.cat([self.up4(d5), e4], 1))
        d3 = self.decoder.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.decoder.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.decoder.dec1(torch.cat([self.up1(d2), e1], 1))

        if self.do_ds:
            return [
                self.decoder.out_final(d1),
                self.decoder.ds1(d2),
                self.decoder.ds2(d3),
                self.decoder.ds3(d4),
                self.decoder.ds4(d5),
            ]

        return self.decoder.out_final(d1)
