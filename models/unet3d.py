import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import ConvBlock3D
from .region_gnn import RegionGNN


# ============================================================
# Decoder Module with Deep Supervision
# ============================================================

class Decoder3D(nn.Module):
    def __init__(self, base_channels, num_classes, do_ds=True):
        super().__init__()
        self.do_ds = do_ds

        # After concatenation channels double
        #self.dec5 = ConvBlock3D(base_channels * 32, base_channels * 16)
        self.dec4 = ConvBlock3D(base_channels * 16, base_channels * 8)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        self.out_final = nn.Conv3d(base_channels, num_classes, 1)

        if self.do_ds:
            self.ds1 = nn.Conv3d(base_channels * 2, num_classes, 1)
            self.ds2 = nn.Conv3d(base_channels * 4, num_classes, 1)
            self.ds3 = nn.Conv3d(base_channels * 8, num_classes, 1)
            #self.ds4 = nn.Conv3d(base_channels * 16, num_classes, 1)

    def forward(self, d1, d2, d3, d4):

        final = self.out_final(d1)

        if self.do_ds:
            return [
                final,
                self.ds1(d2),
                self.ds2(d3),
                self.ds3(d4),
            ]

        return final


# ============================================================
# Hybrid UNet + Multi-Level RegionGNN
# ============================================================

class HUN3DGNN(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        base_channels=48,
        dropout_rate=0.2,
        do_ds=True,
    ):
        super().__init__()

        self.do_ds = do_ds
        self.num_classes = num_classes

        # ===============================
        # Encoder (4 blocks)
        # ===============================
        self.enc1 = ConvBlock3D(in_channels, base_channels)          # 48
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)    # 96
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4) # 192
        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8) # 384
        #self.enc5 = ConvBlock3D(base_channels * 8, base_channels * 16) # 768

        self.pool = nn.MaxPool3d(2)

        # ===============================
        # Bottleneck (5th level)
        # ===============================
        self.bottleneck = ConvBlock3D(
            base_channels * 8,   # 392
            base_channels * 16    # 768
        )

        # ===============================
        # Region GNNs at multiple levels
        # ===============================

        self.gnn2 = RegionGNN(
            in_channels=base_channels * 2,  # 96
            num_classes=num_classes,
            pooled_size=(4, 16, 16),
            use_logits=False,
            hybrid_graph=False,
            gnn_layers=1,
        )

        self.gnn3 = RegionGNN(
            in_channels=base_channels * 4,  # 192
            num_classes=num_classes,
            pooled_size=(2, 8, 8),
            use_logits=True,
            hybrid_graph=False,
            gnn_layers=1,
        )

        self.gnn4 = RegionGNN(
            in_channels=base_channels * 8,  # 384
            num_classes=num_classes,
            pooled_size=(1, 8, 8),
            use_logits=True,
            hybrid_graph=False,
            gnn_layers=1,
        )

        self.gnn_bottleneck = RegionGNN(
            in_channels=base_channels * 16,  # 1536 (FIXED correctly)
            num_classes=num_classes,
            pooled_size=(1, 4, 4),
            use_logits=True,
            hybrid_graph=True,
            gnn_layers=1,
        )

        self.dropout = nn.Dropout3d(dropout_rate)

        # ===============================
        # Upsampling
        # ===============================
        #self.up5 = nn.ConvTranspose3d(base_channels * 32, base_channels * 16, 2, 2)
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 2, 2)
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)

        self.decoder = Decoder3D(base_channels, num_classes, do_ds)

        # Semantic heads for guidance
        self.sem2 = nn.Conv3d(base_channels * 2, num_classes, 1)
        self.sem3 = nn.Conv3d(base_channels * 4, num_classes, 1)
        self.sem4 = nn.Conv3d(base_channels * 8, num_classes, 1)
        self.sem_b = nn.Conv3d(base_channels * 16, num_classes, 1)

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x):

        # ---------------- Encoder ----------------
        e1 = self.enc1(x)                       # 48
        e2 = self.enc2(self.pool(e1))           # 96
        e3 = self.enc3(self.pool(e2))           # 192
        e4 = self.enc4(self.pool(e3))           # 384
        #e5 = self.enc5(self.pool(e4))           # 768

        # ---------------- GNN at e2 ----------------
        e2, _, _, _ = self.gnn2(e2)

        # ---------------- GNN at e3 ----------------
        sem3 = torch.softmax(self.sem3(e3), dim=1)
        e3, _, _, _ = self.gnn3(e3, sem3)

        # ---------------- GNN at e4 ----------------
        sem4 = torch.softmax(self.sem4(e4), dim=1)
        e4, _, _, _ = self.gnn4(e4, sem4)

        # ---------------- Bottleneck ----------------
        b = self.dropout(self.bottleneck(self.pool(e4)))  # 1536
        sem_b = torch.softmax(self.sem_b(b), dim=1)

        b, aux_logits, node_feats, node_labels = self.gnn_bottleneck(b, sem_b)

        # ---------------- Decoder ----------------
        #d5 = self.decoder.dec5(torch.cat([self.up5(b), e5], dim=1))
        d4 = self.decoder.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.decoder.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.decoder.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.decoder.dec1(torch.cat([self.up1(d2), e1], dim=1))

        outputs = self.decoder(d1, d2, d3, d4)

        return outputs, aux_logits, node_feats, node_labels
