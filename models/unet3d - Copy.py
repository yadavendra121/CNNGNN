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

        self.dec5 = ConvBlock3D(base_channels * 32, base_channels * 16)
        self.dec4 = ConvBlock3D(base_channels * 16, base_channels * 8)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        self.out_final = nn.Conv3d(base_channels, num_classes, 1)

        if self.do_ds:
            self.ds1 = nn.Conv3d(base_channels * 2, num_classes, 1)
            self.ds2 = nn.Conv3d(base_channels * 4, num_classes, 1)
            self.ds3 = nn.Conv3d(base_channels * 8, num_classes, 1)
            self.ds4 = nn.Conv3d(base_channels * 16, num_classes, 1)

    def forward(self, d1, d2, d3, d4, d5):

        final = self.out_final(d1)

        if self.do_ds:
            return [
                final,
                self.ds1(d2),
                self.ds2(d3),
                self.ds3(d4),
                self.ds4(d5),
            ]

        return final


# ============================================================
# Hybrid UNet + RegionGNN
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
        # Encoder
        # ===============================
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8)
        self.enc5 = ConvBlock3D(base_channels * 8, base_channels * 16)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(
            base_channels * 16,
            base_channels * 32
        )

        # ===============================
        # Bottleneck Region GNN
        # ===============================
        self.gnn_bottleneck = RegionGNN(
            in_channels=base_channels * 32,
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
        self.up5 = nn.ConvTranspose3d(
            base_channels * 32, base_channels * 16, 2, 2
        )
        self.up4 = nn.ConvTranspose3d(
            base_channels * 16, base_channels * 8, 2, 2
        )
        self.up3 = nn.ConvTranspose3d(
            base_channels * 8, base_channels * 4, 2, 2
        )
        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2, 2, 2
        )
        self.up1 = nn.ConvTranspose3d(
            base_channels * 2, base_channels, 2, 2
        )

        self.decoder = Decoder3D(
            base_channels, num_classes, do_ds
        )

        # Auxiliary classifier for semantic guidance
        self.semantic_head = nn.Conv3d(
            base_channels * 8,
            num_classes,
            1
        )

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Bottleneck
        b = self.dropout(
            self.bottleneck(self.pool(e5))
        )

        # Semantic guidance from e4
        semantic_logits = self.semantic_head(e4)
        semantic_probs = torch.softmax(semantic_logits, dim=1)

        # GNN refinement
        b, aux_logits, node_feats, node_labels = \
            self.gnn_bottleneck(b, semantic_probs)

        # Decoder
        d5 = self.decoder.dec5(
            torch.cat([self.up5(b), e5], dim=1)
        )
        d4 = self.decoder.dec4(
            torch.cat([self.up4(d5), e4], dim=1)
        )
        d3 = self.decoder.dec3(
            torch.cat([self.up3(d4), e3], dim=1)
        )
        d2 = self.decoder.dec2(
            torch.cat([self.up2(d3), e2], dim=1)
        )
        d1 = self.decoder.dec1(
            torch.cat([self.up1(d2), e1], dim=1)
        )

        outputs = self.decoder(d1, d2, d3, d4, d5)

        return outputs, aux_logits, node_feats, node_labels
