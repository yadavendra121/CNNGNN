import torch.nn as nn
from .unet3d import HUN3DGNN


class FullModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = HUN3DGNN(
            in_channels=1, #4
            num_classes=config.num_classes,
            base_channels=48,
            dropout_rate=0.2,
            do_ds=True,
        )

    def forward(self, x):

        outputs, aux_logits, node_feats, node_labels = self.model(x)

        if isinstance(outputs, list):
            final_logits = outputs[0]
        else:
            final_logits = outputs

        return final_logits, aux_logits, node_feats, node_labels
