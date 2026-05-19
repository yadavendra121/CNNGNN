import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxPresenceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, aux_logits, target):
        """
        aux_logits: (B, C)
        target: (B, D, H, W)
        """

        B = target.shape[0]
        device = target.device

        presence = torch.zeros(B, self.num_classes, device=device)

        for b in range(B):
            classes = torch.unique(target[b])
            presence[b, classes] = 1

        return F.binary_cross_entropy_with_logits(aux_logits, presence)
