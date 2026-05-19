import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        """
        logits: (B, C, D, H, W)
        target: (B, D, H, W)
        """

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_onehot = F.one_hot(target, logits.shape[1])
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        focal_weight = (1 - probs) ** self.gamma

        loss = -target_onehot * focal_weight * log_probs

        return loss.mean()
