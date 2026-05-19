import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: (B, C, D, H, W)
        target: (B, D, H, W)
        """
        num_classes = logits.shape[1]

        probs = torch.softmax(logits, dim=1)

        target_onehot = F.one_hot(target, num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)

        intersection = torch.sum(probs * target_onehot, dims)
        union = torch.sum(probs + target_onehot, dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()
