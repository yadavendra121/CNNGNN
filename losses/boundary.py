import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as nd


def compute_sdf(mask, num_classes):
    """
    Compute signed distance map for each class
    mask: (B, D, H, W)
    """
    B, D, H, W = mask.shape
    sdf = np.zeros((B, num_classes, D, H, W))

    mask_np = mask.cpu().numpy()

    for b in range(B):
        for c in range(1, num_classes):  # skip background optional
            posmask = mask_np[b] == c
            if posmask.any():
                negmask = ~posmask

                posdis = nd.distance_transform_edt(posmask)
                negdis = nd.distance_transform_edt(negmask)

                sdf[b, c] = negdis - posdis

    return torch.tensor(sdf).float().to(mask.device)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        """
        logits: (B, C, D, H, W)
        target: (B, D, H, W)
        """

        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        sdf = compute_sdf(target, num_classes)

        loss = torch.mean(probs * sdf)

        return loss
