import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, node_feats, node_labels):

        # convert one-hot to index if needed
        if node_labels.dim() == 3:
            node_labels = torch.argmax(node_labels, dim=-1)

        B, N, C = node_feats.shape

        features = node_feats.reshape(B * N, C)
        labels = node_labels.reshape(B * N)

        features = F.normalize(features, dim=1)

        sim_matrix = torch.matmul(features, features.T)

        labels = labels.unsqueeze(1)
        labels_equal = labels == labels.T

        # remove self similarity
        mask = torch.eye(labels_equal.shape[0], device=labels_equal.device).bool()
        labels_equal = labels_equal & ~mask

        positives = sim_matrix[labels_equal]
        negatives = sim_matrix[~labels_equal]

        loss = -positives.mean() + negatives.mean()

        return loss

