import torch
import torch.nn as nn
from .dice import DiceLoss
from .focal import FocalLoss
from .boundary import BoundaryLoss
from .aux_presence import AuxPresenceLoss
from .contrastive import ContrastiveLoss


class TotalLoss(nn.Module):
    def __init__(self, num_classes,
                 w_dice=1.0,
                 w_focal=1.0,
                 w_boundary=0.3,
                 w_aux=0.1,
                 w_contrast=0.1):
        super().__init__()

        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.boundary = BoundaryLoss()
        self.aux_presence = AuxPresenceLoss(num_classes)
        self.contrastive = ContrastiveLoss()

        # weights
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_boundary = w_boundary
        self.w_aux = w_aux
        self.w_contrast = w_contrast

    def forward(self,
                logits,
                target,
                aux_logits=None,
                node_feats=None,
                node_labels=None):

        # ---- Core segmentation losses ----
        l_dice = self.dice(logits, target)
        l_focal = self.focal(logits, target)
        l_boundary = self.boundary(logits, target)

        total = (
            self.w_dice * l_dice +
            self.w_focal * l_focal +
            self.w_boundary * l_boundary
        )

        loss_dict = {
            "dice": l_dice.item(),
            "focal": l_focal.item(),
            "boundary": l_boundary.item(),
            "aux": 0.0,
            "contrast": 0.0
        }

        # ---- Training-only losses ----
        if self.training:

            if aux_logits is not None:
                l_aux = self.aux_presence(aux_logits, target)
                total += self.w_aux * l_aux
                loss_dict["aux"] = l_aux.item()

            if node_feats is not None and node_labels is not None:
                l_contrast = self.contrastive(node_feats, node_labels)
                total += self.w_contrast * l_contrast
                loss_dict["contrast"] = l_contrast.item()

        return total, loss_dict
