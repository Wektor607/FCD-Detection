import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from torchvision.ops import sigmoid_focal_loss


class SegmentationLoss(nn.Module):
    """
    Combines focal, FP and Dice losses
    with trainable coefficients w = [w_focal, w_fp, w_dice],
    which we normalize via softmax so that w is always >= 0 and in total = 1.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_fn = DiceLoss(include_background=False, sigmoid=True)
        self.init = torch.tensor([0.5, 0.1, 0.4])
        self.w = nn.Parameter(self.init)

    def static_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        focal = sigmoid_focal_loss(
            logits, target, gamma=self.gamma, alpha=self.alpha, reduction="mean"
        )
        dice = self.dice_fn(logits, target)

        tn = ((1 - prob) * (1 - target)).sum()
        fp = (prob * (1 - target)).sum()
        fp_loss = 1 - (tn / (tn + fp + 1e-6))

        print(focal, fp_loss, dice)
        return self.init[0] * focal + self.init[1] * fp_loss + self.init[2] * dice

    def dynamic_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits, target — [B, N], target ∈ {0,1}

        # 1) Focal
        focal = sigmoid_focal_loss(
            logits, target, gamma=self.gamma, alpha=self.alpha, reduction="mean"
        )

        # 2) FP-loss (1 – precision on negative class)
        prob = torch.sigmoid(logits)
        tn = ((1 - prob) * (1 - target)).sum()
        fp = (prob * (1 - target)).sum()
        fp_loss = 1 - (tn / (tn + fp + 1e-6))

        # 3) Dice
        dice = self.dice_fn(logits, target)

        # Coef normalization:
        weights = F.softmax(self.w, dim=0)  # [w_focal, w_fp, w_dice] sum up in 1

        # # 4) Dynamic weights через softmax
        # weights_raw = F.softmax(self.w, dim=0)  # [w_focal, w_fp, w_dice]

        # # 5) Enforce minimum for w_dice и renormalize
        # min_dice = 0.2  # ваш порог
        # w_dice = torch.clamp(weights_raw[2], min=min_dice)
        # sum_rest = weights_raw[0] + weights_raw[1]
        # # если sum_rest == 0,
        # w_focal = weights_raw[0] * (1 - w_dice) / sum_rest
        # w_fp    = weights_raw[1] * (1 - w_dice) / sum_rest
        # weights = torch.stack([w_focal, w_fp, w_dice], dim=0)

        # debug
        print(
            f"w_focal={weights[0]:.3f}, w_fp={weights[1]:.3f}, w_dice={weights[2]:.3f}"
        )

        loss = weights[0] * focal + weights[1] * fp_loss + weights[2] * dice
        return loss
