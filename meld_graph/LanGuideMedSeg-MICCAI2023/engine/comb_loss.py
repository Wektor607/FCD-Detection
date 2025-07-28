import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from monai.losses import DiceLoss

class CombinedSegmentationLoss(nn.Module):
    """
    Сочетает фокальный, FP- и Dice-лоссы
    c обучаемыми коэффициентами w = [w_focal, w_fp, w_dice],
    которые нормируем через softmax, чтобы всегда было w >= 0 и суммарно = 1.
    """
    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_fn = DiceLoss(include_background=False, sigmoid=True)
        # инициализация вектора коэффициентов в равных долях
        init = torch.tensor([0.6, 0.1, 0.3])
        self.w = nn.Parameter(init)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits, target — [B, N], target ∈ {0,1}

        # 1) Focal
        focal = sigmoid_focal_loss(
            logits, target, 
            gamma=self.gamma, 
            alpha=self.alpha, 
            reduction="mean"
        )

        # 2) FP-loss (1 – precision on negative class)
        prob = torch.sigmoid(logits)
        tn = ((1 - prob) * (1 - target)).sum()
        fp = (prob * (1 - target)).sum()
        fp_loss = 1 - (tn / (tn + fp + 1e-6))

        # 3) Dice
        dice = self.dice_fn(logits, target)

        # нормируем коэффициенты, чтобы не улетали в отрицательное или неравномерное:
        weights = F.softmax(self.w, dim=0)  # [w_focal, w_fp, w_dice] суммируются в 1

        loss = weights[0] * focal + weights[1] * fp_loss + weights[2] * dice
        print("\n")
        print("w_focal", weights[0])
        print("w_fp",    weights[1])
        print("w_dice",  weights[2])
        return loss