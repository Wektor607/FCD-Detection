import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from monai.losses import DiceLoss
from torchvision.ops import sigmoid_focal_loss

class SegmentationLoss(nn.Module):
    """
    Сочетает фокальный, FP- и Dice-лоссы
    c обучаемыми коэффициентами w = [w_focal, w_fp, w_dice],
    которые нормируем через softmax, чтобы всегда было w >= 0 и суммарно = 1.
    """
    def __init__(self, num_ds: int, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_fn = DiceLoss(include_background=False, sigmoid=True)
        # инициализация вектора коэффициентов в равных долях
        self.init = torch.tensor([0.5, 0.1, 0.4])
        self.w = nn.Parameter(self.init)

        # логарифмы σ_i, инициализируем нулями (=> σ_i=1, w_i=1/(2·1^2)=0.5)
        # self.log_sigma = nn.Parameter(torch.zeros(num_ds))
    
    def static_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        focal = sigmoid_focal_loss(logits, target, 
                                   gamma=self.gamma, 
                                   alpha=self.alpha, 
                                   reduction="mean")
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

        # # 4) Dynamic weights через softmax
        # weights_raw = F.softmax(self.w, dim=0)  # [w_focal, w_fp, w_dice]

        # # 5) Enforce minimum for w_dice и renormalize
        # min_dice = 0.2  # ваш порог
        # w_dice = torch.clamp(weights_raw[2], min=min_dice)
        # sum_rest = weights_raw[0] + weights_raw[1]
        # # если sum_rest == 0, можно добавить маленькую eps, но обычно оно >0
        # w_focal = weights_raw[0] * (1 - w_dice) / sum_rest
        # w_fp    = weights_raw[1] * (1 - w_dice) / sum_rest
        # weights = torch.stack([w_focal, w_fp, w_dice], dim=0)

        # debug
        print(f"w_focal={weights[0]:.3f}, w_fp={weights[1]:.3f}, w_dice={weights[2]:.3f}")

        loss = weights[0] * focal + weights[1] * fp_loss + weights[2] * dice
        return loss
    
    def deep_supervision_loss(self, logits: torch.Tensor, target:torch.Tensor, ds_logits: List[torch.Tensor]) -> torch.Tensor:
        loss_main = self.static_loss(logits, target)
        loss_ds = 0.0
        for i, logits_i in enumerate(ds_logits):
            B, Ntot, _ = logits_i.shape
            Nh = Ntot // 2

            # 2.1) отделяем «лево» и «право»
            hemi_logits = logits_i.view(B, 2, Nh)  # [B,2,Nh]
            y_ds = self.downsample_labels_uniform(target, Nh) # [B,2,Ni]

            sigma_i = torch.exp(self.log_sigma[i].clamp(min=-2.0))
            w_i = 1.0 / (2 * sigma_i**2)
            loss_ds += w_i * self.static_loss(hemi_logits, y_ds) + torch.log(sigma_i)
        
        return loss_main + loss_ds

    # def author_deep_supervision_loss(self, logits: torch.Tensor, target:torch.Tensor, ds_logits: List[torch.Tensor]=None) -> torch.Tensor:
    # # Changed by icospheres
    @staticmethod    
    def downsample_labels_uniform(y: torch.Tensor, N_out: int) -> torch.Tensor:
        """
        y: [B, 2, N_finest]
        N_out: число вершин в одном полушарии на этом уровне
        returns y_ds: [B, 2, N_out]
        """
        B, _, N_finest = y.shape
        # отношению fine→coarse
        ratio = N_finest / N_out
        # индексы из [0..N_finest)
        idx = torch.arange(N_out, device=y.device).float() * ratio
        idx = idx.floor().long().clamp(0, N_finest-1)  # [N_out]
        # берём для обоих полушарий
        y_ds = y[:, :, idx]  # [B,2,N_out]
        return y_ds
