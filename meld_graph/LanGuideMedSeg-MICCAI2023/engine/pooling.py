import torch
import torch.nn as nn


class HexPool(nn.Module):
    """
    Max pooling for Icospheres (batched).
        neigh_indices: LongTensor [N_lo, K] — for each lower-level vertex
        a list of indices of its K "children" at the upper level.
    """

    def __init__(self, neigh_indices: torch.LongTensor):
        super().__init__()
        # register as buffer so that it goes to the desired device with .to()
        self.register_buffer("neigh_indices", neigh_indices, persistent=False)

    def forward(self, x: torch.Tensor, center_pool: bool = False) -> torch.Tensor:
        """
        x: [B,H,N_hi,C] (features) 
        or [B,H,N_hi] (labels/maps) 
        or [N_hi,C] / [N_hi] (без батча/головы)
        return:
        if input [B,H,N_hi,C] -> [B,H,N_lo,C]
        if input [B,H,N_hi]   -> [B,H,N_lo]
        if input [N_hi,C]     -> [N_lo,C]
        if input [N_hi]       -> [N_lo]
        """
        N_lo = self.neigh_indices.size(0)

        if center_pool:
            # center-pool: просто "отрезаем" первые N_lo вершин
            if x.dim() == 4:
                return x[:, :, :N_lo, :]
            elif x.dim() == 3:
                return x[:, :, :N_lo]
            elif x.dim() == 2:
                return x[:, :N_lo]
            elif x.dim() == 1:
                return x[:N_lo]
            else:
                raise ValueError(f"Unsupported x.dim()={x.dim()} in center_pool")
        
        if x.dim() == 4:
            
            gathered = x[:, :, self.neigh_indices, :]
            return gathered.max(dim=3).values
        elif x.dim() == 3:
            
            gathered = x[:, :, self.neigh_indices]
            return gathered.max(dim=3).values
        elif x.dim() == 2:
            
            gathered = x[:, self.neigh_indices]
            return gathered.max(dim=1).values
        elif x.dim() == 1:
            
            gathered = x[self.neigh_indices]
            return gathered.max(dim=1).values
        else:
            raise ValueError(f"Unsupported x.dim()={x.dim()}, expected 1–4")



class HexUnpool(nn.Module):
    """
    Mean unpooling for Icospheres.
    """

    def __init__(self, upsample_indices, target_size):
        super(HexUnpool, self).__init__()
        self.upsample_indices = upsample_indices
        self.target_size = target_size

    def forward(self, x, device):
        # print(x.shape)
        B, H, N_from, C = x.shape

        # new_x: [B, H, target_size, C]
        new_x = torch.zeros(B, H, self.target_size, C, device=device)
        # print(new_x.shape)

        # 1) copy old features
        new_x[:, :, :N_from, :] = x

        # 2) calculate average features for new vertices
        upsampled = x[:, :, self.upsample_indices, :].mean(
            dim=3
        )  # → [B, H, N_new, 2, C]

        # 3) insert them into new_x after the original N_from positions
        new_x[:, :, N_from:, :] = upsampled

        new_x = new_x.view(B, H * self.target_size, C)
        return new_x
