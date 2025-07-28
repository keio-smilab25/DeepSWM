import numpy as np
import torch
from torch import Tensor


class Losser:
    def __init__(self, model, device: str = "cuda", solar_radius: float = 0.65):
        self.model = model
        self.device = device
        self.solar_radius = solar_radius
        self.accum_mse = []
        self.accum_mae = []
        self.accum_solar_mse = []
        self.accum_solar_mae = []

    def __call__(self, imgs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """Calculate loss during training (no change)"""
        target = self.model.patchify_dim10(imgs)

        if self.model.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        if torch.isnan(loss):
            return None
        return loss

    def evaluate(self, imgs: Tensor, pred: Tensor, mask: Tensor):
        """Calculate metrics during evaluation"""
        target = self.model.patchify_dim10(imgs)

        # Calculate patch center coordinates
        L = target.shape[1]
        center = int(np.sqrt(L)) // 2
        h = w = int(np.sqrt(L))
        y, x = np.unravel_index(np.arange(L), (h, w))
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2) / (np.sqrt(2) * center)

        # Mask for solar region
        solar_mask = torch.tensor(
            dist <= self.solar_radius, device=target.device
        ).float()

        if self.model.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # Calculate MSE
        mse = (pred - target) ** 2
        mse = mse.mean(dim=-1)

        # Calculate MAE
        mae = torch.abs(pred - target)
        mae = mae.mean(dim=-1)

        # Metrics for full region
        full_mse = (mse * mask).sum() / mask.sum()
        full_mae = (mae * mask).sum() / mask.sum()

        # Metrics for solar region
        solar_mse = (mse * mask * solar_mask).sum() / (mask * solar_mask).sum()
        solar_mae = (mae * mask * solar_mask).sum() / (mask * solar_mask).sum()

        # Record metrics
        self.accum_mse.append(full_mse.clone().detach().cpu().item())
        self.accum_mae.append(full_mae.clone().detach().cpu().item())
        self.accum_solar_mse.append(solar_mse.clone().detach().cpu().item())
        self.accum_solar_mae.append(solar_mae.clone().detach().cpu().item())

    def get_metrics(self) -> dict:
        """Return average of evaluation metrics"""
        return {
            "mse": np.mean(self.accum_mse),
            "mae": np.mean(self.accum_mae),
            "solar_mse": np.mean(self.accum_solar_mse),
            "solar_mae": np.mean(self.accum_solar_mae),
        }

    def clear(self):
        """Clear evaluation metrics"""
        self.accum_mse.clear()
        self.accum_mae.clear()
        self.accum_solar_mse.clear()
        self.accum_solar_mae.clear()
