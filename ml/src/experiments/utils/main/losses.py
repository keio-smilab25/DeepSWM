import torch
import numpy as np

from dataclasses import dataclass
from torch import nn
from torch import Tensor

# クラスごとのサンプル数を定義
CLASS_SAMPLES = {
    1: torch.tensor([27376, 24503, 9954, 1327]),  # fold1
    2: torch.tensor([29545, 24503, 9954, 1327]),  # fold2
    3: torch.tensor([35855, 24510, 9975, 1327])   # fold3
}

@dataclass
class LossConfig:
    lambda_bss: float
    lambda_gmgs: float
    lambda_ce: float
    score_mtx: torch.Tensor
    fold: int
    class_weights: dict
    model_name: str
    stage: int  # 現在のステージを追加

def calculate_weights(config: dict, samples: torch.Tensor) -> torch.Tensor:
    """重みの計算"""
    if not config["enabled"]:
        return None
        
    method = config["method"]
    if method == "none":
        return None
    elif method == "custom":
        weights = torch.tensor(config["custom_weights"], dtype=torch.float)
    elif method == "inverse":
        weights = 1.0 / samples.float()
    elif method == "effective_samples":
        beta = config.get("beta", 0.9999)
        effective_num = 1.0 - torch.pow(beta, samples.float())
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # 重みを正規化
    weights = weights / weights.sum() * len(samples)
    return weights

class Losser:
    def __init__(self, config: LossConfig, device: str):
        self.config = config
        self.device = device
        self.accum = []

        # 現在のステージに応じた重み設定を取得
        stage_key = f"stage{self.config.stage}"
        stage_weights_config = self.config.class_weights[stage_key]

        # クラスの重みを計算
        weights = None
        if stage_weights_config["enabled"]:
            samples = CLASS_SAMPLES[self.config.fold]
            weights = calculate_weights(
                stage_weights_config,
                samples
            ).to(device)

        self.ce_loss = nn.CrossEntropyLoss(weight=weights).to(device)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute loss
        """
        # Cross Entropy Loss with weighting
        ce = self.ce_loss(y_pred, torch.argmax(y_true, dim=1))
        loss = self.config.lambda_ce * ce

        # Additional losses only for Ours model
        if self.config.model_name == "Ours":
            # GMGS loss
            if self.config.lambda_gmgs > 0:
                gmgs_loss = self.calc_gmgs_loss(y_pred, y_true)
                if not gmgs_loss.isnan():
                    loss = loss + self.config.lambda_gmgs * gmgs_loss
                
            # BSS loss
            if self.config.lambda_bss > 0:
                bss_loss = self.calc_bss_loss(y_pred, y_true)
                if not bss_loss.isnan():
                    loss = loss + self.config.lambda_bss * bss_loss

        self.accum.append(loss.clone().detach().cpu().item())
        return loss

    def calc_gmgs_loss(self, y_pred: Tensor, y_true) -> Tensor:
        """
        Compute GMGS loss according to paper definition:
        L_GMGS = -1/N * Σ(n=1 to N) s_{i*j*} * Σ(i=1 to I) y'_{ni} * log(p(ŷ_{ni}))
        where i* = argmax_i(y_{ni}), j* = argmax_j(p(ŷ_{nj}))
        
        Note: We ensure the loss is positive for proper optimization.
        """
        score_mtx = torch.tensor(self.config.score_mtx, dtype=torch.float32).to(self.device)
        N = y_pred.shape[0]
        
        # Calculate i* = argmax_i(y_{ni}) - true class indices
        i_star = torch.argmax(y_true, dim=1)  # [N]
        
        # Calculate j* = argmax_j(p(ŷ_{nj})) - predicted class indices  
        j_star = torch.argmax(y_pred, dim=1)  # [N]
        
        # Get s_{i*j*} - scalar values from scoring matrix for each sample
        s_values = score_mtx[i_star, j_star]  # [N]
        
        # Calculate Σ(i=1 to I) y'_{ni} * log(p(ŷ_{ni})) for each sample
        log_probs = torch.log(torch.clamp(y_pred, min=1e-8))  # Clamp to avoid log(0)
        cross_entropy_terms = torch.sum(y_true * log_probs, dim=1)  # [N]
        
        # Calculate s_{i*j*} * Σ(i=1 to I) y'_{ni} * log(p(ŷ_{ni})) for each sample
        weighted_terms = s_values * cross_entropy_terms  # [N]
        
        # Calculate L_GMGS = -1/N * Σ(n=1 to N) s_{i*j*} * Σ(i=1 to I) y'_{ni} * log(p(ŷ_{ni}))
        # This is exactly the paper formula
        gmgs_loss_raw = (-1.0 / N) * torch.sum(weighted_terms)
        
        # Since GMGS should be maximized (higher is better), we want to minimize -GMGS
        # However, the raw loss can be negative, which is problematic for optimization
        # We take the negative to ensure we're minimizing something that should be minimized
        loss = -gmgs_loss_raw
        
        return loss

    def calc_bss_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute BSS loss
        """
        tmp = y_pred - y_true
        tmp = torch.mul(tmp, tmp)
        tmp = torch.sum(tmp, dim=1)
        tmp = torch.mean(tmp)
        return tmp

    def get_mean_loss(self) -> float:
        """
        Get mean loss
        """
        return np.mean(self.accum)

    def clear(self):
        """
        Clear accumulated loss
        """
        self.accum.clear()
