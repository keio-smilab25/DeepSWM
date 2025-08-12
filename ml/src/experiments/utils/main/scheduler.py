import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Tuple


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_epochs: int = 10,
    cosine_epochs: int = 100
) -> Tuple[torch.optim.Optimizer, Optional[_LRScheduler]]:
    """Create optimizer and scheduler
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Number of warmup epochs
        cosine_epochs: Number of cosine annealing epochs
        
    Returns:
        Tuple of optimizer and scheduler
    """
    # Create optimizer
    if optimizer_name.lower() == "adamw":
        # More aggressive approach to find ALL problematic parameters
        decay_params = []
        no_decay_params = []
        
        print("Comprehensive parameter analysis for weight_decay compatibility...")
        for name, param in model.named_parameters():
            exclude_from_decay = False
            reason = ""
            
            # Exclude all potentially problematic patterns:
            
            # 1. All S5SSM related parameters (not just B)
            if '.s5.seq.' in name:
                exclude_from_decay = True
                reason = "S5SSM parameter"
            
            # 2. Any tensor with dimension 2 that might cause conflicts
            elif len(param.shape) >= 2 and 2 in param.shape:
                # Only keep simple bias vectors and common classification patterns
                if (len(param.shape) == 1 and param.shape[0] == 2) or \
                   (len(param.shape) == 2 and param.shape in [torch.Size([2, 2]), torch.Size([4, 2]), torch.Size([2, 4])]):
                    pass  # These are safe
                else:
                    exclude_from_decay = True
                    reason = f"tensor with dimension 2: {param.shape}"
            
            # 3. Any 3D tensor (these often cause issues)
            elif len(param.shape) == 3:
                exclude_from_decay = True
                reason = f"3D tensor: {param.shape}"
            
            if exclude_from_decay:
                print(f"Excluding {name}: {reason}")
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        print(f"Analysis complete: {len(decay_params)} decay, {len(no_decay_params)} no-decay")
        
        # Create optimizer with parameter groups
        if no_decay_params:
            optimizer = torch.optim.AdamW([
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ], lr=lr, betas=(0.9, 0.999))
            print(f"AdamW with parameter groups - effective weight_decay: {weight_decay} for {len(decay_params)}/{len(decay_params)+len(no_decay_params)} parameters")
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            print(f"AdamW with uniform weight_decay={weight_decay}")
    elif optimizer_name.lower() == "adam":
        # More aggressive approach to find ALL problematic parameters
        decay_params = []
        no_decay_params = []
        
        print("Comprehensive parameter analysis for weight_decay compatibility...")
        for name, param in model.named_parameters():
            exclude_from_decay = False
            reason = ""
            
            # Exclude all potentially problematic patterns:
            
            # 1. All S5SSM related parameters (not just B)
            if '.s5.seq.' in name:
                exclude_from_decay = True
                reason = "S5SSM parameter"
            
            # 2. Any tensor with dimension 2 that might cause conflicts
            elif len(param.shape) >= 2 and 2 in param.shape:
                # Only keep simple bias vectors and common classification patterns
                if (len(param.shape) == 1 and param.shape[0] == 2) or \
                   (len(param.shape) == 2 and param.shape in [torch.Size([2, 2]), torch.Size([4, 2]), torch.Size([2, 4])]):
                    pass  # These are safe
                else:
                    exclude_from_decay = True
                    reason = f"tensor with dimension 2: {param.shape}"
            
            # 3. Any 3D tensor (these often cause issues)
            elif len(param.shape) == 3:
                exclude_from_decay = True
                reason = f"3D tensor: {param.shape}"
            
            if exclude_from_decay:
                print(f"Excluding {name}: {reason}")
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        print(f"Analysis complete: {len(decay_params)} decay, {len(no_decay_params)} no-decay")
        
        # Create optimizer with parameter groups
        if no_decay_params:
            optimizer = torch.optim.Adam([
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ], lr=lr, betas=(0.9, 0.999))
            print(f"Adam with parameter groups - effective weight_decay: {weight_decay} for {len(decay_params)}/{len(decay_params)+len(no_decay_params)} parameters")
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            print(f"Adam with uniform weight_decay={weight_decay}")
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Create scheduler with warmup and cosine annealing
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=cosine_epochs,
        T_mult=1,
        eta_max=lr,
        T_up=warmup_epochs,
        gamma=0.95
    )
    
    return optimizer, scheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=5, gamma=0.95, last_epoch=-1
    ):
        """
        Args:
            optimizer (Optimizer): Optimization algorithm
            T_0 (int): Length of the first cycle (number of epochs)
            T_mult (int, optional): Cycle multiplier (default: 1)
            eta_max (float, optional): Maximum learning rate (default: 0.1)
            T_up (int, optional): Warm-up period (number of epochs, default: 5)
            gamma (float, optional): Learning rate decay rate (default: 0.95)
            last_epoch (int, optional): Last epoch (default: -1)
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_up = T_up
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            # Warm-up period: linearly increase learning rate
            return [
                base_lr + (self.eta_max - base_lr) * (self.T_cur / self.T_up)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine Annealing with Warm Restarts
            T_i = self.T_0 * (self.T_mult**self.cycle)
            t = self.T_cur - self.T_up
            return [
                self.eta_max
                * (self.gamma**self.cycle)
                * (1 + math.cos(math.pi * t / T_i))
                / 2
                for _ in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_0 * (self.T_mult**self.cycle):
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_0 * (self.T_mult ** (self.cycle - 1))
        else:
            if epoch < 0:
                raise ValueError("Epoch must be a non-negative integer.")
            self.T_cur = epoch
            if self.T_cur >= self.T_0 * (self.T_mult**self.cycle):
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_0 * (self.T_mult ** (self.cycle - 1))

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
