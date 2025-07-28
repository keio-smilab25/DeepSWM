import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import numpy as np


class SolarTransforms(nn.Module):
    def __init__(self, p=0.4):
        super().__init__()
        self.p = p
        # Register as buffer, but actual device movement will be done later
        self.register_buffer("channel_mask", torch.arange(10), persistent=False)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        if random.random() > self.p:
            return x

        device = x.device
        history, channels, H, W = x.shape

        # Move channel_mask to the same device as the input tensor
        self.channel_mask = self.channel_mask.to(device)

        # Perform all transformations in batch
        x = x.reshape(-1, 1, H, W)  # [history * channels, 1, H, W]

        # 1. Apply small rotation (physically plausible due to solar rotation)
        # Solar rotation is about 14.7 degrees per day, so small rotations are realistic
        if random.random() < 0.7:  # Higher probability for rotation
            angle = random.uniform(-15, 15)  # Reduced range for physical plausibility
            
            # Create affine transformation matrix
            angle_rad = torch.tensor(angle * np.pi / 180, device=device)
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            
            theta = torch.tensor(
                [
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                ],
                device=device,
            ).float()

            # Generate grid
            grid = torch.nn.functional.affine_grid(
                theta.unsqueeze(0).repeat(x.size(0), 1, 1),
                x.size(),
                align_corners=False,
            )

            # Apply rotation
            x = torch.nn.functional.grid_sample(
                x, grid, mode="bilinear", align_corners=False, padding_mode="reflection"
            )

        # 2. Apply very light Gaussian blur (simulates atmospheric/instrumental effects)
        if random.random() < 0.3:  # Lower probability to preserve fine details
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 0.8)  # Very light blur to preserve features
            blur = transforms.GaussianBlur(kernel_size, sigma)
            x = blur(x)

        # Return to original shape
        x = x.reshape(history, channels, H, W)

        # 3. Generate channel-specific observational noise
        # Different wavelengths have different noise characteristics
        if random.random() < 0.8:  # High probability for realistic noise
            # Lower noise levels for better signal preservation
            # EUV channels (0-6) typically have slightly higher noise
            # AIA channels (7-9) have lower noise
            noise_levels = torch.where(
                self.channel_mask < 7,
                torch.empty(channels, device=device).uniform_(0.005, 0.02),  # Reduced noise
                torch.empty(channels, device=device).uniform_(0.002, 0.01),   # Even lower for AIA
            ).view(1, -1, 1, 1)

            # Calculate standard deviation in bulk
            std_per_channel = x.std(dim=(-2, -1), keepdim=True)

            # Generate noise in bulk and apply
            noise = torch.randn_like(x) * noise_levels * std_per_channel

            # Apply noise
            x = x + noise

        # 4. Apply realistic brightness variations
        # Simulates temporal variations in solar activity and instrumental calibration
        if random.random() < 0.9:  # High probability
            # Smaller variations to preserve solar features
            variations = torch.empty(1, channels, 1, 1, device=device).uniform_(
                0.9, 1.1  # Reduced range for more realistic variations
            )
            x = x * variations

        # 5. Apply channel-specific intensity scaling
        # Different wavelengths can have different calibration factors
        if random.random() < 0.5:
            # Very subtle per-channel scaling
            channel_scaling = torch.empty(1, channels, 1, 1, device=device).uniform_(
                0.95, 1.05
            )
            x = x * channel_scaling

        # 6. Add temporal consistency check and gentle smoothing
        # Ensure that temporal variations are physically reasonable
        if random.random() < 0.3:
            # Apply very light temporal smoothing to maintain physical consistency
            # This simulates the fact that solar features evolve gradually
            temporal_weights = torch.tensor([0.1, 0.3, 0.3, 0.3], device=device).view(4, 1, 1, 1)
            
            # Create a slightly smoothed version
            x_smoothed = torch.zeros_like(x)
            x_smoothed[0] = x[0]  # Keep first frame as is
            for t in range(1, 4):
                x_smoothed[t] = temporal_weights[0] * x[0] + temporal_weights[t] * x[t]
            
            # Blend original and smoothed with low weight
            blend_factor = random.uniform(0.05, 0.15)
            x = (1 - blend_factor) * x + blend_factor * x_smoothed

        return x.contiguous()

    def __call__(self, x):
        return self.forward(x)
