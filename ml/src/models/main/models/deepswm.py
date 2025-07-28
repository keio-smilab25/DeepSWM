import torch.nn as nn
import torch
from s5 import S5

class DeepSWM(nn.Module):
    def __init__(
        self,
        D=64,  # Feature dimension (scalar)
        drop_path_rate=0.3,  # default:0.0
        layer_scale_init_value=1e-6,
        L=128,       # L: sequence length
        L_SSE=3,     # L_SSE: number of SolarSpatialEncoder layers
        L_LT=2,      # L_LT: number of LongRangeTemporalSSM layers
        L_mixing=6,  # L_mixing: number of MixingSSM layers
        dropout_rates=None  # Dropout rates for each module
    ) -> None:
        super().__init__()

        self.L = L   # Sequence length
        self.drop_path_rate = drop_path_rate
        self.D = D   # Feature dimension
        self.layer_scale_init_value = layer_scale_init_value
        self.L_SSE = L_SSE
        self.L_LT = L_LT
        
        # Set default dropout rates
        if dropout_rates is None:
            dropout_rates = {
                "sse": drop_path_rate,
                "dwcm": drop_path_rate,
                "stssm": drop_path_rate,
                "ltssm": drop_path_rate,
                "mixing_ssm": drop_path_rate,
                "head": 0.5
            }
        self.dropout_rates = dropout_rates

        # Solar Spatial Encoder (SSE)
        self.sse = SolarSpatialEncoder(
            4,  # input_channel (k)
            D=D,
            drop_path_rate=dropout_rates["sse"],
            dwcm_dropout_rate=dropout_rates["dwcm"],
            stssm_dropout_rate=dropout_rates["stssm"],
            layer_scale_init_value=layer_scale_init_value,
            L=L,
            L_SSE=L_SSE,
        )

        # Long-range Temporal SSM (LT-SSM)
        self.lt_ssm = LongRangeTemporalSSM(
            D=D,
            L=L, 
            L_LT=L_LT,
            dropout_rate=dropout_rates["ltssm"]
        )

        # Mixing SSM Encoder for merged features
        self.mixing_ssm_block = SSMBlock(D, depth=L_mixing, dropout_rate=dropout_rates["mixing_ssm"])

        # Classification Head (FFN)
        self.classification_head = ClassificationHead(L*2, D, D//16, dropout_rate=dropout_rates["head"])
        self.softmax = nn.Softmax(dim=1)

    def freeze_feature_extractor(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.classification_head.parameters():
            param.requires_grad = True
            
    def forward(self, x, x_history=None):
        # Solar Spatial Encoder
        h_sse = self.sse(x)  # [batch, L, D]
        
        # Long-range Temporal SSM
        h_lt = self.lt_ssm(x_history)  # [batch, L, D]
        
        # Feature concatenation
        merged_features = torch.cat([h_sse, h_lt], dim=1)  # [batch, 2*L, D]
        
        # Process with Mixing SSM Block
        mixed_features = self.mixing_ssm_block(merged_features)
        
        # Classification head prediction
        logits = self.classification_head(mixed_features)
        
        output = self.softmax(logits)
        
        return output, x


class ClassificationHead(nn.Module):
    def __init__(self, d_model, hidden_dim, mlp_hidden_size, dropout_rate=0.5):
        super().__init__()
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        input_dim = hidden_dim
        
        # FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(d_model * hidden_dim, mlp_hidden_size),
            # nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            # nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, 4)  # 4-class classification
        )
    
    def forward(self, x):
        # Apply global average pooling
        # x = x.transpose(1, 2)  # [batch, d_model, d_model] -> [batch, d_model, d_model]
        # x = self.gap(x).squeeze(-1) 
        # print("x.shape", x.shape)
        x = x.reshape(x.shape[0], -1)
        # print("x.shape", x.shape)        
        # Classification with FFN
        return self.ffn(x)


class SpatioTemporalSSM(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, dropout_rate=0.3):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.s5_layer = SSMBlock(dim, 1, dropout_rate)

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_s5 = self.s5_layer(x_flat)

        out = x_s5.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class DepthwiseChannelSelectiveModule(nn.Module):
    def __init__(self, D, dropout_rate=0.3):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                D,
                D,
                kernel_size=3,
                padding=1,
                groups=D,
                padding_mode="replicate",
            ),
            nn.InstanceNorm3d(D),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.conv2d = nn.Sequential(
            nn.Conv3d(
                D,
                D,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=D,
                padding_mode="replicate",
            ),
            nn.InstanceNorm3d(D),
            nn.ReLU(),
        )

        # Channel-wise weighting (Image type attention)
        self.image_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)), 
            nn.Conv3d(D, D, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(D, D, kernel_size=1),
            nn.Sigmoid()
        )

        # Point-wise refinement (1×1 Conv)
        self.refine = nn.Sequential(
            nn.Conv3d(D, D, 1),
            nn.InstanceNorm3d(D),
            nn.ReLU(),
        )

    def forward(self, h_ds):
        identity = h_ds

        # Parallel feature extraction
        feat3d = self.conv3d(h_ds)
        feat2d = self.conv2d(h_ds)

        F_fused = feat3d + feat2d

        # Channel-wise weighting
        W = self.image_attention(F_fused)
        
        # Apply weights
        weighted_features = F_fused * W

        # Point-wise refinement
        refined = self.refine(weighted_features)
        
        h_dcs = refined + identity
        
        return h_dcs


class SSMLayer(nn.Module):
    def __init__(self, dim, dropout_rate=0.2, init_value=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.s5 = S5(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        x_s5 = self.s5(x_norm)
        x = residual + self.dropout(x_s5)

        residual = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + self.dropout(mlp_out)
        return x

class SSMBlock(nn.Module):
    def __init__(self, dim, depth=2, dropout_rate=0.4):
        super().__init__()
        self.layers = nn.ModuleList([SSMLayer(dim, dropout_rate) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SolarSpatialEncoder(nn.Module):
    def __init__(
        self,
        in_chans=1,  # k: input channels
        D=64,        # D: feature dimension
        drop_path_rate=0.0,
        dwcm_dropout_rate=0.0,
        stssm_dropout_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2],
        L=128,       # L: sequence length in output (previously d_model)
        L_SSE=3,     # L_SSE: number of layers
    ):
        super().__init__()
        self.L = L   # Sequence length (previously d_model)
        self.D = D   # Feature dimension
        self.L_SSE = L_SSE
        
        # Use same feature dimension for all layers
        dims = [D] * (L_SSE + 1)
        
        self.downsample_layers = nn.ModuleList()
        
        # Stem layer (generates h_sse^(0)) - Reduce spatial dimensions by 1/4
        stem = nn.Sequential(
            nn.Conv3d(in_chans, D, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 2, 2)),
            nn.BatchNorm3d(D),
            nn.ReLU(),
            nn.Dropout(drop_path_rate)
        )
        self.downsample_layers.append(stem)

        # Downsampling layers (corresponding to L_SSE layers)
        for l in range(L_SSE):
            if l < 3:  # First 3 layers reduce spatial dimensions by 1/2
                downsample_layer = nn.Sequential(
                    nn.BatchNorm3d(D),
                    nn.Conv3d(D, D, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)),
                    nn.ReLU(),
                    nn.Dropout(drop_path_rate)
                )
            else:  # Additional layers maintain spatial dimensions
                downsample_layer = nn.Sequential(
                    nn.BatchNorm3d(D),
                    nn.Conv3d(D, D, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.ReLU(),
                    nn.Dropout(drop_path_rate)
                )
            self.downsample_layers.append(downsample_layer)

        # Depth-wise Channel Selective Module (DCSM) and
        # Spatio-Temporal State Space Model (ST-SSM)
        self.dcsm_modules = nn.ModuleList()
        self.stssm_modules = nn.ModuleList()
        
        for l in range(L_SSE):
            # Depth-wise Channel Selective Module
            dcsm = DepthwiseChannelSelectiveModule(D, dropout_rate=dwcm_dropout_rate)
            
            # Spatio-Temporal State Space Model
            stssm = SpatioTemporalSSM(dim=D, dropout_rate=stssm_dropout_rate)
            
            self.dcsm_modules.append(dcsm)
            self.stssm_modules.append(stssm)

        self.out_indices = out_indices
        self.dropout = nn.Dropout(drop_path_rate)

        # Final output transformation layer - process merged temporal and spatial dimensions
        self.conv_layers = nn.Sequential(
            nn.Conv2d(D * 10, D, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(D),
            nn.ReLU(),
            nn.Dropout(drop_path_rate),
            nn.Conv2d(D, D//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(D//2),
            nn.ReLU(),
            nn.Dropout(drop_path_rate),
            nn.Conv2d(D//2, L, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        h_sse = self.downsample_layers[0](x)
        
        for l in range(self.L_SSE):
            # Downsampling layer
            h_ds = self.downsample_layers[l+1](h_sse)
            
            # Depth-wise Channel Selective Module
            h_dcs = self.dcsm_modules[l](h_ds)
            
            # Spatio-Temporal State Space Model
            h_sse = self.stssm_modules[l](h_dcs)
        
        # Merge temporal and feature dimensions
        # h_sse shape: [batch, D, 10, H, W]
        batch_size = h_sse.shape[0]
        h_sse = h_sse.permute(0, 1, 2, 3, 4)  # [batch, D, 10, H, W]
        h_sse = h_sse.reshape(batch_size, self.D * 10, h_sse.shape[3], h_sse.shape[4])  # [batch, D*10, H, W]
        
        # Apply 2D convolutions
        h_sse = self.conv_layers(h_sse)  # [batch, L, H, W]
        
        # Global pooling to remove spatial dimensions
        h_sse = h_sse.reshape(batch_size, self.L, self.D)
        
        return h_sse

class LongRangeTemporalSSM(nn.Module):
    def __init__(self, D=64, L=128, L_LT=2, dropout_rate=0.3):
        super().__init__()
        self.D = D   # Feature dimension
        self.L = L   # Sequence length in output
        
        # Use a single SSMBlock with depth=L_LT instead of multiple blocks
        self.ssm_block = SSMBlock(self.L, depth=L_LT, dropout_rate=dropout_rate)
        
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(self.L, self.L, kernel_size=5, stride=2, padding=2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Conv1d(self.L, self.L, kernel_size=5, stride=2, padding=2),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.D)

    def forward(self, x):
        # Process with SSM block (handles all L_LT layers internally)
        x = self.ssm_block(x)
        
        # Apply 1D convolutions for temporal reduction
        x = x.transpose(1, 2)  # [batch, D, seq_len]
        x = self.conv1d_layers(x)  # [batch, D, reduced_seq_len]
        x = self.dropout(x)
        
        # Ensure output has consistent sequence length L
        x = self.adaptive_pool(x)  # [batch, D, L]
        
        return x