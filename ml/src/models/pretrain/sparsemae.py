# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified for Sparse MAE implementation in Deep Space Weather Model
# 
# This code is based on the original MAE (Masked Autoencoder) implementation 
# from Meta. The main modifications include:
# - Sparse masking strategies for solar images (sunspot-aware masking)
# - Two-phase masking approach (spatial and feature masking)
# - Standard deviation-based masking for preserving critical solar features
# 
# Original References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional

from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath, Attention


from src.models.pretrain.mae_module.pos_embed import get_2d_sincos_pos_embed


import torch
from torch import nn
import torch.nn.functional as F

from timm.layers import to_2tuple, make_divisible, trunc_normal_


def rel_pos_indices(size):
    size = to_2tuple(size)
    pos = torch.stack(
        torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))
    ).flatten(1)
    rel_pos = pos[:, None, :] - pos[:, :, None]
    rel_pos[0] += size[0] - 1
    rel_pos[1] += size[1] - 1
    return rel_pos  # 2, H * W, H * W


class LambdaLayer(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        feat_size=None,
        stride=1,
        num_heads=4,
        dim_head=16,
        r=9,
        qk_ratio=1.0,
        qkv_bias=False,
        linear=False,
    ):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0, " should be divided by num_heads"
        self.dim_qk = (
            dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        )
        self.num_heads = num_heads
        self.dim_v = dim_out // num_heads
        self.linear = linear

        self.qkv = nn.Conv2d(
            dim,
            num_heads * self.dim_qk + self.dim_qk + self.dim_v,
            kernel_size=1,
            bias=qkv_bias,
        )
        self.norm_q = nn.BatchNorm2d(num_heads * self.dim_qk)
        self.norm_v = nn.BatchNorm2d(self.dim_v)
        self.conv_lambda = None

        if r is not None:
            # local lambda convolution for pos
            if not self.linear:
                self.conv_lambda = nn.Conv3d(
                    1, self.dim_qk, (r, r, 1), padding=(r // 2, r // 2, 0)
                )

            self.pos_emb = None
            self.rel_pos_indices = None
        else:
            # relative pos embedding
            assert feat_size is not None
            feat_size = to_2tuple(feat_size)
            rel_size = [2 * s - 1 for s in feat_size]
            self.conv_lambda = None
            self.pos_emb = nn.Parameter(
                torch.zeros(rel_size[0], rel_size[1], self.dim_qk)
            )
            self.register_buffer(
                "rel_pos_indices", rel_pos_indices(feat_size), persistent=False
            )

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)  # fan-in
        if self.conv_lambda is not None:
            trunc_normal_(self.conv_lambda.weight, std=self.dim_qk**-0.5)
        if self.pos_emb is not None:
            trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        M = H * W
        qkv = self.qkv(x)
        q, k, v = torch.split(
            qkv, [self.num_heads * self.dim_qk, self.dim_qk, self.dim_v], dim=1
        )
        q = (
            self.norm_q(q).reshape(B, self.num_heads, self.dim_qk, M).transpose(-1, -2)
        )  # B, num_heads, M, K
        v = self.norm_v(v).reshape(B, self.dim_v, M).transpose(-1, -2)  # B, M, V
        k = F.softmax(k.reshape(B, self.dim_qk, M), dim=-1)  # B, K, M

        content_lam = k @ v  # B, K, V
        content_out = q @ content_lam.unsqueeze(1)  # B, num_heads, M, V

        if self.linear:
            out = content_out.transpose(-1, -2).reshape(
                B, C, H, W
            )  # B, C (num_heads * V), H, W
            out = self.pool(out)
            return out

        if self.pos_emb is None:
            position_lam = self.conv_lambda(
                v.reshape(B, 1, H, W, self.dim_v)
            )  # B, H, W, V, K
            position_lam = position_lam.reshape(
                B, 1, self.dim_qk, H * W, self.dim_v
            ).transpose(
                2, 3
            )  # B, 1, M, K, V
        else:
            # FIXME relative pos embedding path not fully verified
            pos_emb = self.pos_emb[
                self.rel_pos_indices[0], self.rel_pos_indices[1]
            ].expand(B, -1, -1, -1)
            position_lam = (pos_emb.transpose(-1, -2) @ v.unsqueeze(1)).unsqueeze(
                1
            )  # B, 1, M, K, V
        position_out = (q.unsqueeze(-2) @ position_lam).squeeze(
            -2
        )  # B, num_heads, M, V

        out = (
            (content_out + position_out).transpose(-1, -2).reshape(B, C, H, W)
        )  # B, C (num_heads * V), H, W
        out = self.pool(out)
        return out


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        baseline="attn",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if baseline == "attn":
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif baseline == "lambda" or baseline == "linear":
            self.attn = LambdaLayer(
                dim, num_heads=num_heads, qk_ratio=1.0, linear=(baseline == "linear")
            )
        else:
            assert False

        self.baseline = baseline
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.baseline == "attn":
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        elif self.baseline == "lambda" or self.baseline == "linear":
            B, L, D = x.shape
            z = self.norm1(x)
            z = z.transpose(-1, -2).unsqueeze(2)
            z = self.attn(z)
            z = z.squeeze(2).transpose(-1, -2)

            x = x + self.drop_path1(self.ls1(z))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        else:
            assert False

        return x


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class LambdaBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        num_heads: int = 8,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.lamb = LambdaLayer(width, num_heads=num_heads, qk_ratio=1.0, linear=False)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        out = self.lamb(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LambdaResnet(nn.Module):  # Reference CNNModel in train.py
    def __init__(self, output_channel=4, size=2, pretrain=False):
        super().__init__()

        self.pretrain = pretrain
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        downsample = nn.Sequential(
            conv1x1(16, 8 * 4, 1),
            nn.BatchNorm2d(8 * 4),
        )
        self.layer1 = LambdaBottleneck(
            16, 8, 1, downsample=downsample, norm_layer=nn.BatchNorm2d
        )

        self.avgpool = nn.AdaptiveAvgPool2d((size, size))
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Linear(32 * size * size, 32)
        self.fc2 = nn.Linear(32, output_channel)
        self.bn3 = nn.BatchNorm2d(8 * 4)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        if not self.pretrain:
            return x

        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class SparseMAE(nn.Module):
    """Sparse Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        baseline="attn",
        mask_ratio=0.75,
        stdwise=False,
        pyramid=False,
        sunspot=False,
        inner_mask_ratio=0.75,
        outer_mask_ratio=0.90,
        sunspot_ratio=0.90,
        solar_radius=0.45,
        base_mask_ratio=0.75,
        sunspot_spatial_ratio=0.9,
        feature_mask_ratio=0.25,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        if baseline == "lambda_resnet":
            self.blocks = nn.ModuleList(
                [
                    LambdaResnet(embed_dim),
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                        baseline=baseline,
                    )
                    for i in range(depth)
                ]
            )

        self.baseline = baseline
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_baseline = baseline if baseline != "lambda_resnet" else "attn"
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    baseline=decoder_baseline,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.mask_ratio = mask_ratio
        self.stdwise = stdwise
        self.pyramid = pyramid
        self.norm_pix_loss = norm_pix_loss
        self.sunspot = sunspot
        self.inner_mask_ratio = inner_mask_ratio
        self.outer_mask_ratio = outer_mask_ratio
        self.sunspot_ratio = sunspot_ratio
        self.solar_radius = solar_radius

        self.base_mask_ratio = base_mask_ratio
        self.sunspot_spatial_ratio = sunspot_spatial_ratio
        self.feature_mask_ratio = feature_mask_ratio

        self.initialize_weights()

    def set_train_flag_encoeder(self):
        self.patch_embed.train()
        self.blocks.train()
        self.norm.train()

        del self.decoder_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred
        import gc

        gc.collect()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def patchify_dim3(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify_dim3(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def patchify_dim13(self, imgs):
        """
        imgs: (N, 13, H, W)
        x: (N, L, patch_size**2 * 13)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 13, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 13))
        return x

    def unpatchify_dim13(self, x):
        """
        x: (N, L, patch_size**2 * 13)
        imgs: (N, 13, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 13))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 13, h * p, h * p))
        return imgs

    def patchify_dim10(self, imgs):
        """
        imgs: (N, 10, H, W)
        x: (N, L, patch_size**2 * 10)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 10, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 10))
        return x

    def unpatchify_dim10(self, x):
        """
        x: (N, L, patch_size**2 * 10)
        imgs: (N, 10, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 10))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 10, h * p, h * p))
        return imgs

    def random_masking_vit(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_pyramid_masking_vit(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        len_keep_dim = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x_not_masked = torch.gather(
            x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        noise_dim = torch.rand(D, device=x.device)
        ids_shuffle_dim = torch.argsort(noise_dim)
        ids_restore_dim = torch.argsort(ids_shuffle_dim)

        ids_keep_dim = ids_shuffle_dim[:len_keep_dim]
        ids_not_keep_dim = ids_shuffle_dim[len_keep_dim:]

        x_not_masked[:, :, ids_not_keep_dim] = 0

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask[:, len_keep:] = mask_ratio
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, x_not_masked, mask, ids_restore

    def stdwise_masking_vit(self, x, stds, mask_ratio):
        """
        Perform per-sample std-wise masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # sort stds for each sample
        # descend: large is keep, small is remove
        ids_shuffle = torch.argsort(stds, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # mask = torch.gather(mask, dim=2, index=ids_restore_dim)

        return x_masked, mask, ids_restore

    def random_masking_resnet(self, x, mask_ratio):
        B, C, H, W = x.shape
        L = H * W
        x = x.view(B, C, L).transpose(-1, -2)
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        B, l, C = x_masked.shape
        h = int(np.sqrt(l))
        w = l // h
        assert h == w
        x_masked = x_masked.transpose(-1, -2).view(B, C, h, w)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder_vit(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking_vit(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_vit_std(self, x, mask_ratio):  # x: (B,C,H,W)
        # embed patches
        stds = torch.std(self.patchify_dim10(x), dim=2, unbiased=False)

        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.stdwise_masking_vit(x, stds, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_vit_pyramid(
        self,
        x,
        mask_ratio,
        base_mask_ratio=0.75,
        sunspot_spatial_ratio=0.25,
        feature_mask_ratio=0.75,
    ):  # x: (B,C,H,W)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.sunspot:
            x_masked, x_not_masked, mask, ids_restore = self.sparse_mae_masking_vit(
                x, base_mask_ratio, sunspot_spatial_ratio, feature_mask_ratio
            )
        else:
            x_masked, x_not_masked, mask, ids_restore = self.random_pyramid_masking_vit(
                x, mask_ratio
            )
        x = torch.cat([x_masked, x_not_masked], dim=1)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_lambda(self, x, mask_ratio):  # x: (B,C,H,W)
        x = self.patch_embed(x)  # (B,H*W/patch**2,embed_dim)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking_vit(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_lambda_resnet(self, x, mask_ratio):  # x: (B,C,H,W)
        # masking: length -> length * mask_ratio
        print(x.shape)
        x, mask, ids_restore = self.random_masking_resnet(x, mask_ratio)

        print(x.shape)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        print(x.shape)
        x = x.unsqueeze(1)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        if self.baseline == "attn":
            return self.forward_encoder_vit(x, mask_ratio)
        elif self.baseline == "lambda" or self.baseline == "linear":
            return self.forward_encoder_lambda(x, mask_ratio)
        elif self.baseline == "lambda_resnet":
            return self.forward_encoder_lambda_resnet(x, mask_ratio)
        else:
            assert False

    def forward_encoder_pyramid(
        self,
        x,
        mask_ratio,
        base_mask_ratio=0.75,
        sunspot_spatial_ratio=0.35,
        feature_mask_ratio=0.5,
    ):
        if self.baseline == "attn":
            if self.sunspot:
                return self.forward_encoder_vit_pyramid(
                    x,
                    mask_ratio,
                    base_mask_ratio=base_mask_ratio,
                    sunspot_spatial_ratio=sunspot_spatial_ratio,
                    feature_mask_ratio=feature_mask_ratio,
                )
            else:
                return self.forward_encoder_vit_pyramid(x, mask_ratio)
        elif self.baseline == "lambda" or self.baseline == "linear":
            return self.forward_encoder_lambda(x, mask_ratio)
        elif self.baseline == "lambda_resnet":
            return self.forward_encoder_lambda_resnet(x, mask_ratio)
        else:
            assert False

    def forward_encoder_std(self, x, mask_ratio):
        if self.baseline == "attn":
            return self.forward_encoder_vit_std(x, mask_ratio)
        elif self.baseline == "lambda" or self.baseline == "linear":
            return self.forward_encoder_lambda(x, mask_ratio)
        elif self.baseline == "lambda_resnet":
            return self.forward_encoder_lambda_resnet(x, mask_ratio)
        else:
            assert False

    def forward_encoder_sunspot(self, x, mask_ratio):
        if self.baseline == "attn":
            return self.forward_encoder_vit_sunspot(x, mask_ratio)
        elif self.baseline == "lambda" or self.baseline == "linear":
            return self.forward_encoder_lambda(x, mask_ratio)
        elif self.baseline == "lambda_resnet":
            return self.forward_encoder_lambda_resnet(x, mask_ratio)
        else:
            assert False

    def forward_encoder_vit_sunspot(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.sunspot_masking_vit(
            x, self.inner_mask_ratio, self.outer_mask_ratio
        )

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def sunspot_masking_vit(
        self, x, sunspot_ratio=0.9, inner_ratio=0.75, outer_ratio=0.85
    ):
        """
        Calculate standard deviation for each channel independently and identify sunspot regions for masking
        Args:
            x: [N, L, D], sequence
        """
        N, L, D = x.shape
        device = x.device

        h = w = int(np.sqrt(L))
        center = h // 2
        y, x_coords = np.meshgrid(np.arange(h), np.arange(w))
        dist = np.sqrt((x_coords - center) ** 2 + (y - center) ** 2) / (
            np.sqrt(2) * center
        )
        dist = torch.tensor(dist.flatten(), device=device)

        # Inner region mask
        inner_mask = dist <= 0.65

        # Prepare indices for each region
        len_keep = int(L * (1 - inner_ratio))  # Base number to keep

        # Calculate standard deviation for each patch
        x_reshaped = x.reshape(N, h, w, -1)  # [N, h, w, p*p*C]
        patch_stds = torch.std(x_reshaped, dim=-1)  # [N, h, w]
        patch_stds = patch_stds.reshape(N, -1)  # [N, L]

        # Initialize mask and restore indices
        mask = torch.ones([N, L], device=device)
        ids_restore = torch.arange(L, device=device).repeat(N, 1)

        # Process each sample in the batch
        x_masked_list = []

        for i in range(N):
            # Standard deviation in inner region
            inner_stds = patch_stds[i][inner_mask]
            std_threshold = torch.quantile(inner_stds, 0.9)

            # Get indices for each region
            sunspot_indices = torch.where(
                (patch_stds[i] >= std_threshold) & inner_mask
            )[0]
            non_sunspot_inner = torch.where(
                (patch_stds[i] < std_threshold) & inner_mask
            )[0]
            outer_indices = torch.where(~inner_mask)[0]

            # Calculate number to keep for each region
            len_keep_sunspot = int(len(sunspot_indices) * (1 - sunspot_ratio))
            len_keep_inner = int(len(non_sunspot_inner) * (1 - inner_ratio))
            len_keep_outer = int(len(outer_indices) * (1 - outer_ratio))

            # Randomly select indices to keep
            perm_sunspot = torch.randperm(len(sunspot_indices), device=device)
            keep_sunspot = sunspot_indices[perm_sunspot[:len_keep_sunspot]]

            perm_inner = torch.randperm(len(non_sunspot_inner), device=device)
            keep_inner = non_sunspot_inner[perm_inner[:len_keep_inner]]

            perm_outer = torch.randperm(len(outer_indices), device=device)
            keep_outer = outer_indices[perm_outer[:len_keep_outer]]

            # Combine indices to keep
            keep_indices = torch.cat([keep_sunspot, keep_inner, keep_outer])
            mask[i, keep_indices] = 0

            # Get masked sequence
            x_masked = torch.gather(x[i], 0, keep_indices.unsqueeze(-1).repeat(1, D))
            x_masked_list.append(x_masked)

        # Pad to same size
        max_len = max([x.size(0) for x in x_masked_list])
        x_masked_padded = []
        for x_masked in x_masked_list:
            if x_masked.size(0) < max_len:
                padding = torch.zeros(max_len - x_masked.size(0), D, device=device)
                x_masked = torch.cat([x_masked, padding], dim=0)
            x_masked_padded.append(x_masked)

        x_masked = torch.stack(x_masked_padded)

        return x_masked, mask, ids_restore

    def sparse_mae_masking_vit(
        self,
        x,
        base_mask_ratio=0.75,
        sunspot_spatial_ratio=0.35,
        feature_mask_ratio=0.25,
    ):
        """
        Simple extension of random_pyramid_masking_vit.
        Identifies sunspot regions as the top 20% of standard deviation for each channel,
        and applies a lower mask ratio to these regions.
        """
        N, L, D = x.shape
        device = x.device

        # Calculate basic len_keep
        len_keep = int(L * (1 - base_mask_ratio))

        # Identify sunspot regions for each channel
        patch_stds = torch.std(x, dim=-1)  # [N, L]
        threshold = torch.quantile(patch_stds, 0.8, dim=1, keepdim=True)
        is_sunspot = patch_stds >= threshold

        # Initialize mask (1 is mask, 0 is keep)
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]

        # Apply different mask ratios to sunspot regions and other regions
        for i in range(N):
            # Adjust noise for sunspot regions (more likely to be kept)
            noise[i, is_sunspot[i]] *= sunspot_spatial_ratio
            noise[i, ~is_sunspot[i]] *= base_mask_ratio

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Feature dimension masking
        len_keep_dim = int(D * (1 - feature_mask_ratio))
        noise_dim = torch.rand(D, device=device)
        ids_shuffle_dim = torch.argsort(noise_dim)
        x_masked[:, :, ids_shuffle_dim[len_keep_dim:]] = 0

        # Generate masked part
        x_not_masked = torch.zeros_like(x)
        ids_not_keep = ids_shuffle[:, len_keep:]
        x_not_masked_temp = torch.gather(
            x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        x_not_masked_temp[:, :, ids_shuffle_dim[len_keep_dim:]] = 0

        return x_masked, x_not_masked_temp, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_pyramid(self, x, x_not_masked, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        x_ = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        mask_sum = mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=mask.device)

        loss = (loss * mask).sum() / mask_sum 
        return loss

    def forward_loss_huber(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        # huber loss
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_loss_pyramid(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L, D], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = (loss * mask).sum / mask.sum()
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_loss_sparse(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        loss = loss.mean()

        return loss

    def forward(self, imgs):
        if self.sunspot:
            if self.pyramid:
                latent, mask, ids_restore = self.forward_encoder_pyramid(
                    imgs,
                    self.mask_ratio,
                    base_mask_ratio=self.base_mask_ratio,
                    sunspot_spatial_ratio=self.sunspot_spatial_ratio,
                    feature_mask_ratio=self.feature_mask_ratio,
                )
            else:
                latent, mask, ids_restore = self.forward_encoder_sunspot(
                    imgs, self.mask_ratio
                )
        elif self.stdwise:
            latent, mask, ids_restore = self.forward_encoder_std(imgs, self.mask_ratio)
        elif self.pyramid:
            latent, mask, ids_restore = self.forward_encoder_pyramid(
                imgs, self.mask_ratio
            )
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)

        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask

    def get_loss(self, imgs, pred, mask):
        if self.mask_ratio > 0:
            return self.forward_loss(imgs, pred, mask)
        else:
            return self.forward_loss_sparse(imgs, pred, mask)

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = SparseMAE(
        patch_size=16,
        embed_dim=64,
        depth=12,
        num_heads=8,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = SparseMAE(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_for_FT512d8b(embed_dim=512, **kwargs):
    model = SparseMAE(
        patch_size=8,
        embed_dim=embed_dim,
        depth=12,
        num_heads=8, 
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_for_FT64d4b(embed_dim=64, patch_size=8, **kwargs):
    model = SparseMAE(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=4,
        num_heads=8, 
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_for_FT64d1b(embed_dim=64, patch_size=8, **kwargs):
    model = SparseMAE(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=1,
        num_heads=8, 
        decoder_embed_dim=128,
        decoder_depth=1,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def vit_for_FT128db(embed_dim=128, **kwargs):
    model = SparseMAE(
        patch_size=8,
        embed_dim=embed_dim,
        depth=8,
        num_heads=8, 
        decoder_embed_dim=128,
        decoder_depth=12,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = SparseMAE(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b 
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b 
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b

vit_for_FT = vit_for_FT64d4b  


def vit_for_FT32d4b(embed_dim=32, patch_size=8, **kwargs):
    """
    embed_dim=32 small model for Deep Space Weather Model
    - embed_dim: 32 (divisible by num_heads=8)
    - depth: 4
    - decoder_embed_dim: 64 (reduced from original 128)
    """
    model = SparseMAE(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=4,
        num_heads=8, 
        decoder_embed_dim=64, 
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
