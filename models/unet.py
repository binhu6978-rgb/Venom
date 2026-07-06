from functools import partial

import torch
import torch.nn as nn

from .layers import (exists, default, Residual, Upsample, Downsample,
                     ResnetBlock, Attention, LinearAttention, PreNorm,
                     UpsampleBlock, ResidualBlock)
from .lte import LTEBlock


class Unet(nn.Module):
    """
    Diff-LRN 主网络。

    输入:
        x    : [B, channels, 64, 64]   SPE 提取的特征（默认 channels=16）
        time : [B] long                噪声强度 t（由 Dataset 估计）

    输出:
        [B, 3, 256, 256]               重建 RGB 图像
    """
    def __init__(
        self,
        dim               = 64,
        init_dim          = None,
        out_dim           = None,
        dim_mults         = (1, 2, 4),
        channels          = 16,
        self_condition    = False,
        resnet_block_groups = 4,
    ):
        super().__init__()

        self.channels      = channels
        self.self_condition = self_condition
        input_channels     = channels * (2 if self_condition else 1)

        init_dim       = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims   = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim    = dim * 4

        # 液态时间嵌入
        self.time_embed = LTEBlock(dim=dim, hidden_dim=time_dim, time_dim=time_dim)

        # 编码器
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in,  time_emb_dim=time_dim),
                block_klass(dim_in, dim_in,  time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        # 中间层
        mid_dim         = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 解码器
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        # 最终残差块
        self.out_dim        = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv1     = nn.Conv2d(dim, self.out_dim, 1)

        # 上采样路径：64×64 → 256×256
        self.initial_conv = nn.Sequential(nn.Conv2d(16, 64, 3, padding=1), nn.ReLU())
        self.res_blocks   = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.upsample1    = UpsampleBlock(64)   # 64×64  → 128×128
        self.upsample2    = UpsampleBlock(64)   # 128×128 → 256×256

        # 输出卷积：→ [B, 3, 256, 256]
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor,
                x_self_cond=None) -> torch.Tensor:
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_embed(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t); h.append(x)
            x = block2(x, t); x = attn(x); h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1); x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1); x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv1(x)

        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        return self.final_conv(x)
