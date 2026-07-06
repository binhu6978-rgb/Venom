from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps    = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean   = reduce(weight, "o ... -> o 1 1 1", "mean")
        var    = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        return F.conv2d(x, (weight - mean) * (var + eps).rsqrt(),
                        self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act  = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
                    if exists(time_emb_dim) else None)
        self.block1   = Block(dim,     dim_out, groups=groups)
        self.block2   = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb    = self.mlp(time_emb)
            time_emb    = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        from torch import einsum
        self._einsum = einsum
        self.scale    = dim_head ** -0.5
        self.heads    = heads
        hidden_dim    = dim_head * heads
        self.to_qkv   = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out   = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        from torch import einsum
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q   = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out  = einsum("b h i j, b h d j -> b h i d", attn, v)
        out  = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale     = dim_head ** -0.5
        self.heads     = heads
        hidden_dim     = dim_head * heads
        self.to_qkv    = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out    = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q       = q.softmax(dim=-2) * self.scale
        k       = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out     = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out     = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn   = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class UpsampleBlock(nn.Module):
    """PixelShuffle ×2 上采样块。"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv          = nn.Conv2d(in_channels, in_channels * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu          = nn.ReLU()

    def forward(self, x):
        return self.relu(self.pixel_shuffle(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x
