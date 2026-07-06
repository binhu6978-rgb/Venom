import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Time2Vec 编码：线性项 + 正弦周期项。"""
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.freq   = nn.Linear(1, dim - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(-1)          # [B] -> [B, 1]
        elif t.dim() == 2 and t.shape[-1] != 1:
            t = t[:, :1]
        v1 = self.linear(t)              # [B, 1]
        v2 = torch.sin(self.freq(t))     # [B, dim-1]
        return torch.cat([v1, v2], dim=-1)


class LiquidNeuralUnit(nn.Module):
    """液态神经元单元（Liquid Time-Constant neuron）。"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w_in      = nn.Linear(input_dim, hidden_dim)
        self.w_rec     = nn.Linear(hidden_dim, hidden_dim)
        self.alpha     = nn.Parameter(torch.ones(1) * 0.1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor, h=None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.w_rec.out_features, device=x.device)
        h = h + self.alpha * (self.activation(self.w_in(x) + self.w_rec(h)) - h)
        return h


class GatedLiquidTimeEmbedding(nn.Module):
    """门控液态时间嵌入。"""
    def __init__(self, input_dim: int, hidden_dim: int, time_dim: int):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.gate        = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.liquid_unit = LiquidNeuralUnit(hidden_dim, time_dim)
        self.output_proj = nn.Linear(time_dim, time_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        g      = self.gate(x_proj)
        h      = self.liquid_unit(x_proj)
        return self.output_proj(g * h)


class LTEBlock(nn.Module):
    """
    Liquid Time Embedding Block。
    Time2Vec → GatedLiquidTimeEmbedding → 时间嵌入向量。
    """
    def __init__(self, dim: int, hidden_dim: int, time_dim: int):
        super().__init__()
        self.time2vec = Time2Vec(dim)
        self.lte      = GatedLiquidTimeEmbedding(dim, hidden_dim, time_dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.lte(self.time2vec(time))
