import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    MSE + Cosine Similarity 混合损失。
    loss = 0.5 * MSE(pred, target) + 0.5 * (1 - CosineSim(pred, target))
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        flat_p   = pred.view(pred.size(0), -1)
        flat_t   = target.view(target.size(0), -1)
        cos_loss = 1 - F.cosine_similarity(flat_p, flat_t, dim=1).mean()
        return 0.5 * self.mse(pred, target) + 0.5 * cos_loss
