import numpy as np
import torch

from .wavelet import wavelet_refine


def spe_extract(fc_weight_grad, fc_bias_grad, sigma2, reshape_target,
                wavelet='db4', level=2):
    """
    Structural Prior Extraction
    Args:
        fc_weight_grad : Tensor [K, d]
        fc_bias_grad   : Tensor [K]
        sigma2         : float, DP 噪声方差
        reshape_target : tuple, e.g. (16, 64, 64)

    Returns:
        x_refined : np.ndarray, shape=reshape_target
    """
    K = fc_bias_grad.shape[0]

    E = torch.norm(fc_weight_grad, dim=1) * fc_bias_grad.abs()

    mu_E, sigma_E = E.mean(), E.std()
    gamma  = min(2.0, float(torch.log(torch.tensor(float(K))).sqrt()))
    S_mask = E > (mu_E + gamma * sigma_E)
    if S_mask.sum() == 0:
        S_mask = torch.zeros_like(E, dtype=torch.bool)
        S_mask[E.argmax()] = True

    lambda_reg = sigma2 / (fc_bias_grad[S_mask].abs().max().item() ** 2 + 1e-8)
    E_S   = E[S_mask]
    w     = E_S / (E_S.sum() + 1e-8)
    denom = fc_bias_grad[S_mask].unsqueeze(1) + lambda_reg * sigma2
    x_hat = (w.unsqueeze(1) * (fc_weight_grad[S_mask] / (denom + 1e-8))).sum(dim=0)

    x_np      = x_hat.detach().cpu().numpy().reshape(reshape_target)
    C, H, W   = reshape_target
    x_refined = np.zeros_like(x_np)
    for c in range(C):
        x_refined[c] = wavelet_refine(x_np[c], wavelet=wavelet, level=level)
    return x_refined


def fc_direct_extract(fc_weight_grad, fc_bias_grad, reshape_target):
    """
    直接 FC 相除（对比实验用）：x̂ = ∇W_j / ∇b_j
    Args:
        fc_weight_grad : Tensor [K, d]
        fc_bias_grad   : Tensor [K]
        reshape_target : tuple, e.g. (16, 64, 64)

    Returns:
        x_raw : np.ndarray, shape=reshape_target
    """
    j_star   = fc_bias_grad.abs().argmax()
    bias_j   = fc_bias_grad[j_star]
    weight_j = fc_weight_grad[j_star]
    sign     = bias_j.sign() if bias_j.sign() != 0 else torch.tensor(1.0)
    x_hat    = weight_j / (bias_j + 1e-8 * sign)
    return x_hat.detach().cpu().numpy().reshape(reshape_target)
