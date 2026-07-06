import numpy as np
import torch


def clip_fc_gradients_l2(fc_weight_grad, fc_bias_grad, clip_threshold):
    """对 FC 层 weight 和 bias 梯度做联合 L2 clipping。"""
    total_norm = torch.sqrt(fc_weight_grad.pow(2).sum() + fc_bias_grad.pow(2).sum())
    clip_coef  = min(1.0, clip_threshold / (total_norm.item() + 1e-6))
    return fc_weight_grad * clip_coef, fc_bias_grad * clip_coef


def gaussian_noise_scale(dp_epsilon, dp_delta, dp_c, batch_size=1, lr=0.001):
    """
    Mjölnir-style Gaussian 噪声标准差。
    sensitivity = 2 * lr * C / m
    M = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    """
    sensitivity = 2.0 * lr * dp_c / batch_size
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / dp_delta)) / dp_epsilon


def laplace_noise_scale(dp_epsilon, dp_c, batch_size=1, lr=0.001):
    """
    Mjölnir-style Laplace 噪声 scale。
    sensitivity = 2 * lr * C / m
    b = sensitivity / epsilon
    """
    sensitivity = 2.0 * lr * dp_c / batch_size
    return sensitivity / dp_epsilon


def add_dp_noise_to_fc(fc_weight_grad, fc_bias_grad,
                        dp_epsilon, dp_delta, dp_c,
                        mechanism='gaussian', batch_size=1, lr=0.001):
    """
    FC 梯度 L2 clipping → Mjölnir-style DP 加噪。

    Returns:
        noisy_weight, noisy_bias, sigma2
    """
    fc_weight_grad, fc_bias_grad = clip_fc_gradients_l2(
        fc_weight_grad, fc_bias_grad, dp_c
    )

    if mechanism == 'gaussian':
        M            = gaussian_noise_scale(dp_epsilon, dp_delta, dp_c, batch_size, lr)
        noisy_weight = fc_weight_grad + torch.randn_like(fc_weight_grad) * M
        noisy_bias   = fc_bias_grad   + torch.randn_like(fc_bias_grad)   * M
        sigma2       = M ** 2

    elif mechanism == 'laplace':
        b = laplace_noise_scale(dp_epsilon, dp_c, batch_size, lr)
        noisy_weight = fc_weight_grad + torch.distributions.Laplace(0, b).sample(
            fc_weight_grad.shape).to(fc_weight_grad.device)
        noisy_bias   = fc_bias_grad   + torch.distributions.Laplace(0, b).sample(
            fc_bias_grad.shape).to(fc_bias_grad.device)
        sigma2       = 2.0 * b ** 2

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return noisy_weight, noisy_bias, sigma2
