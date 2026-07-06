import numpy as np


def feature_to_rgb(x_chw: np.ndarray) -> np.ndarray:
    """
    多通道特征图 → 可视化 RGB uint8 图像。
    取前3通道，逐通道 min-max 归一化。

    Args:
        x_chw : (C, H, W)
    Returns:
        rgb   : (H, W, 3) uint8
    """
    C   = x_chw.shape[0]
    vis = x_chw[:3] if C >= 3 else np.stack([x_chw[0]] * 3, axis=0)
    rgb = np.zeros_like(vis, dtype=np.float32)
    for i in range(3):
        ch = vis[i]
        vmin, vmax = ch.min(), ch.max()
        if vmax - vmin > 1e-8:
            rgb[i] = (ch - vmin) / (vmax - vmin)
    return (rgb * 255).astype(np.uint8).transpose(1, 2, 0)
