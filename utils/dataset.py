import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class NoisyImageDataset(Dataset):
    """
    加载 (noisy_feature, clean_image, t) 三元组。

    noisy_feature : SPE 提取的 .npy 文件，shape (C, H, W)
    clean_image   : 原始图像
    t             : 噪声强度估计（long），用于 Diff-LRN 时间嵌入
    """
    def __init__(self, clean_dir, noisy_dir, transform=None, use_adaptive_t=True):
        self.clean_dir      = clean_dir
        self.noisy_dir      = noisy_dir
        self.transform      = transform
        self.use_adaptive_t = use_adaptive_t
        self.clean_images   = sorted(os.listdir(clean_dir))
        self.noisy_files    = sorted(f for f in os.listdir(noisy_dir) if f.endswith('.npy'))

    def __len__(self):
        return len(self.clean_images)

    def _estimate_t(self, x_np: np.ndarray) -> int:
        """用通道均值近似估计噪声强度 T，量化为 [0, 100] 整数。"""
        c_np  = x_np.mean(axis=(1, 2), keepdims=True)
        num   = np.sum((x_np - c_np) ** 2)
        den   = np.sum(c_np ** 2) + 1e-8
        t_val = float(2.0 * np.log1p(num / den))
        return int(np.clip(round(t_val * 10.0), 0, 100))

    def __getitem__(self, idx):
        # clean image
        clean = Image.open(os.path.join(self.clean_dir, self.clean_images[idx]))
        if self.transform:
            clean = self.transform(clean)

        # noisy feature
        noisy = torch.tensor(
            np.load(os.path.join(self.noisy_dir, self.noisy_files[idx])),
            dtype=torch.float32
        ).squeeze(0)

        # t value
        t_idx = self._estimate_t(noisy.numpy()) if self.use_adaptive_t else 10

        return noisy, clean, torch.tensor(t_idx, dtype=torch.long)
