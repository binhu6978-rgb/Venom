import numpy as np
import pywt


def wavelet_refine(x_2d: np.ndarray, wavelet: str = 'db4', level: int = 2) -> np.ndarray:
    """
    小波域软阈值精炼，抑制高频 DP 噪声，保留结构信号。

    Args:
        x_2d    : (H, W) 单通道特征图
        wavelet : 小波基，默认 'db4'
        level   : 分解层数，默认 2

    Returns:
        refined : (H, W) 精炼后特征图
    """
    coeffs    = pywt.wavedec2(x_2d, wavelet=wavelet, level=level)
    sigma_est = np.median(np.abs(coeffs[-1][0])) / 0.6745
    threshold = sigma_est * np.sqrt(2 * np.log(x_2d.size + 1e-8))

    coeffs_thresh = [coeffs[0]]
    for detail in coeffs[1:]:
        coeffs_thresh.append(
            tuple(pywt.threshold(d, threshold, mode='soft') for d in detail)
        )

    refined = pywt.waverec2(coeffs_thresh, wavelet=wavelet)
    return refined[:x_2d.shape[0], :x_2d.shape[1]]
