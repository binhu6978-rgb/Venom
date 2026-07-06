<<<<<<< HEAD
# Venom: Gradient Inversion Attack under Differential Privacy

PyTorch implementation of a gradient inversion attack for federated learning with differential privacy, based on [Venom (AAAI 2026)](https://arxiv.org/abs/xxxx) and [Mjölnir (AAAI 2025)](https://arxiv.org/abs/2407.05285).

## Requirements

```bash
pip install torch torchvision numpy Pillow PyWavelets tqdm matplotlib
```

## Project Structure

```
├── 1_get_data.py               # Download and save dataset
├── 2_generate_features.py      # SPE feature extraction with DP noise
├── 3_train.py                  # Train Diff-LRN reconstruction network
│
├── configs/
│   └── config.py               # Centralized configuration
│
├── models/
│   ├── lenet.py      # Victim model (LeNet)
│   ├── layers.py     # Basic building blocks (ResnetBlock, Attention, etc.)
│   ├── lte.py        # Liquid Time Embedding (Time2Vec + LiquidNeuralUnit)
│   └── unet.py       # Diff-LRN main network (UNet + LTE)
│
└── utils/
    ├── seed.py                 # Random seed control
    ├── dp_noise.py             # Mjölnir-style DP noise injection
    ├── wavelet.py              # Wavelet soft-thresholding refinement
    ├── spe.py                  # Structural Prior Extraction (SPE)
    ├── dataset.py              # NoisyImageDataset
    ├── loss.py                 # DiffusionLoss (MSE + Cosine)
    ├── metrics.py              # PSNR computation
    └── visualize.py            # Visualization utilities
```

## Usage

**Step 1: Prepare data**

```bash
python get_data.py
```

Edit `DataConfig.dataset` in `configs/config.py` to switch between `cifar10` and `mnist`.

**Step 2: Generate SPE features**

```bash
python generate_features.py
```

Key settings in `configs/config.py`:

| Config     | Field         | Description                                         |
|-----------|---------------|-----------------------------------------------------|
| `DPConfig` | `epsilons`    | DP privacy budget ε ∈ {1, 5, 10}                   |
| `DPConfig` | `dp_c`        | Gradient clipping threshold C                       |
| `DPConfig` | `lr`          | Local learning rate (Mjölnir-style sensitivity)     |
| `SPEConfig`| `mode`        | `'spe'` = full SPE + wavelet, `'direct'` = raw FC division |

**Step 3: Train**

```bash
python train.py
```

Key settings:

| Config       | Field             | Description                                  |
|-------------|-------------------|----------------------------------------------|
| `TrainConfig`| `use_adaptive_t`  | `True` = adaptive T per sample, `False` = fixed t=10 |
| `TrainConfig`| `epsilon`         | Must match Step 2                            |
| `TrainConfig`| `psnr_target`     | Early stop when PSNR reaches this value      |


## Citation

```bibtex
@inproceedings{venom2026,
  title     = {Venom: Liquid Diffusion-Guided Gradient Inversion for Breaking Differential Privacy in Federated Learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}

@inproceedings{mjolnir2025,
  title     = {Mjölnir: Breaking the Shield of Perturbation-Protected Gradients via Adaptive Diffusion},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```
=======
# Venom: Liquid Diffusion-Guided Gradient Inversion for Breaking Differential Privacy in Federated Learning

<p align="center">
  <img src="assets/network.png" width="800"/>
</p>

> **Venom: Liquid Diffusion-Guided Gradient Inversion for Breaking Differential Privacy in Federated Learning**  
> Bin Hu, Jingling Yuan, Jiawei Jiang, Chuang Hu  
> *AAAI 2026*

> **Note**  
> This repository is currently under preparation. The codebase is being cleaned and organized, and will be open-sourced soon.

---

## Overview

Venom is a novel two-stage gradient inversion framework that reconstructs private training data directly from **DP-protected gradients** without requiring any prior knowledge of the noise distribution.

**Stage 1 — Structural Prior Extraction (SPE)**  
Analytically recovers deep feature representations from perturbed gradients via energy-based multi-class aggregation and wavelet-domain structural refinement.

**Stage 2 — Diffusion-driven Liquid Recovery Network (Diff-LRN)**  
A one-step deterministic diffusion model enhanced with liquid neural dynamics (LTE module) that adapts its denoising behavior to unknown and heterogeneous DP noise.

---

## Project Status

Code will be uploaded soon.
>>>>>>> 6b236a46b7fb8f947b8a95837317a823b633406c
