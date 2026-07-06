"""
Step 2: Extract SPE features from LeNet FC gradients with DP noise.
Output: .npy feature files used for Diff-LRN training.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from configs import DPConfig, SPEConfig
from models  import LeNet, weights_init
from utils   import (set_seed, add_dp_noise_to_fc,
                     spe_extract, fc_direct_extract,
                     feature_to_rgb)

dp_cfg  = DPConfig()
spe_cfg = SPEConfig()


def load_image(path: str) -> torch.Tensor:
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    return transform(Image.open(path).convert('RGB')).unsqueeze(0)


if __name__ == '__main__':
    for data_name in [spe_cfg.__class__.__name__]:
        data_name = 'cifar10'   # edit here to switch dataset
        for dp_epsilon in dp_cfg.epsilons:
            for split in ['train', 'test']:

                image_dir   = os.path.join('data', data_name, split, 'images')
                labels_path = os.path.join('data', data_name, split, 'labels.npy')

                all_labels      = np.load(labels_path) if os.path.exists(labels_path) else None
                use_real_labels = all_labels is not None

                save_dir  = os.path.join(spe_cfg.save_root, data_name,
                                         f'epsilons_{dp_epsilon}', f'{split}_noise')
                label_dir = os.path.join(save_dir, 'labels_Linear')
                vis_dir   = os.path.join(save_dir, 'vis')
                os.makedirs(label_dir, exist_ok=True)
                if spe_cfg.save_vis:
                    os.makedirs(vis_dir, exist_ok=True)

                print(f'\n[{data_name}] ε={dp_epsilon} | {split} | mode={spe_cfg.mode}')

                files = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))
                count = 0

                for file in tqdm(files):
                    if spe_cfg.max_samples and count >= spe_cfg.max_samples:
                        break

                    set_seed(spe_cfg.seed + count)

                    try:
                        image = load_image(os.path.join(image_dir, file))
                    except Exception as e:
                        print(f'  [Skip] {file}: {e}')
                        continue

                    if use_real_labels:
                        img_idx = int(os.path.splitext(file)[0])
                        target  = torch.tensor([all_labels[img_idx]], dtype=torch.long)
                    else:
                        target = torch.randn(1, 10)

                    model = LeNet()
                    model.apply(weights_init)
                    model.eval()

                    nn.CrossEntropyLoss()(model(image), target).backward()

                    fc_wg = model.fc.weight.grad.detach().clone()
                    fc_bg = model.fc.bias.grad.detach().clone()

                    fc_wg_n, fc_bg_n, sigma2 = add_dp_noise_to_fc(
                        fc_wg, fc_bg,
                        dp_epsilon=dp_epsilon,
                        dp_delta=dp_cfg.dp_delta,
                        dp_c=dp_cfg.dp_c,
                        mechanism=dp_cfg.mechanism,
                        lr=dp_cfg.lr
                    )

                    if spe_cfg.mode == 'spe':
                        x_feat = spe_extract(fc_wg_n, fc_bg_n, sigma2, spe_cfg.reshape,
                                             wavelet=spe_cfg.wavelet, level=spe_cfg.wavelet_level)
                    else:
                        x_feat = fc_direct_extract(fc_wg_n, fc_bg_n, spe_cfg.reshape)

                    stem = os.path.splitext(file)[0]
                    np.save(os.path.join(label_dir, f'{stem}.npy'), x_feat)
                    if spe_cfg.save_vis:
                        Image.fromarray(feature_to_rgb(x_feat)).save(
                            os.path.join(vis_dir, f'{stem}.png'))
                    count += 1

                print(f'  Saved {count} samples → {label_dir}')
