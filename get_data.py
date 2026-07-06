"""
Step 1: Download dataset and save images + labels.npy
"""
import os
import numpy as np
from torchvision import datasets
from PIL import Image
from tqdm import tqdm

from configs import DataConfig

cfg = DataConfig()


def save_dataset(dataset, save_dir, split_name, stop):
    image_dir = os.path.join(save_dir, split_name, 'images')
    os.makedirs(image_dir, exist_ok=True)
    labels = []
    for idx, (img, label) in enumerate(tqdm(dataset, desc=f'Saving {split_name}')):
        if idx == stop:
            break
        img.resize((256, 256), Image.BILINEAR).save(os.path.join(image_dir, f'{idx:05d}.png'))
        labels.append(int(label))
    np.save(os.path.join(save_dir, split_name, 'labels.npy'), np.array(labels, dtype=np.int64))
    print(f'  {len(labels)} samples → {image_dir}')


if __name__ == '__main__':
    save_root = os.path.join(cfg.save_root, cfg.dataset)

    if cfg.dataset == 'cifar10':
        train_ds = datasets.CIFAR10(root='~/.torch', train=True,  download=True)
        test_ds  = datasets.CIFAR10(root='~/.torch', train=False, download=True)
    elif cfg.dataset == 'mnist':
        train_ds = datasets.MNIST(root='~/.torch', train=True,  download=True)
        test_ds  = datasets.MNIST(root='~/.torch', train=False, download=True)
    else:
        raise ValueError(f'Unsupported dataset: {cfg.dataset}')

    save_dataset(train_ds, save_root, 'train', cfg.train_stop)
    save_dataset(test_ds,  save_root, 'test',  cfg.test_stop)
    print('Done.')
