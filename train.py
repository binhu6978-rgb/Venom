"""
Step 3: Train Diff-LRN (UNet + Liquid Time Embedding) for image reconstruction.
每个 epoch 保存一次权重到 checkpoints/ 目录。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from configs import TrainConfig
from models  import Unet
from utils   import NoisyImageDataset, DiffusionLoss

cfg = TrainConfig()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for noisy, clean, t in tqdm(loader, desc='Train', leave=False):
        noisy, clean, t = noisy.to(device), clean.to(device), t.to(device)
        pred = model(noisy, t)
        loss = criterion(pred, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == '__main__':
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.ToTensor()

    root     = f'train_data_lenet/{cfg.dataset}/epsilons_{cfg.epsilon}'
    train_ds = NoisyImageDataset(
        f'./data/{cfg.dataset}/train/images',
        f'{root}/train_noise/labels_Linear',
        transform=transform,
        use_adaptive_t=cfg.use_adaptive_t
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )

    model     = Unet(channels=16).to(device)
    criterion = DiffusionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr,
        total_steps=cfg.num_epochs, pct_start=0.3
    )

    # 每个 epoch 的权重保存到 checkpoints/
    ckpt_dir = os.path.join(cfg.save_root, cfg.dataset, f'epsilons_{cfg.epsilon}', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        print(f'Epoch [{epoch+1:03d}/{cfg.num_epochs}]  Loss: {avg_loss:.4f}')

        # 每个 epoch 保存一次权重
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, f'epoch_{epoch+1:03d}.pth')
        )

        torch.cuda.empty_cache()

    print(f'\nDone. Checkpoints saved to: {ckpt_dir}')
