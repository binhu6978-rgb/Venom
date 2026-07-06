import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    LeNet backbone，用于生成 SPE 训练数据。
    输入：(B, 3, 256, 256)
    FC 输入维度：16 * 64 * 64 = 65536
    """
    def __init__(self, num_classes: int = 10, fc_dim: int = 65536):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6,  kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.fc    = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.view(x.size(0), -1))


def weights_init(m):
    """U(-0.5, 0.5) 均匀初始化。"""
    if hasattr(m, 'weight') and m.weight is not None:
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.uniform_(-0.5, 0.5)
