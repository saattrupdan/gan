import torch
from torch import nn
import torch.nn.functional as F

from utils import mish

class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mid_dim = out_dim // 2
        self.conv1 = nn.Conv1d(in_dim, mid_dim, kernel_size = 9, padding = 4)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.conv2 = nn.Conv1d(mid_dim, out_dim, kernel_size = 5, padding = 2)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.conv3 = nn.Conv1d(out_dim, out_dim, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = mish(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = mish(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = mish(x)

        return x + input

class TSCResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResBlock(1, 128)
        self.block2 = ResBlock(128, 128)
        self.block3 = ResBlock(128, 128)
        self.proj = nn.Linear(128, 7)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.mean(x, dim = 2)
        x = self.proj(x)
        return torch.softmax(x, dim = 1)

if __name__ == '__main__':
    model = TSCResNet()
    print(model)
