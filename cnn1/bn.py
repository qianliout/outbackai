import torch
import torch
from torch import nn
from d2l import torch as d2l

X = torch.tensor([
    # 样本 0 (batch=0)
    [
        [[1, 2], [3, 4]],  # 通道 0
        [[5, 6], [7, 8]],  # 通道 1
        [[9, 10], [11, 12]]  # 通道 2
    ],
    # 样本 1 (batch=1)
    [
        [[13, 14], [15, 16]],  # 通道 0
        [[17, 18], [19, 20]],  # 通道 1
        [[21, 22], [23, 24]]  # 通道 2
    ]
], dtype=torch.float32)

print(X.shape)  # 输出: torch.Size([2, 3, 2, 2])

mean = X.mean(dim=(0, 2), keepdim=True)
print(mean)  # 输出: tensor([[[[ 8.5000]], [[12.5000]], [[16.5000]]]])
print(mean.shape)  # 输出: torch.Size([1, 3, 1, 1])


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))