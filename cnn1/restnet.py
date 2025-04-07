import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  # @save
    """The Residual block of ResNet models."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        Y = F.relu(Y)
        return Y


class ResNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)


@d2l.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()  # 调用父类Classifier的初始化
    self.save_hyperparameters()  # 保存超参数(lr, num_classes等)

    # 1. 构建初始卷积块(b1)
    self.net = nn.Sequential(self.b1())

    # 2. 添加残差块组
    for i, b in enumerate(arch):  # arch格式如[(2,64),(2,128)...]
        # 每个b包含(残差块数量, 通道数)
        self.net.add_module(
            f'b{i + 2}',  # 命名从b2开始
            self.block(*b, first_block=(i == 0))  # 第一个块特殊处理
        )

    # 3. 添加最后的分类层
    self.net.add_module('last', nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化到1x1
        nn.Flatten(),  # 展平
        nn.LazyLinear(num_classes)  # 全连接输出层
    ))

    # 4. 初始化CNN权重
    self.net.apply(d2l.init_cnn)


class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                         lr, num_classes)


if __name__ == '__main__':
    blk = Residual(3)
    X = torch.randn(4, 3, 6, 6)
    print(blk(X).shape)

    blk = Residual(6, use_1x1conv=True, strides=2)
    print(blk(X).shape)

    ResNet18().layer_summary((1, 1, 96, 96))

    model = ResNet18(lr=0.01)
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model, data)
