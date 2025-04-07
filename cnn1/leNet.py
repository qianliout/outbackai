import torch
from torch import nn
from d2l import torch as d2l


def init_cnn(module):  # @save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


class LeNet(d2l.Classifier):  # @save
    """The LeNet-5 model."""

    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.Sigmoid(),
            nn.LazyLinear(84),
            nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )


@d2l.add_to_class(d2l.Classifier)  # @save
def layer_summary(self, X_shape):
    X = torch.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)


if __name__ == '__main__':
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128)
    model = LeNet(lr=0.1)
    model.layer_summary((1, 1, 28, 28))

    """
    整个过程相当于：
    1. 获取一个batch的真实训练数据样本
    2. 用这些样本"刺激"网络，让Lazy层确定自己的形状
    3. 对所有卷积和全连接层应用Xavier初始化
    """
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
    trainer.fit(model, data)

"""
这行代码 `model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)` 是用于初始化LeNet模型权重的关键操作，我来详细分解它的每个部分：
1. **`data.get_dataloader(True)`**:
   - 获取训练数据的数据加载器(DataLoader)
   - 参数`True`表示获取训练集(而非测试集)
2. **`iter(...)`**:
   - 创建数据加载器的迭代器
   - 用于逐个batch访问数据
3. **`next(...)[0]`**:
   - 获取第一个batch的数据
   - `[0]`取batch中的特征部分(图像数据)
   - 形状通常是[batch_size, 1, 28, 28]
4. **`[ ... ]`**:
   - 将输入数据包装成列表
   - 因为`apply_init`方法需要输入样本的列表
5. **`init_cnn`**:
   - 初始化函数，使用Xavier均匀分布初始化
   - 会应用到所有`nn.Linear`和`nn.Conv2d`层
6. **`model.apply_init()`**:
   - 自定义的权重初始化方法
   - 需要实际输入数据来自动推断Lazy层的形状
   - 然后应用指定的初始化函数
整个过程相当于：
1. 获取一个batch的真实训练数据样本
2. 用这些样本"刺激"网络，让Lazy层确定自己的形状
3. 对所有卷积和全连接层应用Xavier初始化

这种初始化方式比纯随机初始化更科学，能帮助网络更快收敛，特别适合配合Sigmoid激活函数使用。
"""