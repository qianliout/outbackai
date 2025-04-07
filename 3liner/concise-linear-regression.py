import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


class LinearRegression(d2l.Module):  # @save
    """The linear regression model implemented with high-level APIs."""

    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        # - 不会立即要求输入维度
        # - 在第一次接收输入数据时自动推断输入特征数
        # - 适合网络结构动态变化的场景
        # 相当于：nn.Linear(input_features, 1)  输出维度为1,但省去了手动指定输入维度的步骤
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)


@d2l.add_to_class(LinearRegression)  # @save
def forward(self, X):
    # 自动根据X的shape[-1]确定输入维度
    # X = [batch_size, input_features]
    l = X.shape[-1]
    return self.net(X)


@d2l.add_to_class(LinearRegression)  # @save
def loss(self, y_hat, y):
    # 均方误差损失函数
    fn = nn.MSELoss()
    # 计算预测值y_hat和真实值y之间的均方误差
    res =  fn(y_hat, y)
    #  这里执行fn时，其实是执行的MSELoss的 forward 方法
    return res


@d2l.add_to_class(LinearRegression)  # @save
def configure_optimizers(self):
    # 随机梯度下降简称：SGD
    # self.parameters()  # 获取模型所有可训练参数(权重和偏置)
    # self.lr  # 从超参数中获取学习率(这里为0.03)
    return torch.optim.SGD(self.parameters(), self.lr)


if __name__ == '__main__':
    model = LinearRegression(lr=0.03)
    data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)
