import torch
from torch import nn
from d2l import torch as d2l


class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens),
            nn.ReLU(),
            nn.LazyLinear(num_outputs)
        )


# def forward(self, X):
#     X = X.reshape((-1, self.num_inputs))
#     H = relu(torch.matmul(X, self.W1) + self.b1)
#     return torch.matmul(H, self.W2) + self.b2




if __name__ == "__main__":
    model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
    data = d2l.FashionMNIST(batch_size=256)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)

"""
torch.randn(num_inputs, num_hiddens) * sigma 这行代码用于初始化神经网络权重，我来详细解释每个部分：

1. torch.randn() :

- 生成服从标准正态分布（均值为0，标准差为1）的随机数
- 参数 (num_inputs, num_hiddens) 指定输出张量形状
- 示例：输入784维，隐藏层256单元 → 生成[784, 256]的矩阵
一定注意的是：每个元素独立服从 N(0,1) 分布，而不是每一行，也就是说这784个元素一起均值是0，方差是1，
- 每个随机数都是独立采样自标准正态分布
- 均值为0，标准差为1（方差也是1）

2. * sigma :

- 将随机数缩放σ倍（这里σ=0.01）
- 效果：将标准差从1变为σ
- 数学表达：N(0,1)×σ = N(0,σ²)
3. nn.Parameter() :

- 将张量包装为可训练参数
- 会自动加入模型的parameters()列表中
- 在反向传播时会自动计算梯度
4. 为什么这样初始化 ：

- 小随机数打破对称性，避免所有神经元学得相同
- 控制初始权重范围，防止梯度爆炸/消失
- 适合配合ReLU激活函数使用（He初始化的一种简单形式）
    """
