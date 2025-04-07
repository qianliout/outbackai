import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class SoftmaxRegression(d2l.Classifier):  # @save
    """The softmax regression model."""

    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            # 功能作用 ：
            # - 将任意维度的输入张量转换为二维张量
            # - 保持第0维（batch维度）不变，其他所有维度展平为一维
            nn.Flatten(),
            #  LazyLiner 会初始化权重和偏置，而不需要在初始化时就指定输入维度
            nn.LazyLinear(num_outputs)
        )

    def forward(self, X):
        return self.net(X)


@d2l.add_to_class(d2l.Classifier)  # @save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    Y = Y.reshape((-1,))
    return F.cross_entropy(
        Y_hat, Y, reduction='mean' if averaged else 'none')


if __name__ == "__main__":
    data = d2l.FashionMNIST(batch_size=256)
    model = SoftmaxRegression(num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)

    X, y = next(iter(data.val_dataloader()))
    preds = model(X).argmax(axis=1)
    # print( preds.shape)
    # preds.type(y.dtype) :
    # - 将预测结果 preds 转换为与 y 相同的数据类型
    # - 因为 argmax 返回的结果可能和原始标签类型不同
    wrong = preds.type(y.dtype) != y
    X, y, preds = X[wrong], y[wrong], preds[wrong]
    labels = [a + '\n' + b for a, b in zip(
        data.text_labels(y), data.text_labels(preds))]
    data.visualize([X, y], labels=labels)
