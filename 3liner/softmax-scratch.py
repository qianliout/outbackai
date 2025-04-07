import random

import torch

from d2l import torch as d2l

import torch
from d2l import torch as d2l

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# 1. 核心原则：sum(dim=k) 的含义
# dim=k 表示 沿着第 k 个维度求和，该维度会被消除（除非 keepdim=True）。
# 记忆口诀：
# “沿着哪个维度求和，哪个维度就消失！”
y1 = X.sum(0, keepdims=True)
y2 = X.sum(1, keepdims=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here


X = torch.rand((2, 5))
X_prob = softmax(X)
y3 = X_prob.sum(1)


class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]


@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    # print("forward pre x shape:", X.shape)
    # reshape参数解析 ：
    # -1 ：表示该维度由PyTorch自动计算,也就是第0个维度
    # self.W.shape[0] ：权重矩阵的第1维大小（这里是784，对应1*28*28展平）
    # 所以这里的reshape操作将输入数据X从形状为(batch_size, 1, 28, 28)的4D张量展平为(batch_size, 784)的2D张量
    # 如果这里的784没有被整除，会报错：(一定要正确指定)
    X = X.reshape((-1, self.W.shape[0]))
    # print("forward after x shape:", X.shape)
    return softmax(torch.matmul(X, self.W) + self.b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]


def cross_entropy(y_hat, y):
    r0 = (list(range(len(y_hat))),y)
    r1 = y_hat[list(range(len(y_hat))), y]
    r2 = torch.log(r1)
    r3 = -r2.mean()
    return r3
    # return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()


# ce =  cross_entropy(y_hat, y)
# print("cross_entropy:", ce)


@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == "__main__":
    data = d2l.FashionMNIST(batch_size=256)
    model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=1)
    trainer.fit(model, data)

    #  预测
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