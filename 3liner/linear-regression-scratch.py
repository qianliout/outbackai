import random

import torch

from d2l import torch as d2l


class LinearRegressionScratch(d2l.Module):  # @save
    """The linear regression model implemented from scratch."""

    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        r"""
        forward(self, X) 是PyTorch模型的标准方法，定义了如何从输入计算输出
        X 是输入特征矩阵，形状通常为 (batch_size, num_features)
        self.w 是模型的权重参数，形状为 (num_features, 1)
        torch.matmul(X, self.w) 执行矩阵乘法，计算特征与权重的线性组合
        + self.b 加上偏置项(bias)，完成线性变换 y = Xw + b
        最终返回的是模型的预测值
        """
        res = torch.matmul(X, self.w) + self.b
        return res

    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)


class SGD(d2l.HyperParameters):  # @save
    """Minibatch stochastic gradient descent."""

    def __init__(self, params, lr):
        self.save_hyperparameters()

    #  随机梯度下降法
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


@d2l.add_to_class(d2l.Trainer)  # @save
def prepare_batch(self, batch):
    return batch


@d2l.add_to_class(d2l.Trainer)  # @save
def fit_epoch(self):
    # 设置模型为训练模式（影响Dropout/BatchNorm等层的行为）
    self.model.train()
    # 遍历训练数据的所有批次
    for batch in self.train_dataloader:
        # 1. 前向传播：计算当前批次的损失
        loss = self.model.training_step(self.prepare_batch(batch))
        # 2. 清空优化器中所有参数的梯度
        self.optim.zero_grad()
        # 3. 反向传播（在no_grad上下文中执行）
        # 必须在no_grad上下文中执行反向传播，因为这个程序是手动计算梯度的:self.optim.step()
        with torch.no_grad():
            loss.backward()
            # 4. 梯度裁剪（防止梯度爆炸）
            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val, self.model)
            # 5. 参数更新：根据梯度更新模型参数
            self.optim.step()

        # 更新训练批次计数器
        self.train_batch_idx += 1
    # 如果没有验证数据则直接返回
    if self.val_dataloader is None:
        return
    # 设置模型为评估模式（禁用Dropout/BatchNorm的训练行为）
    self.model.eval()
    # 遍历验证数据的所有批次
    for batch in self.val_dataloader:
        # 在验证阶段不需要计算梯度
        with torch.no_grad():
            # 执行验证步骤（只计算指标，不更新参数）
            self.model.validation_step(self.prepare_batch(batch))
        # 更新验证批次计数器
        self.val_batch_idx += 1


if __name__ == "__main__":
    model = LinearRegressionScratch(2, lr=0.03)
    data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)

    # 这段代码定义了PyTorch中的no_grad上下文管理器，用于临时禁用梯度计算。以下是详细解释：
    # 线程安全性：
    # 只影响当前线程，不会干扰其他线程的计算
    # 典型应用场景包括模型推理、参数冻结等不需要反向传播的情况。
    with torch.no_grad():
        print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
        print(f'error in estimating b: {data.b - model.b}')
