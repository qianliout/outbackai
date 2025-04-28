import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class RNNScratch(d2l.Module):  # @save
    """The RNN model implemented from scratch."""

    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = torch.zeros(
                (inputs.shape[1], self.num_hiddens), device=inputs.device
            )
        else:
            # 元组解包方式以下两种方式等价：
            # state = state[0]    # 索引访问方式
            (state,) = state

        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = torch.tanh(
                torch.matmul(X, self.W_xh) + torch.matmul(state, self.W_hh) + self.b_h
            )
            outputs.append(state)
        return outputs, state


def check_len(a, n):  # @save
    """Check the length of a list."""
    assert len(a) == n, f"list's length {len(a)} != expected length {n}"


def check_shape(a, shape):  # @save
    """Check the shape of a tensor."""
    assert a.shape == shape, f"tensor's shape {a.shape} != expected shape {shape}"


class RNNLMScratch(d2l.Classifier):  # @save
    """The RNN-based language model implemented from scratch."""

    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        # W_hq 主要用于将 RNN 隐藏状态转换为词汇表大小的输出向量，在 RNNLMScratch 类的 output_layer 方法中被使用，代码如下：
        self.W_hq = nn.Parameter(
            torch.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
        )
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def training_step(self, batch):
        # 1. 计算损失：self(*batch[:-1])是模型预测结果，batch[-1]是真实标签
        #    batch结构通常为(输入序列, 目标序列)
        #  其中输入序列和输出序列的shape:(batch_size,num_steps)
        l = self.loss(self(*batch[:-1]), batch[-1])

        # 2. 计算并记录困惑度(perplexity)
        #    torch.exp(l)将负对数似然损失转换为困惑度
        #    train=True表示这是训练集的指标
        self.plot("ppl", torch.exp(l), train=True)

        # 3. 返回损失值用于梯度更新
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("ppl", torch.exp(l), train=False)

    def one_hot(self, X):
        # Output shape: (num_steps, batch_size, vocab_size)
        res = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return res

    def output_layer(self, rnn_outputs):
        # run_out shape [number_steps,batch_size,num_hidden]
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        # outputs shape [number_steps,batch_size,num_output]
        out = torch.stack(outputs, 1)
        # out shape [batch_size,number_steps,num_output]
        return out

    def forward(self, X, state=None):
        # 每个batch的初始state都是None
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        res = self.output_layer(rnn_outputs)
        return res

    def predict(self, prefix, num_preds, vocab, device=None):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(axis=2).reshape(1)))
        pre = "".join([vocab.idx_to_token[i] for i in outputs])
        return pre


@d2l.add_to_class(d2l.Trainer)  # @save
def clip_gradients(self, grad_clip_val, model):
    # 获取模型中所有需要计算梯度的参数
    params = [p for p in model.parameters() if p.requires_grad]

    # 计算所有参数梯度的L2范数（整体梯度大小）
    # 1. 对每个参数计算其梯度的平方和(p.grad ** 2)
    # 2. 对所有参数的梯度平方和求和
    # 3. 最后开平方得到整体梯度范数
    # norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    # 计算梯度平方和
    grad_squares = [torch.sum(p.grad**2) for p in params]
    # 求和
    total_sum = sum(grad_squares)
    # 开平方得到L2范数
    norm = torch.sqrt(total_sum)

    # 如果梯度范数超过阈值，进行裁剪
    if norm > grad_clip_val:
        # 对每个参数进行梯度缩放
        for param in params:
            # 按比例缩小梯度，保持方向不变
            # grad_clip_val/norm是缩放因子
            param.grad[:] *= grad_clip_val / norm


if __name__ == "__main__":
    # batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
    # rnn = RNNScratch(num_inputs, num_hiddens)
    # X = torch.ones((num_steps, batch_size, num_inputs))
    # outputs, state = rnn(X)
    #
    # check_len(outputs, num_steps)
    # check_shape(outputs[0], (batch_size, num_hiddens))
    # check_shape(state, (batch_size, num_hiddens))

    # model = RNNLMScratch(rnn, num_inputs)
    # outputs = model(torch.ones((batch_size, num_steps), dtype=torch.int64))
    # check_shape(outputs, (batch_size, num_steps, num_inputs))

    # 在 TimeMachine 类中， num_steps 参数的作用是定义每个训练样本的序列长度（时间步数）。具体来说：
    # 1. 序列建模 ：在语言模型中，我们需要将文本序列分割成固定长度的子序列进行训练。 num_steps 就是控制这个子序列的长度。
    # 2. 输入输出构造 ：
    #  - 每个样本包含 num_steps 个连续的token作为输入

    data = d2l.TimeMachine(batch_size=1024, num_steps=32)
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=24)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)

    pr = model.predict("it is", 20, data.vocab, d2l.try_gpu())
    print(pr)
