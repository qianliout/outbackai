import torch
from torch import nn
from d2l import torch as d2l

# 生成1000个时间点
T = 1000  

# 创建从1到1000的时间序列[1,2,3,...,1000]，使用float32类型
time = torch.arange(1, T + 1, dtype=torch.float32)

# 生成正弦波信号并添加高斯噪声：
# 1. torch.sin(0.01 * time) 生成基础正弦波：
#    - 0.01控制频率，使波形在1000个点内完成约1.6个完整周期
#    - 公式：sin(2πft)，这里f≈0.0016 (0.01/2π)
# 2. torch.normal(0, 0.2, (T,)) 添加噪声：
#    - 均值0，标准差0.2的正态分布噪声
#    - 形状(T,)表示生成1000个噪声点
# 3. 最终x是带噪声的正弦波信号
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

# 设置时间窗口大小为4（用前4个时间点预测下一个时间点）
tau = 4  

# 初始化特征矩阵：形状为 (996, 4)
# - 行数 T-tau = 996（总时间点1000减去窗口大小4）
# - 列数 tau = 4（每个样本包含连续4个时间点）
features = torch.zeros((T - tau, tau))

# 填充特征矩阵：使用滑动窗口方法
for i in range(tau):
    # 第i列填充从第i个时间点开始的连续序列
    # x[i: T-tau+i] 获取从i到T-tau+i-1的子序列
    features[:, i] = x[i: T - tau + i]

# 构建标签：形状为 (996, 1)
# - 取x从tau(4)开始的所有点作为标签
# - reshape(-1,1)确保是列向量形式
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    net.apply(init_weights)
    return net


# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

# 初始化一个全零的张量用于存储多步预测结果，长度与原始序列相同
multistep_preds = torch.zeros(T)

# 用原始数据的前n_train+tau个点作为初始值（训练集+窗口大小）
# 这部分数据作为已知真实值，不进行预测
multistep_preds[: n_train + tau] = x[: n_train + tau]

# 对测试集部分进行迭代预测（从n_train+tau开始到序列结束）
for i in range(n_train + tau, T):
    # 用前tau个预测值作为输入，预测下一个点
    # 1. multistep_preds[i-tau:i] 获取最近的tau个预测值
    # 2. reshape(1,-1) 转换为(1,tau)的形状，符合网络输入要求
    # 3. net() 进行预测得到下一个点的值
    output = net(multistep_preds[i - tau:i].reshape((1, -1)))
    multistep_preds[i] = output

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))

max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

if __name__ == '__main__':
    steps = (1, 4, 16, 64)
    d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))


