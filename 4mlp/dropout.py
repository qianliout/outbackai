import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    res =  mask * X / (1.0 - dropout)
    return res

if __name__ == '__main__':
    X = torch.arange(8, dtype=torch.float32).reshape((2, 4))
    print(X)
    print(dropout_layer(X, 1))