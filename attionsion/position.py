import math
import torch
from torch import nn
from d2l import torch as d2l

num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens), (batch_size, num_queries, num_hiddens))


class PositionalEncoding(nn.Module):  # @save
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


encoding_dim, num_steps = 4, 2
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, : X.shape[1], :]
d2l.plot(
    torch.arange(num_steps),
    P[0, :, 6:10].T,
    xlabel="Row (position)",
    figsize=(6, 2.5),
    legend=["Col %d" % d for d in torch.arange(6, 10)],
)
for i in range(8):
    print(f"{i} in binary is {i:>03b}")
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(
    P,
    xlabel="Column (encoding dimension)",
    ylabel="Row (position)",
    figsize=(3.5, 4),
    cmap="Blues",
)
