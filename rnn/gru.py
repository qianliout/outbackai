import torch
from torch import nn
from d2l import torch as d2l


class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (
            init_weight(num_inputs, num_hiddens),
            init_weight(num_hiddens, num_hiddens),
            nn.Parameter(torch.zeros(num_hiddens)),
        )
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state

    def forward(self, inputs, H=None):
        if H is None:
            # Initial state with shape: (batch_size, num_hiddens)
            H = torch.zeros((inputs.shape[1], self.num_hiddens), device=inputs.device)
        outputs = []
        #  inputs.shape: (num_steps, batch_size, vab_size)
        for X in inputs:
            Z = torch.sigmoid(
                torch.matmul(X, self.W_xz) + torch.matmul(H, self.W_hz) + self.b_z
            )
            R = torch.sigmoid(
                torch.matmul(X, self.W_xr) + torch.matmul(H, self.W_hr) + self.b_r
            )
            H_tilde = torch.tanh(
                torch.matmul(X, self.W_xh) + torch.matmul(R * H, self.W_hh) + self.b_h
            )
            H = Z * H + (1 - Z) * H_tilde
            outputs.append(H)
        return outputs, H


if __name__ == "__main__":
    data = d2l.TimeMachine(batch_size=1024, num_steps=32)
    gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)
