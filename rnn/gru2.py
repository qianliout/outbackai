import torch
from torch import nn
from d2l import torch as d2l


class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens)


if __name__ == "__main__":
    data = d2l.TimeMachine(batch_size=1024, num_steps=32)
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=128)
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)
