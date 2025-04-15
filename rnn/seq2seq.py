import collections
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def init_seq2seq(module):  # @save
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


class Seq2SeqEncoder(d2l.Encoder):  # @save
    """The RNN encoder for sequence-to-sequence learning."""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)

    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        # x.t() 是张量的转置操作(transpose)
        embs = self.embedding(X.t().type(torch.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens) *   包含每个时间步的隐藏状态（即RNN在每个时间步的输出）。
        # state shape: (num_layers, batch_size, num_hiddens) *    包含最后一个时间步的所有层的隐藏状态(所以没有 num_steps)
        return outputs, state


class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(X.t().type(torch.int32))
        # enc_output shape: (num_steps, batch_size, num_hiddens) *   包含每个时间步的隐藏状态（即RNN在每个时间步的输出）。
        # hidden_state shape: (num_layers, batch_size, num_hiddens) *    包含最后一个时间步的所有层的隐藏状态(所以没有 num_steps)
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)
        # Concat at the feature dimension
        embs_and_context = torch.cat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = self.dense(outputs).swapaxes(0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]


class Seq2Seq(d2l.EncoderDecoder):  # @save
    """The RNN encoder--decoder for sequence to sequence learning."""

    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot("loss", self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, Y_hat, Y):
        l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
        mask = (Y.reshape(-1) != self.tgt_pad).type(torch.float32)
        return (l * mask).sum() / mask.sum()


@d2l.add_to_class(d2l.EncoderDecoder)  # @save
def predict_step(self, batch, device, num_steps, save_attention_weights=False):
    # 将batch中所有数据移动到指定设备(GPU/CPU)
    batch = [a.to(device) for a in batch]
    # 解包batch数据：源序列、目标序列、源序列有效长度等
    src, tgt, src_valid_len, _ = batch
    # 编码器处理源序列，获取编码输出
    enc_all_outputs = self.encoder(src, src_valid_len)
    # 初始化解码器状态（使用编码器最终状态）
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    # 初始化输出序列：以目标序列的第一个词(通常是<bos>开始符)作为起始
    outputs, attention_weights = [tgt[:, (0)].unsqueeze(1)], []
    # 逐步生成输出序列
    for _ in range(num_steps):
        # 解码器预测下一个词的概率分布
        Y, dec_state = self.decoder(outputs[-1], dec_state)
        # 取概率最大的词索引作为当前步的输出
        outputs.append(Y.argmax(2))
        # 可选：保存注意力权重用于可视化
        if save_attention_weights:
            attention_weights.append(self.decoder.attention_weights)
    # 拼接所有时间步的输出（跳过初始的<bos>）
    return torch.cat(outputs[1:], 1), attention_weights


def bleu(pred_seq, label_seq, k):  # @save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(" "), label_seq.split(" ")
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i : i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[" ".join(pred_tokens[i : i + n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i : i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def predict():
    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    preds, _ = model.predict_step(data.build(engs, fras), d2l.try_gpu(), data.num_steps)
    for en, fr, p in zip(engs, fras, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == "<eos>":
                break
            translation.append(token)
        print(
            f"{en} => {translation}, bleu,"
            f'{bleu(" ".join(translation), fr, k=2):.3f}'
        )


if __name__ == "__main__":
    vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
    batch_size, num_steps = 4, 9
    encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
    X = torch.zeros((batch_size, num_steps))
    enc_outputs, enc_state = encoder(X)
    d2l.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
    d2l.check_shape(enc_state, (num_layers, batch_size, num_hiddens))

    decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
    state = decoder.init_state(encoder(X))
    dec_outputs, state = decoder(X, state)
    d2l.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
    d2l.check_shape(state[1], (num_layers, batch_size, num_hiddens))

    data = d2l.MTFraEng(batch_size=128)
    embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    decoder = Seq2SeqDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab["<pad>"], lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)
