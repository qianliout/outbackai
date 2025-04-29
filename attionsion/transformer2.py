import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """标准的位置编码层"""
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: 嵌入维度
            max_len: 最大序列长度
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入序列 [batch_size, seq_len, d_model]
        Returns:
            添加位置编码后的序列 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: 输入输出维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len, seq_len]
        Returns:
            注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # 线性变换并分头 [batch_size, num_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数 [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码（如需要）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和 [batch_size, num_heads, seq_len, d_k]
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头 [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出线性变换
        return self.W_o(attn_output)


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络（FFN）"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 输入输出维度
            d_ff: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """前向传播"""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 输入输出维度
            num_heads: 注意力头数
            d_ff: FFN隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """前向传播（含残差连接）"""
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # FFN子层
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 输入输出维度
            num_heads: 注意力头数
            d_ff: FFN隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """前向传播（含残差连接）"""
        # 自注意力子层（解码器自注意力）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 交叉注意力子层（编码器-解码器注意力）
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # FFN子层
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型（编码器-解码器结构）"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            d_ff: FFN隐藏层维度
            dropout: Dropout概率
            max_len: 最大序列长度
        """
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, src, src_mask):
        """编码器前向传播"""
        src = self.dropout(self.pos_encoding(self.encoder_embedding(src)))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        return src
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """解码器前向传播"""
        tgt = self.dropout(self.pos_encoding(self.decoder_embedding(tgt)))
        for layer in self.decoder_layers:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        return tgt
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """完整前向传播"""
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)


# 示例用法
if __name__ == "__main__":
    # 定义超参数
    src_vocab_size = 5000  # 源语言词汇表大小
    tgt_vocab_size = 5000  # 目标语言词汇表大小
    d_model = 512          # 模型维度
    num_heads = 8          # 注意力头数
    num_layers = 6         # 编码器/解码器层数
    d_ff = 2048            # FFN隐藏层维度
    dropout = 0.1          # Dropout概率
    
    # 初始化模型
    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads, 
        num_layers, num_layers, d_ff, dropout
    )
    
    # 模拟输入数据
    batch_size = 32
    src_seq_len = 50
    tgt_seq_len = 60
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 生成掩码（示例中全为1，表示无遮盖）
    src_mask = torch.ones(batch_size, 1, src_seq_len)
    tgt_mask = torch.ones(batch_size, tgt_seq_len, tgt_seq_len)
    
    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)
    print(f"输出形状: {output.shape}")  # 应为 [batch_size, tgt_seq_len, tgt_vocab_size]