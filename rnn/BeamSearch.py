import torch
import torch.nn.functional as F


def beam_search(model, src, beam_size=5, max_length=50, device='cuda'):
    """
    Args:
        model: Seq2Seq 模型（需实现 encode() 和 decode_step()）
        src: 输入序列 [1, src_len]
        beam_size: 束宽（默认5）
        max_length: 最大生成长度（默认50）
        device: 设备类型
    Returns:
        best_seq: 最优生成序列 [1, tgt_len]
        best_score: 对应的得分
    """
    # 编码输入序列
    encoder_outputs = model.encode(src)  # [1, src_len, hidden_dim]

    # 初始化束：保存 (序列, 累积得分, 隐藏状态)
    beams = [(torch.tensor([[model.sos_idx]], device=device), 0.0, None)]  # (seq, score, hidden)

    for step in range(max_length):
        candidates = []
        # 对每个候选序列扩展
        for seq, score, hidden in beams:
            if seq[0, -1] == model.eos_idx:  # 已生成<EOS>则直接保留
                candidates.append((seq, score, hidden))
                continue
            # 解码下一步
            decoder_input = seq[:, -1:]  # 最后一步的token [1, 1]
            logits, hidden = model.decode_step(decoder_input, encoder_outputs, hidden)  # logits: [1, vocab_size]
            # 取Top-K个可能的下一个token
            log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]
            topk_scores, topk_tokens = log_probs.topk(beam_size, dim=-1)  # [1, beam_size]
            # 生成新候选序列
            for i in range(beam_size):
                new_seq = torch.cat([seq, topk_tokens[0, i].unsqueeze(0).unsqueeze(0)], dim=-1)  # [1, seq_len+1]
                new_score = score + topk_scores[0, i].item()
                candidates.append((new_seq, new_score, hidden))

        # 按得分排序，保留Top-K个候选
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

        # 检查是否所有候选均已生成<EOS>
        if all(seq[0, -1] == model.eos_idx for seq, _, _ in beams):
            break

    # 返回得分最高的序列
    best_seq, best_score, _ = beams[0]
    return best_seq, best_score
