import torch
import torch_npu
import math
import time

batch_configs = [[1600, 8, 10, 1200]]

rules = [0, 1, 2, 2]
D = 64

B = len(batch_configs)
num_heads = 8

scale = 1.0 / math.sqrt(D)
device = "npu:0"
dtype = torch.bfloat16

seq_lens = [sum(cfg) for cfg in batch_configs]
max_seq = max(seq_lens)


def compute_offsets(cfg):
    offsets = [0]
    for v in cfg:
        offsets.append(offsets[-1] + v)
    return offsets


def generate_mask_b(seq_len, offsets, rules):
    mask_b = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    logical_positions = torch.arange(seq_len)
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32)
    seg_ids = torch.searchsorted(offsets_tensor[1:], logical_positions, right=True)

    for seg_id_val in range(len(rules)):
        rule = rules[seg_id_val]
        q_indices = (seg_ids == seg_id_val).nonzero().squeeze(-1)
        if q_indices.numel() == 0:
            continue
        if rule == 0:
            k_range = torch.arange(seq_len)
            causal_mask = k_range.unsqueeze(0) <= logical_positions[q_indices].unsqueeze(1)
            mask_b[q_indices] = causal_mask.float()
        elif rule == 1:
            end = offsets[seg_id_val + 1]
            mask_b[q_indices, :end] = 1.0
        elif rule == 2:
            start = offsets[seg_id_val]
            mask_b[q_indices, :start] = 1.0
            mask_b[q_indices, logical_positions[q_indices]] = 1.0
    return mask_b


def build_fia_inputs(seg_lengths, rules, D, num_heads):
    """从统一 config 构建 BNSD 布局的 q/k/v + atten_mask。

    Args:
        seg_lengths: list of list, 每个 batch 的 segment 长度
        rules: list, 每个 segment 的 rule
        D: int, head dim
        num_heads: int, head 数量

    Returns:
        (q_input, k_input, v_input, atten_mask, actual_seq_lengths, scale)
    """
    B = len(seg_lengths)
    seq_lens = [sum(cfg) for cfg in seg_lengths]
    max_seq = max(seq_lens)
    scale = 1.0 / math.sqrt(D)
    device = "npu:0"
    dtype = torch.bfloat16

    masks = []
    for i, cfg in enumerate(seg_lengths):
        seq_len = seq_lens[i]
        offsets = compute_offsets(cfg)
        m = generate_mask_b(seq_len, offsets, rules)
        m_padded = torch.zeros(max_seq, max_seq, dtype=torch.float32)
        m_padded[:seq_len, :seq_len] = m
        masks.append(m_padded)

    atten_mask = torch.stack(masks).unsqueeze(1).bool().to(device)

    q_list, k_list, v_list = [], [], []
    for i, cfg in enumerate(seg_lengths):
        seq_len = seq_lens[i]
        q = torch.randn(num_heads, seq_len, D, dtype=dtype, device=device)
        k = torch.randn(num_heads, seq_len, D, dtype=dtype, device=device)
        v = torch.randn(num_heads, seq_len, D, dtype=dtype, device=device)
        q_pad = torch.zeros(num_heads, max_seq, D, dtype=dtype, device=device)
        k_pad = torch.zeros(num_heads, max_seq, D, dtype=dtype, device=device)
        v_pad = torch.zeros(num_heads, max_seq, D, dtype=dtype, device=device)
        q_pad[:, :seq_len] = q
        k_pad[:, :seq_len] = k
        v_pad[:, :seq_len] = v
        q_list.append(q_pad)
        k_list.append(k_pad)
        v_list.append(v_pad)

    q_input = torch.stack(q_list)
    k_input = torch.stack(k_list)
    v_input = torch.stack(v_list)
    actual_seq_lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    return q_input, k_input, v_input, atten_mask, actual_seq_lengths, scale


def run_fia_once(q_input, k_input, v_input, atten_mask, actual_seq_lengths, scale):
    """单次调用 npu_fused_infer_attention_score。"""
    return torch_npu.npu_fused_infer_attention_score(
        q_input,
        k_input,
        v_input,
        num_heads=q_input.shape[1],
        input_layout="BNSD",
        scale=scale,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        pre_tokens=65535,
        next_tokens=65535,
        sparse_mode=0,
    )


def run_fia_timed(q_input, k_input, v_input, atten_mask, actual_seq_lengths, scale, warmup=5, repeat=20):
    """计时 npu_fused_infer_attention_score，返回 total_us。

    Args:
        warmup: int, 预热次数
        repeat: int, 计时重复次数

    Returns:
        total_us: float, repeat 次总耗时（微秒）
    """
    for _ in range(warmup):
        run_fia_once(q_input, k_input, v_input, atten_mask, actual_seq_lengths, scale)
        torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        run_fia_once(q_input, k_input, v_input, atten_mask, actual_seq_lengths, scale)
        torch.npu.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1_000_000


if __name__ == "__main__":
    q_input, k_input, v_input, atten_mask, actual_seq_lengths, scale = build_fia_inputs(batch_configs, rules, D, num_heads)

    for _ in range(5):
        out, _ = torch_npu.npu_fused_infer_attention_score(
            q_input,
            k_input,
            v_input,
            num_heads=num_heads,
            input_layout="BNSD",
            scale=scale,
            atten_mask=atten_mask,
            actual_seq_lengths=actual_seq_lengths,
            pre_tokens=65535,
            next_tokens=65535,
            sparse_mode=0,
        )
    torch.npu.synchronize()

    for _ in range(20):
        out, _ = torch_npu.npu_fused_infer_attention_score(
            q_input,
            k_input,
            v_input,
            num_heads=num_heads,
            input_layout="BNSD",
            scale=scale,
            atten_mask=atten_mask,
            actual_seq_lengths=actual_seq_lengths,
            pre_tokens=65535,
            next_tokens=65535,
            sparse_mode=0,
        )
    torch.npu.synchronize()
