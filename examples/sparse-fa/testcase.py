import math
import random
import torch


def prepare_data(config):
    H = config["H"]
    D = config["D"]
    seg_lengths = config["seg_lengths"]
    rules = config["rules"]
    matched_prefix_arr = config["matched_prefix_arr"]
    kv_group = config.get("kv_group", 1)
    block_N = config.get("block_N", 128)
    block_size = block_N

    B = len(seg_lengths)
    kv_heads = H // kv_group

    for b_idx, plen in enumerate(matched_prefix_arr):
        if plen % block_N != 0:
            raise ValueError(
                f"matched_prefix_arr[{b_idx}] = {plen} is not a multiple of block_N={block_N}"
            )

    S_logical_list = [sum(sl) for sl in seg_lengths]

    offsets_list = []
    for sl in seg_lengths:
        off = [0]
        for s in sl:
            off.append(off[-1] + s)
        offsets_list.append(off)

    actual_q_len_arr = [S_logical_list[b] - matched_prefix_arr[b] for b in range(B)]

    q_seq_starts_arr = [0]
    for b in range(1, B):
        q_seq_starts_arr.append(q_seq_starts_arr[b - 1] + actual_q_len_arr[b - 1])
    q_seq_starts_arr.append(q_seq_starts_arr[B - 1] + actual_q_len_arr[B - 1])

    torch.manual_seed(0)

    q_list, k_list_full, v_list_full, k_list_live, v_list_live = [], [], [], [], []
    for b in range(B):
        q_b = torch.randn(actual_q_len_arr[b], H, D, dtype=torch.float32, device="cpu").to(torch.bfloat16)
        k_b_full = torch.randn(S_logical_list[b], kv_heads, D, dtype=torch.float32, device="cpu").to(torch.bfloat16)
        v_b_full = torch.randn(S_logical_list[b], kv_heads, D, dtype=torch.float32, device="cpu").to(torch.bfloat16)
        q_list.append(q_b)
        k_list_full.append(k_b_full)
        v_list_full.append(v_b_full)
        prefix_len = matched_prefix_arr[b]
        k_list_live.append(k_b_full[prefix_len:])
        v_list_live.append(v_b_full[prefix_len:])

    query_snd = torch.cat(q_list, dim=0)
    key_snd = torch.cat(k_list_live, dim=0)
    value_snd = torch.cat(v_list_live, dim=0)

    num_cache_blocks = max(1, sum((matched_prefix_arr[b] + block_size - 1) // block_size for b in range(B)))
    key_cache = torch.zeros(num_cache_blocks, block_size, kv_heads, D, dtype=torch.bfloat16, device="cpu")
    value_cache = torch.zeros(num_cache_blocks, block_size, kv_heads, D, dtype=torch.bfloat16, device="cpu")

    block_table_arr = []
    physical_block_offset = 0
    for b in range(B):
        prefix_len = matched_prefix_arr[b]
        num_logical_blocks = (S_logical_list[b] + block_size - 1) // block_size
        bt_row = []
        for lb in range(num_logical_blocks):
            if lb < (prefix_len + block_size - 1) // block_size:
                bt_row.append(physical_block_offset + lb)
            else:
                bt_row.append(0)
        block_table_arr.append(bt_row)
        physical_block_offset += (prefix_len + block_size - 1) // block_size

    for b in range(B):
        prefix_len = matched_prefix_arr[b]
        if prefix_len > 0:
            for p in range(prefix_len):
                block_idx = p // block_size
                block_offset = p % block_size
                physical_block = block_table_arr[b][block_idx]
                key_cache[physical_block, block_offset, :, :] = k_list_full[b][p, :, :]
                value_cache[physical_block, block_offset, :, :] = v_list_full[b][p, :, :]

    max_blocks_per_request = max(1, max(len(bt) for bt in block_table_arr))
    block_table_tensor = torch.zeros(B, max_blocks_per_request, dtype=torch.int32, device="cpu")
    for b in range(B):
        for lb in range(len(block_table_arr[b])):
            block_table_tensor[b, lb] = block_table_arr[b][lb]

    segment_offsets_i32 = torch.tensor(offsets_list, dtype=torch.int32, device="cpu")
    segment_rules_i32 = torch.tensor(rules, dtype=torch.int32, device="cpu")
    q_seq_starts_i32 = torch.tensor(q_seq_starts_arr, dtype=torch.int32, device="cpu")
    matched_prefix_lens_i32 = torch.tensor(matched_prefix_arr, dtype=torch.int32, device="cpu")

    sm_scale = 1.0 / math.sqrt(D)

    return dict(
        query_snd=query_snd,
        key_snd=key_snd,
        value_snd=value_snd,
        segment_offsets_i32=segment_offsets_i32,
        segment_rules_i32=segment_rules_i32,
        q_seq_starts_i32=q_seq_starts_i32,
        matched_prefix_lens_i32=matched_prefix_lens_i32,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table_tensor=block_table_tensor,
        num_cache_blocks=num_cache_blocks,
        sm_scale=sm_scale,
        H=H,
        D=D,
        kv_group=kv_group,
        block_size=block_size,
    )


_base_patterns = [
    {
        "seg_lengths": [[1600, 8, 200, 1200]],
        "rules": [0, 1, 2, 2],
        "matched_prefix_arr": [0],
    },
    {
        "seg_lengths": [[1600, 8, 200, 1200]],
        "rules": [0, 1, 0, 2],
        "matched_prefix_arr": [0],
    },
    {
        "seg_lengths": [[1600, 8, 200, 1200], [1700, 8, 300, 1024]],
        "rules": [0, 1, 2, 2],
        "matched_prefix_arr": [0, 0],
    },
    {
        "seg_lengths": [[1800, 8, 100, 1500], [1500, 8, 300, 1200]],
        "rules": [0, 1, 0, 2],
        "matched_prefix_arr": [0, 0],
    },
    {
        "seg_lengths": [
            [1600, 8, 200, 1200],
            [1700, 8, 300, 1024],
            [1680, 8, 200, 1280],
            [2000, 8, 700, 2048],
        ],
        "rules": [0, 1, 2, 2],
        "matched_prefix_arr": [0, 0, 0, 0],
    },
    {
        "seg_lengths": [
            [3200, 8, 200, 1200],
            [2300, 8, 400, 1800],
            [2080, 8, 200, 1800],
            [1700, 8, 100, 1024],
        ],
        "rules": [0, 1, 0, 2],
        "matched_prefix_arr": [0, 0, 0, 0],
    },
    {
        "seg_lengths": [
            [2200, 8, 200, 1024],
            [1700, 8, 100, 1100],
            [2440, 8, 200, 2048],
            [1600, 8, 600, 1900],
            [3300, 8, 200, 1300],
            [1700, 8, 300, 2100],
            [1780, 8, 700, 1200],
            [2048, 8, 500, 1800],
        ],
        "rules": [0, 1, 2, 2],
        "matched_prefix_arr": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    {
        "seg_lengths": [
            [2200, 8, 200, 1024],
            [1700, 8, 300, 1100],
            [2440, 8, 200, 2048],
            [1600, 8, 600, 1800],
            [3300, 8, 200, 1300],
            [1700, 8, 300, 2048],
            [1780, 8, 300, 1024],
            [2048, 8, 500, 1800],
        ],
        "rules": [0, 1, 0, 2],
        "matched_prefix_arr": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    {
        "seg_lengths": [
            [1600, 200, 1024],
        ],
        "rules": [1, 0, 2],
        "matched_prefix_arr": [0],
    },
    {
        "seg_lengths": [
            [1600, 200, 1000],
            [2000, 300, 1100],
        ],
        "rules": [1, 0, 2],
        "matched_prefix_arr": [0, 0],
    },
    {
        "seg_lengths": [
            [1600, 200, 1024],
            [1700, 300, 1600],
            [1680, 200, 1200],
            [2000, 700, 2400],
        ],
        "rules": [1, 0, 2],
        "matched_prefix_arr": [0, 0, 0, 0],
    },
    {
        "seg_lengths": [
            [2200, 200, 1024],
            [1700, 300, 1100],
            [2440, 200, 2048],
            [1600, 400, 1800],
            [2200, 200, 1300],
            [1700, 300, 2048],
            [180, 300, 1024],
            [2048, 700, 1800],
        ],
        "rules": [1, 0, 2],
        "matched_prefix_arr": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    {
        "seg_lengths": [[1600, 8, 10, 1200]],
        "rules": [0, 1, 2, 2],
        "matched_prefix_arr": [0],
    },
    {
        "seg_lengths": [[1600, 8, 8, 12, 1200], [1700, 8, 6, 15, 1024]],
        "rules": [0, 1, 0, 0, 2],
        "matched_prefix_arr": [0, 0],
    },
    {
        "seg_lengths": [
            [1600, 8, 5, 6, 7, 8, 1200],
            [1700, 8, 10, 12, 13, 15, 1024],
            [1680, 8, 10, 12, 13, 15, 1280],
            [2000, 8, 13, 14, 15, 15, 2048],
        ],
        "rules": [0, 1, 2, 2, 2, 2, 2],
        "matched_prefix_arr": [0, 0, 0, 0],
    },
    {
        "seg_lengths": [
            [2200, 8, 5, 5, 5, 5, 5, 5, 5, 5, 1024],
            [1700, 8, 5, 5, 5, 5, 5, 5, 5, 10, 1100],
            [2440, 8, 5, 5, 5, 5, 5, 5, 5, 12, 2048],
            [1600, 8, 5, 15, 5, 5, 5, 5, 5, 5, 1900],
            [3300, 8, 5, 10, 10, 5, 5, 5, 5, 5, 1300],
            [1700, 8, 15, 15, 5, 5, 5, 5, 5, 5, 2100],
            [1780, 8, 15, 15, 15, 5, 5, 5, 5, 5, 1200],
            [2048, 8, 15, 15, 15, 15, 5, 5, 5, 5, 1800],
        ],
        "rules": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        "matched_prefix_arr": [0, 0, 0, 0, 0, 0, 0, 0],
    },
]


def _generate_multi_seg_patterns():
    patterns = []
    rng = random.Random(42)

    prefix_pool = [1600, 2000, 2400, 3200]
    suffix_pool = [1024, 1200, 1800, 2048]

    cases = [
        (64,   16,  4), (64,   16,  2), (64,   16,  1),
        (128,  16,  2), (128,  16,  1),
        (256,  16,  1),
        (512,  16,  1),
        (1024, 16,  1),
        (64,   32,  2),
        (64,   64,  2), (64,   64,  1),
        (64,   128, 1),
        (64,   256, 1),
        (64,   512, 1),
        (128,  32,  2), (128,  32,  1),
        (128,  64,  1),
        (128,  128, 1),
        (128,  256, 1),
        (128,  512, 1),
        (256,  32,  1),
        (256,  64,  1),
        (256,  128, 1),
        (256,  256, 1),
        (256,  512, 1),
        (512,  32,  1),
        (512,  64,  1),
        (512,  128, 1),
        (512,  256, 1),
        (512,  512, 1),
        (1024, 32,  1),
        (1024, 64,  1),
        (1024, 128, 1),
        (1024, 256, 1),
        (1024, 512, 1),
    ]

    for num_seg, seg_len, B in cases:
        prefix = rng.choice(prefix_pool)
        suffix = rng.choice(suffix_pool)
        sl = [prefix, 8] + [seg_len] * num_seg + [suffix]
        seg_lengths = [sl] * B
        rules = [0, 1] + [0] * num_seg + [2]
        patterns.append({
            "seg_lengths": seg_lengths,
            "rules": rules,
            "matched_prefix_arr": [0] * B,
        })

    return patterns


_base_patterns.extend(_generate_multi_seg_patterns())

_D_values = [128, 64, 32]

test_configs = [
    {"H": 8, "D": d, **pattern}
    for d in _D_values
    for pattern in _base_patterns
]