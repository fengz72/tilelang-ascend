import argparse
import csv
import glob
import math
import os
import subprocess
import sys
import time
import torch

from testcase import test_configs
from sparse_fa_v5 import high_perf_sparse_attn_wrapper, golden_attention


def prepare_data(config):
    H = config["H"]
    D = config["D"]
    seg_lengths = config["seg_lengths"]
    rules = config["rules"]
    matched_prefix_arr = config["matched_prefix_arr"]
    kv_group = config.get("kv_group", 1)
    block_size = config.get("block_N", 128)

    B = len(seg_lengths)
    kv_heads = H // kv_group

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


def run_kernel(data, config):
    block_M = config.get("block_M", 128)
    block_N = config.get("block_N", 128)
    core_num = config.get("core_num", 24)
    num_stages = config.get("num_stages", 14)
    kv_group = data["kv_group"]
    cross_interval = config.get("cross_interval", 2)

    return high_perf_sparse_attn_wrapper(
        data["query_snd"].npu(),
        data["key_snd"].npu(),
        data["value_snd"].npu(),
        data["segment_offsets_i32"],
        data["segment_rules_i32"],
        data["q_seq_starts_i32"],
        data["sm_scale"],
        num_blocks=data["num_cache_blocks"],
        key_cache=data["key_cache"].npu(),
        value_cache=data["value_cache"].npu(),
        block_table_i32=data["block_table_tensor"].npu(),
        matched_prefix_lens_i32=data["matched_prefix_lens_i32"].npu(),
        block_M=block_M,
        block_N=block_N,
        core_num=core_num,
        kv_group=kv_group,
        num_stages=num_stages,
        cross_interval=cross_interval,
    )


def run_accuracy(data, config, rtol, atol):
    output_snd = run_kernel(data, config)

    ref_output = golden_attention(
        data["query_snd"],
        data["key_snd"],
        data["value_snd"],
        data["segment_offsets_i32"],
        data["segment_rules_i32"],
        data["q_seq_starts_i32"],
        data["matched_prefix_lens_i32"],
        data["key_cache"],
        data["value_cache"],
        data["block_table_tensor"],
        data["block_size"],
        data["sm_scale"],
    )

    torch.npu.synchronize()
    max_diff = (ref_output.npu().float() - output_snd.float()).abs().max().item()
    try:
        torch.testing.assert_close(ref_output.npu(), output_snd, rtol=rtol, atol=atol)
        print(f"  accuracy: PASSED (max_diff={max_diff:.6f})")
        return True
    except AssertionError as e:
        print(f"  accuracy: FAILED (max_diff={max_diff:.6f})")
        print(f"    {e}")
        return False


def run_perf_timed(data, config, warmup, repeat):
    for _ in range(warmup):
        run_kernel(data, config)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        run_kernel(data, config)
    torch.npu.synchronize()
    t1 = time.perf_counter()

    total_us = (t1 - t0) * 1_000_000
    avg_us = total_us / repeat
    print(f"  total: {total_us:.2f}us  avg: {avg_us:.2f}us  ({repeat} runs)")
    return total_us


def case_label(idx, config):
    name = config.get("name", "")
    B = len(config["seg_lengths"])
    D = config["D"]
    if name:
        return f"Case[{idx}] {name}"
    return f"Case[{idx}] D{D}_batch{B}"


def parse_kernel_times(log_dir):
    pattern = os.path.join(log_dir, "OPPROF_*", "OpBasicInfo.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    csv_path = max(matches, key=os.path.getmtime)
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Op Name", "").strip()
            dur = row.get("Task Duration(us)", "").strip()
            if name and dur:
                entries.append({"name": name, "duration_us": float(dur)})
    if not entries:
        return None
    half = len(entries) // 2
    return entries[half:] if half > 0 else entries


def run_msprof_case(case_index, config, log_dir):
    cmd = [
        "msprof", "op",
        "--kernel-name=main_kernel",
        f"--output={log_dir}",
        sys.executable, os.path.abspath(__file__),
        "--_msprof-worker",
        "--case-index", str(case_index),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  msprof failed (exit {result.returncode})")
        if result.stderr:
            print(f"    {result.stderr.strip()[-200:]}")
        return None
    return parse_kernel_times(log_dir)


def print_summary(results):
    name_w = max(len(r["label"]) for r in results)
    name_w = max(name_w, 4)
    header = f"{'Case':<{name_w}}  {'Total(us)':>12}  {'Avg(us)':>12}  {'Kernel(us)':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        total = f"{r['total_us']:.2f}" if r["total_us"] is not None else "-"
        avg = f"{r['avg_us']:.2f}" if r["avg_us"] is not None else "-"
        kernel_us = r["kernel_us"]
        kernel = f"{kernel_us:.2f}" if kernel_us is not None else "-"
        print(f"{r['label']:<{name_w}}  {total:>12}  {avg:>12}  {kernel:>12}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Sparse FlashAttention test runner")
    parser.add_argument("--mode", choices=["accuracy", "perf", "all"], default="all")
    parser.add_argument("--case-index", type=int, default=None, help="Run a single case by index (0-based)")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--msprof-log", type=str, default="./log", help="msprof output directory")
    parser.add_argument("--_msprof-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._msprof_worker:
        config = test_configs[args.case_index]
        data = prepare_data(config)
        run_kernel(data, config)
        run_kernel(data, config)
        return

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(test_configs):
            print(f"Error: case-index must be 0~{len(test_configs)-1}")
            return
        indices = [args.case_index]
    else:
        indices = list(range(len(test_configs)))

    passed, failed = 0, 0
    perf_results = []

    for idx in indices:
        config = test_configs[idx]
        label = case_label(idx, config)
        print(label)
        data = prepare_data(config)

        if args.mode in ("accuracy", "all"):
            ok = run_accuracy(data, config, args.rtol, args.atol)
            if ok:
                passed += 1
            else:
                failed += 1

        if args.mode in ("perf", "all"):
            total_us = run_perf_timed(data, config, args.warmup, args.repeat)
            avg_us = total_us / args.repeat

            print(f"  running msprof...")
            kernel_entries = run_msprof_case(idx, config, args.msprof_log)
            if kernel_entries:
                kernel_us = sum(e["duration_us"] for e in kernel_entries)
                print(f"  kernel total: {kernel_us:.2f}us ({len(kernel_entries)} kernels)")
                for e in kernel_entries:
                    print(f"    {e['name']}: {e['duration_us']:.2f}us")
            else:
                kernel_us = None
                kernel_entries = []
                print(f"  kernel: N/A")

            perf_results.append({
                "label": label,
                "total_us": total_us,
                "avg_us": avg_us,
                "kernel_us": kernel_us,
                "kernel_entries": kernel_entries,
            })

    if args.mode in ("accuracy", "all"):
        print(f"\nAccuracy: {passed} passed, {failed} failed, {passed + failed} total")

    if perf_results:
        print_summary(perf_results)


if __name__ == "__main__":
    main()
