import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
import torch
import torch_npu

from testcase import test_configs, prepare_data
from sparse_fa_scalar import high_perf_sparse_attn_wrapper
from golden import golden_attention_float64

from fia import build_fia_inputs, run_fia_timed, run_fia_once

_MASK_LIB_LOADED = False

def _ensure_mask_lib():
    global _MASK_LIB_LOADED
    if _MASK_LIB_LOADED:
        return
    mask_repo = "/export/home/weinan5/hejun/workspace/mask"
    so_path = os.path.join(mask_repo, "build_ascend910_93", "lib", "libmask_ops.so")
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"libmask_ops.so not found at {so_path}")
    if mask_repo not in sys.path:
        sys.path.insert(0, os.path.join(mask_repo, "tests", "torch"))
    torch.ops.load_library(so_path)
    _MASK_LIB_LOADED = True

_ensure_mask_lib()
from test_perf import build_mask_inputs, run_mask_op_timed, run_mask_op_once

KERNEL_NAMES = {
    "sparse-fa": "main_kernel",
    "fia": "FusedInferAttentionScore",
    "mask": "Mask",
}

TOTAL_WARMUP = 5
TOTAL_REPEAT = 10
KERNEL_WARMUP = 5
KERNEL_REPEAT = 5


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
        data["segment_offsets_i32"].npu(),
        data["segment_rules_i32"].npu(),
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
    torch.npu.synchronize()
    output_snd = run_kernel(data, config)
    torch.npu.synchronize()

    ref_output = golden_attention_float64(
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
    max_diff = (ref_output.cpu().float() - output_snd.cpu().float()).abs().max().item()
    try:
        torch.testing.assert_close(ref_output.cpu(), output_snd.cpu(), rtol=rtol, atol=atol)
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
    pattern = os.path.join(log_dir, "OPPROF_*", "**", "OpBasicInfo*.csv")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        pattern = os.path.join(log_dir, "OPPROF_*", "OpBasicInfo.csv")
        matches = glob.glob(pattern)
    if not matches:
        return None
    entries = []
    for csv_path in matches:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Op Name", "").strip()
                dur = row.get("Task Duration(us)", "").strip()
                if name and dur:
                    entries.append({"name": name, "duration_us": float(dur)})
    if not entries:
        return None
    return entries


def run_msprof_case(case_index, config, log_dir, target="sparse-fa",
                    warmup=KERNEL_WARMUP, repeat=KERNEL_REPEAT):
    kernel_name = KERNEL_NAMES.get(target, "main_kernel")
    target_log_dir = os.path.join(log_dir, target)
    if os.path.exists(target_log_dir):
        shutil.rmtree(target_log_dir)
    os.makedirs(target_log_dir, exist_ok=True)
    cmd = [
        "msprof", "op",
        f"--kernel-name={kernel_name}",
        f"--output={target_log_dir}",
        f"--warm-up={warmup}",
        f"--launch-count={repeat}",
        "--kill=on",
        sys.executable, os.path.abspath(__file__),
        "--_msprof-worker",
        "--case-index", str(case_index),
        "--_msprof-target", target,
        "--_msprof-repeat", str(warmup + repeat),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  msprof failed (exit {result.returncode})")
        if result.stderr:
            print(f"    {result.stderr.strip()[-200:]}")
        return None
    return parse_kernel_times(target_log_dir)


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


def print_compare_summary(results):
    name_w = max(len(r["label"]) for r in results)
    name_w = max(name_w, 4)
    header = (f"{'Case':<{name_w}}  "
              f"{'s_tot(us)':>10}  {'s_ker(us)':>10}  "
              f"{'f_tot(us)':>10}  {'f_ker(us)':>10}  "
              f"{'m_tot(us)':>10}  {'m_ker(us)':>10}  "
              f"{'b_tot(us)':>10}  {'b_ker(us)':>10}  "
              f"{'sp_tot':>8}  {'sp_ker':>8}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        def fmt(key):
            v = r.get(key)
            return f"{v:.2f}" if v is not None else "-"
        def fmtx(key):
            v = r.get(key)
            return f"{v:.2f}x" if v is not None else "-"
        print(f"{r['label']:<{name_w}}  "
              f"{fmt('sparse_total_avg'):>10}  {fmt('sparse_kernel_avg'):>10}  "
              f"{fmt('fia_total_avg'):>10}  {fmt('fia_kernel_avg'):>10}  "
              f"{fmt('mask_total_avg'):>10}  {fmt('mask_kernel_avg'):>10}  "
              f"{fmt('baseline_total_avg'):>10}  {fmt('baseline_kernel_avg'):>10}  "
              f"{fmtx('speedup_total'):>8}  {fmtx('speedup_kernel'):>8}")
    print(sep)


def run_compare(data, config, case_index, log_dir):
    seg_lengths = config["seg_lengths"]
    rules = config["rules"]
    D = config["D"]
    H = config["H"]

    result = {}

    # ---- sparse-fa ----
    # print(f"  [sparse-fa] total time (warmup={TOTAL_WARMUP}, repeat={TOTAL_REPEAT})...")
    sparse_total_us = run_perf_timed(data, config, TOTAL_WARMUP, TOTAL_REPEAT)
    result["sparse_total_avg"] = sparse_total_us / TOTAL_REPEAT

    # print(f"  [sparse-fa] kernel time (msprof warmup={KERNEL_WARMUP}, repeat={KERNEL_REPEAT})...")
    entries = run_msprof_case(case_index, config, log_dir, target="sparse-fa")
    if entries:
        result["sparse_kernel_avg"] = sum(e["duration_us"] for e in entries) / KERNEL_REPEAT
    else:
        result["sparse_kernel_avg"] = None

    # ---- FIA ----
    # print(f"  [FIA] total time (warmup={TOTAL_WARMUP}, repeat={TOTAL_REPEAT})...")
    q_input, k_input, v_input, atten_mask, actual_seq_lengths, scale = build_fia_inputs(
        seg_lengths, rules, D, H
    )
    fia_total_us = run_fia_timed(q_input, k_input, v_input, atten_mask,
                                 actual_seq_lengths, scale,
                                 warmup=TOTAL_WARMUP, repeat=TOTAL_REPEAT)
    result["fia_total_avg"] = fia_total_us / TOTAL_REPEAT

    # print(f"  [FIA] kernel time (msprof warmup={KERNEL_WARMUP}, repeat={KERNEL_REPEAT})...")
    entries = run_msprof_case(case_index, config, log_dir, target="fia")
    if entries:
        result["fia_kernel_avg"] = sum(e["duration_us"] for e in entries) / KERNEL_REPEAT
    else:
        result["fia_kernel_avg"] = None

    # ---- mask op ----
    # print(f"  [mask] total time (warmup={TOTAL_WARMUP}, repeat={TOTAL_REPEAT})...")
    mask_tensor, seg_offsets, seg_rules, q_starts, T = build_mask_inputs(seg_lengths, rules)
    mask_total_us = run_mask_op_timed(mask_tensor, seg_offsets, seg_rules, q_starts,
                                      warmup=TOTAL_WARMUP, repeat=TOTAL_REPEAT)
    result["mask_total_avg"] = mask_total_us / TOTAL_REPEAT

    # print(f"  [mask] kernel time (msprof warmup={KERNEL_WARMUP}, repeat={KERNEL_REPEAT})...")
    entries = run_msprof_case(case_index, config, log_dir, target="mask")
    if entries:
        result["mask_kernel_avg"] = sum(e["duration_us"] for e in entries) / KERNEL_REPEAT
    else:
        result["mask_kernel_avg"] = None

    # ---- baseline & speedup ----
    result["baseline_total_avg"] = result["fia_total_avg"] + result["mask_total_avg"]

    fia_k = result["fia_kernel_avg"]
    mask_k = result["mask_kernel_avg"]
    if fia_k is not None and mask_k is not None:
        result["baseline_kernel_avg"] = fia_k + mask_k
    else:
        result["baseline_kernel_avg"] = None

    s_tot = result["sparse_total_avg"]
    s_ker = result["sparse_kernel_avg"]
    result["speedup_total"] = result["baseline_total_avg"] / s_tot if s_tot and s_tot > 0 else None
    result["speedup_kernel"] = result["baseline_kernel_avg"] / s_ker if s_ker and s_ker > 0 and result["baseline_kernel_avg"] else None

    b_tot = result["baseline_total_avg"]
    b_ker = result["baseline_kernel_avg"]
    sp_tot = result["speedup_total"]
    sp_ker = result["speedup_kernel"]

    def _f(v):
        return f"{v:.2f}" if v is not None else "N/A"
    def _fx(v):
        return f"{v:.2f}x" if v is not None else "N/A"
    print(f"  => sparse: tot={_f(s_tot)}us ker={_f(s_ker)}us  "
          f"baseline: tot={_f(b_tot)}us ker={_f(b_ker)}us  "
          f"speedup: tot={_fx(sp_tot)} ker={_fx(sp_ker)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Sparse FlashAttention test runner")
    parser.add_argument("--mode", choices=["accuracy", "perf", "compare", "all"], default="accuracy")
    parser.add_argument("--case-index", type=int, default=None, help="Run a single case by index (0-based)")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--msprof-log", type=str, default="./log", help="msprof output directory")
    parser.add_argument("--_msprof-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_msprof-target", type=str, default="sparse-fa", help=argparse.SUPPRESS)
    parser.add_argument("--_msprof-repeat", type=int, default=KERNEL_WARMUP + KERNEL_REPEAT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._msprof_worker:
        config = test_configs[args.case_index]
        target = args._msprof_target
        total = args._msprof_repeat

        if target == "sparse-fa":
            data = prepare_data(config)
            for _ in range(total):
                run_kernel(data, config)
                torch.npu.synchronize()
        elif target == "fia":
            q, k, v, mask, seq_lens, scale = build_fia_inputs(
                config["seg_lengths"], config["rules"], config["D"], config["H"])
            for _ in range(total):
                run_fia_once(q, k, v, mask, seq_lens, scale)
                torch.npu.synchronize()
        elif target == "mask":
            mask_tensor, seg_offsets, seg_rules, q_starts, T = build_mask_inputs(
                config["seg_lengths"], config["rules"])
            for _ in range(total):
                run_mask_op_once(mask_tensor, seg_offsets, seg_rules, q_starts)
                torch.npu.synchronize()
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
    compare_results = []

    for idx in indices:
        config = test_configs[idx]
        label = case_label(idx, config)
        print(label)
        torch.npu.synchronize()
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

        if args.mode in ("compare", "all"):
            result = run_compare(data, config, idx, args.msprof_log)
            result["label"] = label
            compare_results.append(result)

    if args.mode in ("accuracy", "all"):
        print(f"\nAccuracy: {passed} passed, {failed} failed, {passed + failed} total")

    if perf_results:
        print_summary(perf_results)

    if compare_results:
        print_compare_summary(compare_results)


if __name__ == "__main__":
    main()
