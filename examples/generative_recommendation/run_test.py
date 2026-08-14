import argparse
import csv
import gc
import glob
import os
import random
import shutil
import subprocess
import sys
import time
import torch
import torch_npu

from testcase import test_configs, prepare_data
from mtgr_ragged_segment_attention import mtgr_ragged_segment_attention
from golden import golden_attention_float64, golden_attention_simulated_kernel

from tilelang_precision_checker import (
    PrecisionLevel,
    dual_benchmark_precision_check,
    bootstrap_retest,
)

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


def _rle(values):
    if not values:
        return []
    result = []
    cur = values[0]
    count = 1
    for v in values[1:]:
        if v == cur:
            count += 1
        else:
            result.append(f"{cur}x{count}" if count >= 2 else str(cur))
            cur = v
            count = 1
    result.append(f"{cur}x{count}" if count >= 2 else str(cur))
    return result


def _serialize_seg_lengths(seg_lengths):
    parts = []
    for seg in seg_lengths:
        if len(seg) <= 2:
            parts.append("|".join(str(x) for x in seg))
        else:
            tokens = [str(seg[0])] + _rle(seg[1:-1]) + [str(seg[-1])]
            parts.append("|".join(tokens))
    return ";".join(parts)


def _serialize_list(lst):
    return "|".join(_rle(lst))


def _case_row(case_index, config):
    seg_lengths = config["seg_lengths"]
    return {
        "case_index": case_index,
        "H": config["H"],
        "D": config["D"],
        "B": len(seg_lengths),
        "num_segs": len(seg_lengths[0]),
        "seg_lengths": _serialize_seg_lengths(seg_lengths),
        "rules": _serialize_list(config["rules"]),
        "matched_prefix_arr": _serialize_list(config["matched_prefix_arr"]),
    }


_ACC_FIELDS = [
    "case_index",
    "H",
    "D",
    "B",
    "num_segs",
    "seg_lengths",
    "rules",
    "matched_prefix_arr",
    "total_runs",
    "passed",
    "status",
    "mare_min",
    "mare_max",
    "mere_min",
    "mere_max",
    "rmse_min",
    "rmse_max",
]

_COMPARE_FIELDS = [
    "case_index",
    "H",
    "D",
    "B",
    "num_segs",
    "seg_lengths",
    "rules",
    "matched_prefix_arr",
    "sparse_total_avg",
    "sparse_kernel_avg",
    "fia_total_avg",
    "fia_kernel_avg",
    "mask_total_avg",
    "mask_kernel_avg",
    "baseline_total_avg",
    "baseline_kernel_avg",
    "speedup_total",
    "speedup_kernel",
]

_COMPARE_FLOAT_FIELDS = [
    "sparse_total_avg",
    "sparse_kernel_avg",
    "fia_total_avg",
    "fia_kernel_avg",
    "mask_total_avg",
    "mask_kernel_avg",
    "baseline_total_avg",
    "baseline_kernel_avg",
]
_COMPARE_RATIO_FIELDS = ["speedup_total", "speedup_kernel"]


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  results written to {path}")


def run_kernel(data, config):
    block_M = config.get("block_M", 128)
    core_num = config.get("core_num", 24)
    num_stages = config.get("num_stages", 14)
    kv_group = data["kv_group"]
    cross_interval = config.get("cross_interval", 2)

    q_seq_starts = data["q_seq_starts_i32"].tolist()
    max_request_len = max(q_seq_starts[b + 1] - q_seq_starts[b] for b in range(len(q_seq_starts) - 1))
    prefix_lens = data["matched_prefix_lens_i32"].tolist()
    if all(p == 0 for p in prefix_lens):
        match_mode = 0
    elif all(p > 0 for p in prefix_lens):
        match_mode = 1
    else:
        match_mode = 2

    output_snd = torch.empty_like(data["query_snd"].npu())
    return mtgr_ragged_segment_attention(
        data["query_snd"].npu(),
        data["key_snd"].npu(),
        data["value_snd"].npu(),
        data["segment_offsets_i32"].npu(),
        data["segment_rules_i32"].npu(),
        data["q_seq_starts_i32"].npu(),
        data["matched_prefix_lens_i32"].npu(),
        match_mode,
        data["key_cache"].npu(),
        data["value_cache"].npu(),
        data["block_table_tensor"].npu(),
        data["block_size"],
        max_request_len,
        data["sm_scale"],
        output_snd,
        block_M=block_M,
        core_num=core_num,
        kv_group=kv_group,
        num_stages=num_stages,
        cross_interval=cross_interval,
    )


def _golden_args(data):
    return (
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


def run_accuracy(data, config, golden_sim, golden_f64, verbose=False):
    _ = run_kernel(data, config)

    torch.npu.synchronize()
    output_snd = run_kernel(data, config)
    torch.npu.synchronize()

    report = dual_benchmark_precision_check(
        npu_output=output_snd.cpu(),
        ref_output=golden_sim.cpu(),
        golden=golden_f64.cpu(),
        level=PrecisionLevel.L0,
        verbose=verbose,
    )
    return report.passed, report


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


def run_msprof_case(case_index, config, log_dir, target="sparse-fa", warmup=KERNEL_WARMUP, repeat=KERNEL_REPEAT):
    kernel_name = KERNEL_NAMES.get(target, "main_kernel")
    target_log_dir = os.path.join(log_dir, target)
    if os.path.exists(target_log_dir):
        shutil.rmtree(target_log_dir)
    os.makedirs(target_log_dir, exist_ok=True)
    cmd = [
        "msprof",
        "op",
        f"--kernel-name={kernel_name}",
        f"--output={target_log_dir}",
        f"--warm-up={warmup}",
        f"--launch-count={repeat}",
        "--kill=on",
        sys.executable,
        os.path.abspath(__file__),
        "--_msprof-worker",
        "--case-index",
        str(case_index),
        "--_msprof-target",
        target,
        "--_msprof-repeat",
        str(warmup + repeat),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  msprof failed (exit {result.returncode})")
        if result.stderr:
            print(f"    {result.stderr.strip()[-200:]}")
        return None
    return parse_kernel_times(target_log_dir)


def print_compare_summary(results):
    name_w = max(len(r["label"]) for r in results)
    name_w = max(name_w, 4)
    header = (
        f"{'Case':<{name_w}}  "
        f"{'s_tot(us)':>10}  {'s_ker(us)':>10}  "
        f"{'f_tot(us)':>10}  {'f_ker(us)':>10}  "
        f"{'m_tot(us)':>10}  {'m_ker(us)':>10}  "
        f"{'b_tot(us)':>10}  {'b_ker(us)':>10}  "
        f"{'sp_tot':>8}  {'sp_ker':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    def _fmt(r, key):
        v = r.get(key)
        return f"{v:.2f}" if v is not None else "-"

    def _fmtx(r, key):
        v = r.get(key)
        return f"{v:.2f}x" if v is not None else "-"

    for r in results:
        print(
            f"{r['label']:<{name_w}}  "
            f"{_fmt(r, 'sparse_total_avg'):>10}  {_fmt(r, 'sparse_kernel_avg'):>10}  "
            f"{_fmt(r, 'fia_total_avg'):>10}  {_fmt(r, 'fia_kernel_avg'):>10}  "
            f"{_fmt(r, 'mask_total_avg'):>10}  {_fmt(r, 'mask_kernel_avg'):>10}  "
            f"{_fmt(r, 'baseline_total_avg'):>10}  {_fmt(r, 'baseline_kernel_avg'):>10}  "
            f"{_fmtx(r, 'speedup_total'):>8}  {_fmtx(r, 'speedup_kernel'):>8}"
        )
    print(sep)


_TARGET_KEY = {"sparse-fa": "sparse", "fia": "fia", "mask": "mask"}


def _run_target_perf(target, data, config, case_index, log_dir):
    if target == "sparse-fa":
        total_us = run_perf_timed(data, config, TOTAL_WARMUP, TOTAL_REPEAT)
    elif target == "fia":
        q, k, v, mask, seq_lens, scale = build_fia_inputs(config["seg_lengths"], config["rules"], config["D"], config["H"])
        total_us = run_fia_timed(q, k, v, mask, seq_lens, scale, warmup=TOTAL_WARMUP, repeat=TOTAL_REPEAT)
    elif target == "mask":
        mask_tensor, seg_offsets, seg_rules, q_starts, _ = build_mask_inputs(config["seg_lengths"], config["rules"])
        total_us = run_mask_op_timed(mask_tensor, seg_offsets, seg_rules, q_starts, warmup=TOTAL_WARMUP, repeat=TOTAL_REPEAT)
    else:
        raise ValueError(f"Unknown target: {target}")

    total_avg = total_us / TOTAL_REPEAT
    entries = run_msprof_case(case_index, config, log_dir, target=target)
    kernel_avg = sum(e["duration_us"] for e in entries) / KERNEL_REPEAT if entries else None
    return total_avg, kernel_avg


def run_compare(data, config, case_index, log_dir):
    result = {}

    for target in ("sparse-fa", "fia", "mask"):
        key = _TARGET_KEY[target]
        total_avg, kernel_avg = _run_target_perf(target, data, config, case_index, log_dir)
        result[f"{key}_total_avg"] = total_avg
        result[f"{key}_kernel_avg"] = kernel_avg

    result["baseline_total_avg"] = result["fia_total_avg"] + result["mask_total_avg"]

    fia_k = result["fia_kernel_avg"]
    mask_k = result["mask_kernel_avg"]
    result["baseline_kernel_avg"] = fia_k + mask_k if fia_k is not None and mask_k is not None else None

    s_tot = result["sparse_total_avg"]
    s_ker = result["sparse_kernel_avg"]
    b_tot = result["baseline_total_avg"]
    b_ker = result["baseline_kernel_avg"]
    result["speedup_total"] = b_tot / s_tot if s_tot and s_tot > 0 else None
    result["speedup_kernel"] = b_ker / s_ker if s_ker and s_ker > 0 and b_ker else None

    def _f(v):
        return f"{v:.2f}" if v is not None else "N/A"

    def _fx(v):
        return f"{v:.2f}x" if v is not None else "N/A"

    print(
        f"  => sparse: tot={_f(s_tot)}us ker={_f(s_ker)}us  "
        f"baseline: tot={_f(b_tot)}us ker={_f(b_ker)}us  "
        f"speedup: tot={_fx(result['speedup_total'])} ker={_fx(result['speedup_kernel'])}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Sparse FlashAttention test runner")
    parser.add_argument("--mode", choices=["accuracy", "compare"], default="accuracy")
    parser.add_argument("--case-index", type=int, default=None, help="Run a single case by index (0-based)")
    parser.add_argument("--case-indices", type=str, default=None, help="Run multiple cases by comma-separated indices (e.g. '0,3,8,22')")
    parser.add_argument("--acc-seeds", type=int, default=50, help="精度测试随机种子数")
    parser.add_argument("--acc-repeat", type=int, default=10, help="每个种子下重复跑次数")
    parser.add_argument("--acc-verbose-fail", action="store_true", help="精度失败时打印首个失败用例的完整 precision report")
    parser.add_argument("--no-bootstrap", action="store_true", help="关闭 Bootstrap 复检（调试用，标准要求默认开启）")
    parser.add_argument("--all-multi-seed", action="store_true", help="强制所有 case 跑多种子（忽略 multi_seed flag）")
    parser.add_argument("--all-perf", action="store_true", help="强制所有 case 跑性能测试（忽略 perf_test flag）")
    parser.add_argument("--msprof-log", type=str, default="./log", help="msprof output directory")
    parser.add_argument("--output-dir", type=str, default="./results", help="CSV results output directory")
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
            q, k, v, mask, seq_lens, scale = build_fia_inputs(config["seg_lengths"], config["rules"], config["D"], config["H"])
            for _ in range(total):
                run_fia_once(q, k, v, mask, seq_lens, scale)
                torch.npu.synchronize()
        elif target == "mask":
            mask_tensor, seg_offsets, seg_rules, q_starts, T = build_mask_inputs(config["seg_lengths"], config["rules"])
            for _ in range(total):
                run_mask_op_once(mask_tensor, seg_offsets, seg_rules, q_starts)
                torch.npu.synchronize()
        return

    if args.case_indices is not None:
        indices = []
        for part in args.case_indices.split(","):
            part = part.strip()
            if not part:
                continue
            idx = int(part)
            if idx < 0 or idx >= len(test_configs):
                print(f"Error: case index {idx} out of range (0~{len(test_configs) - 1})")
                return
            indices.append(idx)
        if not indices:
            print("Error: --case-indices provided but no valid indices parsed")
            return
        print(f"Running {len(indices)} selected cases: {indices}")
    elif args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(test_configs):
            print(f"Error: case-index must be 0~{len(test_configs) - 1}")
            return
        indices = [args.case_index]
    else:
        indices = list(range(len(test_configs)))

    passed, failed = 0, 0
    compare_results = []
    acc_results = []

    for idx in indices:
        config = test_configs[idx]
        label = case_label(idx, config)
        print(label)
        torch.npu.synchronize()
        data = prepare_data(config)

        if args.mode == "accuracy":
            case_pass = 0
            case_total = 0
            mare_ratio_list, mere_ratio_list, rmse_ratio_list = [], [], []
            mare_list, mere_list, rmse_list = [], [], []
            first_fail_info = None
            first_fail_report = None
            hard_fail = False
            effective_seeds = args.acc_seeds if (config.get("multi_seed", False) or args.all_multi_seed) else 1
            for _ in range(effective_seeds):
                seed = random.randint(0, 2**31 - 1)
                data = prepare_data(config, seed)
                ga = _golden_args(data)
                golden_f64 = golden_attention_float64(*ga)
                golden_sim = golden_attention_simulated_kernel(*ga)
                for r in range(args.acc_repeat):
                    ok, report = run_accuracy(data, config, golden_sim, golden_f64, verbose=False)
                    case_total += 1
                    case_pass += int(ok)
                    mare_ratio_list.append(report.mare_ratio)
                    mere_ratio_list.append(report.mere_ratio)
                    rmse_ratio_list.append(report.rmse_ratio)
                    mare_list.append(report.mare_npu)
                    mere_list.append(report.mere_npu)
                    rmse_list.append(report.rmse_npu)
                    print(
                        f"    index={r} seed={seed} run={r}: {'PASS' if ok else 'FAIL'} "
                        f"MARE_r={report.mare_ratio:.4e} MERE_r={report.mere_ratio:.4e} "
                        f"RMSE_r={report.rmse_ratio:.4e}"
                    )
                    if not ok and first_fail_info is None:
                        first_fail_info = (
                            f"seed={seed} run={r} "
                            f"MARE_ratio={report.mare_ratio:.4f} "
                            f"MERE_ratio={report.mere_ratio:.4f} "
                            f"RMSE_ratio={report.rmse_ratio:.4f} "
                            f"small_val={'PASS' if report.small_value_passed else 'FAIL'}"
                            f"(npu={report.small_value_error_count_npu},"
                            f"ref={report.small_value_error_count_ref}) "
                            f"inf_nan={'PASS' if report.inf_nan_match else 'FAIL'}"
                        )
                        first_fail_report = report
                    if not ok and not hard_fail and (not report.inf_nan_match or not report.small_value_passed):
                        hard_fail = True
                del golden_f64, golden_sim
                gc.collect()
            print(
                f"  accuracy: {case_pass}/{case_total} passed "
                f"(seeds={effective_seeds}, repeat={args.acc_repeat}), "
                f"MARE_r[{min(mare_ratio_list):.4e},{max(mare_ratio_list):.4e}] "
                f"MERE_r[{min(mere_ratio_list):.4e},{max(mere_ratio_list):.4e}] "
                f"RMSE_r[{min(rmse_ratio_list):.4e},{max(rmse_ratio_list):.4e}]"
            )
            if first_fail_info:
                print(f"    first fail: {first_fail_info}")
                if args.acc_verbose_fail and first_fail_report is not None:
                    print(first_fail_report.summary())

            if case_pass == case_total:
                case_passed = True
            elif hard_fail:
                case_passed = False
                print("  hard fail: INF/NAN or SmallValue failed, skip bootstrap")
            elif not args.no_bootstrap:
                bs = bootstrap_retest(
                    mare_ratio_list,
                    mere_ratio_list,
                    rmse_ratio_list,
                    verbose=True,
                )
                case_passed = bs["passed"]
            else:
                case_passed = False

            if case_passed:
                passed += 1
            else:
                failed += 1

            acc_results.append(
                {
                    **_case_row(idx, config),
                    "total_runs": case_total,
                    "passed": case_pass,
                    "status": "PASS" if case_passed else "FAIL",
                    "mare_min": f"{min(mare_list):.4e}",
                    "mare_max": f"{max(mare_list):.4e}",
                    "mere_min": f"{min(mere_list):.4e}",
                    "mere_max": f"{max(mere_list):.4e}",
                    "rmse_min": f"{min(rmse_list):.4e}",
                    "rmse_max": f"{max(rmse_list):.4e}",
                }
            )

        if args.mode == "compare":
            if not config.get("perf_test", False) and not args.all_perf:
                print(f"  skipped (perf_test=False)")
                del data
                gc.collect()
                torch.npu.empty_cache()
                continue
            result = run_compare(data, config, idx, args.msprof_log)
            result["label"] = label
            result.update(_case_row(idx, config))
            compare_results.append(result)

        del data
        gc.collect()
        torch.npu.empty_cache()

    if args.mode == "accuracy":
        print(f"\nAccuracy: {passed} passed, {failed} failed, {passed + failed} total")

    if compare_results:
        print_compare_summary(compare_results)

    if acc_results:
        rows = []
        for r in acc_results:
            rows.append({k: r.get(k, "") for k in _ACC_FIELDS})
        _write_csv(os.path.join(args.output_dir, "accuracy_results.csv"), rows, _ACC_FIELDS)

    if compare_results:
        rows = []
        for r in compare_results:
            row = {k: r.get(k, "") for k in _COMPARE_FIELDS}
            for k in _COMPARE_FLOAT_FIELDS:
                v = r.get(k)
                if v is not None:
                    row[k] = f"{v:.2f}"
            for k in _COMPARE_RATIO_FIELDS:
                v = r.get(k)
                if v is not None:
                    row[k] = f"{v:.4f}"
            rows.append(row)
        _write_csv(os.path.join(args.output_dir, "compare_results.csv"), rows, _COMPARE_FIELDS)


if __name__ == "__main__":
    main()
