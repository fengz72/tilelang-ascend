# msprof op --kernel-name="main_kernel" --output="./log" python examples/sparse-fa/sparse_fa_v5.py
# msprof op simulator --soc-version=Ascend910B2 --kernel-name="main_kernel" --output="./log" python examples/sparse-fa/sparse_fa_v5.py

import torch
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
from golden import golden_attention_float64
from testcase import prepare_data

torch.manual_seed(41)
tilelang.disable_cache()
tilelang.cache.clear_cache()

# ---------------------------------------------------------------------------
# 常量与优化 Pass 配置
# ---------------------------------------------------------------------------
NEG_INF = -(2.0**30)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@tilelang.jit(
    pass_configs=PASS_CONFIGS,
)
def high_perf_mtgr_sparse_attn_kernel(
    heads,
    dim,
    num_blocks,
    kv_group=1,
    sm_scale=None,
    block_M=128,
    block_N=128,
    core_num=24,
    num_stages=14,
    cross_interval=2,
    max_splits=32,
):
    sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    dtype = "bfloat16"
    accum_dtype = "float32"

    batch = T.symbolic("batch")
    total_q = T.symbolic("total_q")
    max_blocks = T.symbolic("max_blocks")
    max_segs = T.symbolic("max_segs")

    kv_heads = heads // kv_group
    half_M = block_M // 2
    sub_M = half_M // 2

    # 跨核信号与单核内信号定义
    SEM_WS1_C2V = 0
    SEM_WS1_V2C = 1
    SEM_WS2_V2C = 2
    SEM_WS2_C2V = 3
    SEM_WS3_C2V = 4
    SEM_WS3_V2C = 5

    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_L0AB = 3
    SIG_L0C = 5
    SIG_Q_L1 = 7

    SIG_IO_UB = 0
    SIG_S_HALF = 1
    SIG_O_READY = 3
    SIG_V_S_READY = 5

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, heads, dim], dtype),
        K: T.Tensor([total_q, kv_heads, dim], dtype),
        V: T.Tensor([total_q, kv_heads, dim], dtype),
        Output: T.Tensor([total_q, heads, dim], dtype),
        q_seq_starts: T.Tensor([batch + 1], "int32"),
        split_points: T.Tensor([batch, max_splits], "int32"),
        tiles_prefix_sum: T.Tensor([batch + 1], "int32"),
        segment_offsets: T.Tensor([batch, max_segs + 1], "int32"),
        segment_rules: T.Tensor([max_segs], "int32"),
        workspace_1: T.Tensor([core_num, num_stages, block_M, block_N], dtype),
        workspace_2: T.Tensor([core_num, num_stages, block_M, block_N], dtype),
        workspace_3: T.Tensor([core_num, num_stages, block_M, dim], dtype),
        key_cache: T.Tensor([num_blocks, block_N, kv_heads, dim], dtype),
        value_cache: T.Tensor([num_blocks, block_N, kv_heads, dim], dtype),
        block_table: T.Tensor([batch, max_blocks], "int32"),
        prefix_lens: T.Tensor([batch], "int32"),
        total_seq_tiles: T.int32
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )

            l0a_s = T.alloc_L0A([2, block_M, dim], dtype)
            l0b_s = T.alloc_L0B([2, dim, block_N], dtype)
            l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            l0a_o = T.alloc_L0A([2, block_M, block_N], dtype)
            l0b_o = T.alloc_L0B([2, block_N, dim], dtype)
            l0c_o_0 = T.alloc_L0C([block_M, dim], accum_dtype)
            l0c_o_1 = T.alloc_L0C([block_M, dim], accum_dtype)

            acc_o = T.alloc_ub([half_M, dim], accum_dtype)
            r_factors = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, half_M, 1], accum_dtype)

            io_buf = T.alloc_ub([half_M, block_N], dtype)
            acc_s_half = T.alloc_ub([half_M, block_N], dtype)
            work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)

            bcast_buf = T.alloc_ub([half_M, dim], accum_dtype)
            o_io_buf = T.alloc_ub([half_M, dim], dtype)
            o_work_buf = T.alloc_ub([half_M, dim], accum_dtype)
            o_acc_half = T.alloc_ub([half_M, dim], dtype)

            # 用于存储平铺流水线的真实有效 k 索引
            valid_k_indices = T.alloc_ub([max_splits], "int32")

            T.annotate_address(
                {
                    l0a_s: 0,
                    l0a_o: 0,
                    l0b_s: 0,
                    l0b_o: 0,
                    l0c_s: 0,
                    l0c_o_0: 0,
                    l0c_o_1: block_M * block_N * 4,
                    acc_o: 0,
                    r_factors: half_M * dim * 4,
                    sumexp_is: half_M * dim * 4 + num_stages * half_M * 4,
                    sumexp: half_M * dim * 4 + num_stages * half_M * 4 * 2,
                    neg_sm: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 4,
                    io_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12,
                    o_io_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12,
                    acc_s_half: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 2,
                    o_acc_half: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 2,
                    work_ub: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 4,
                    o_work_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 4,
                    buf_2d: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 8,
                    bcast_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 8,
                }
            )

            b_i = T.alloc_var("int32", init=0)
            seg_id = T.alloc_var("int32", init=0)
            next_seg_first_tile = T.alloc_var("int32", init=0)
            valid_k_total = T.alloc_var("int32", init=0)
            scan_count = T.alloc_var("int32", init=0)
            scan_k_idx = T.alloc_var("int32", init=-1)

            total_tasks = total_seq_tiles * heads
            my_iters = T.if_then_else(
                cid < total_tasks,
                T.ceildiv(total_tasks - cid, core_num),
                0,
            )

            # =========================================================================
            # Scope C: Cube 核心 (负责张量搬运与矩阵乘)
            # =========================================================================
            with T.Scope("C"):
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for core_index in T.serial(my_iters):
                    pid = cid + core_index * core_num
                    tile_id = pid // heads
                    h_i = pid % heads
                    h_kv = h_i // kv_group

                    b_i = 0
                    for _b in T.serial(batch):
                        b_i = T.if_then_else(tile_id >= tiles_prefix_sum[_b + 1], _b + 1, b_i)
                    s_local = tile_id - tiles_prefix_sum[b_i]
                    q_start = split_points[b_i, s_local]
                    q_end = split_points[b_i, s_local + 1]
                    q_tile_size = q_end - q_start

                    seg_id = 0
                    for _seg in T.serial(max_segs - 1):
                        seg_id = T.if_then_else(q_start >= segment_offsets[b_i, _seg + 1], _seg + 1, seg_id)

                    rule = segment_rules[seg_id]
                    prefix_len_b = prefix_lens[b_i]
                    q_start_live = T.if_then_else(q_start >= prefix_len_b, q_start, prefix_len_b)
                    q_tile_size_live = q_end - q_start_live
                    q_tile_size_live = T.if_then_else(q_tile_size_live > 0, q_tile_size_live, 0)
                    q_packed_start = q_seq_starts[b_i] + q_start_live - prefix_len_b
                    seg_end_offset = segment_offsets[b_i, seg_id + 1]

                    tiles_this_batch = tiles_prefix_sum[b_i + 1] - tiles_prefix_sum[b_i]
                    next_seg_first_tile = tiles_this_batch
                    for _k in T.serial(max_splits):
                        next_seg_first_tile = T.if_then_else(
                            (_k > s_local)
                            & (_k <= tiles_this_batch)
                            & (split_points[b_i, _k] >= seg_end_offset)
                            & (_k < next_seg_first_tile),
                            _k,
                            next_seg_first_tile,
                        )
                    kv_iter_end = T.if_then_else(rule == 1, next_seg_first_tile, s_local + 1)

                    # 计算有效 K 切片数量（Cube 无法读写 UB，仅计数）
                    seg_start = segment_offsets[b_i, seg_id]
                    seg_end = segment_offsets[b_i, seg_id + 1]
                    max_row_pos = q_start + q_tile_size - 1
                    valid_k_total = 0
                    for k_i in T.serial(kv_iter_end):
                        kv_start_k = split_points[b_i, k_i]
                        kv_end_k = split_points[b_i, k_i + 1]
                        process_cond = (
                            ((rule == 0) & (kv_start_k <= max_row_pos))
                            | ((rule == 1) & (kv_start_k < seg_end))
                            | ((rule == 2) & ((kv_start_k < seg_start) | ((kv_start_k < q_end) & (kv_end_k > q_start))))
                        )
                        if process_cond:
                            valid_k_total += 1

                    # 载入 Q
                    T.copy(Q[q_packed_start : q_packed_start + q_tile_size_live, h_i, :], q_l1[:, :])
                    T.barrier_all()
                    num_outer = T.ceildiv(valid_k_total, num_stages)

                    for k_outer in T.serial(num_outer):
                        _remaining = valid_k_total - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # ---------------------------------
                        # GEMM1: S = Q * K^T (写入 workspace_1)
                        # ---------------------------------
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            scan_count = 0
                            scan_k_idx = -1
                            target = k_outer * num_stages + i
                            for k_scan in T.serial(kv_iter_end):
                                kv_scan_start = split_points[b_i, k_scan]
                                kv_scan_end = split_points[b_i, k_scan + 1]
                                process_cond_scan = (
                                    ((rule == 0) & (kv_scan_start <= max_row_pos))
                                    | ((rule == 1) & (kv_scan_start < seg_end))
                                    | ((rule == 2) & ((kv_scan_start < seg_start) | ((kv_scan_start < q_end) & (kv_scan_end > q_start))))
                                )
                                scan_k_idx = T.if_then_else(process_cond_scan & (scan_count == target), k_scan, scan_k_idx)
                                scan_count = T.if_then_else(process_cond_scan, scan_count + 1, scan_count)
                            kv_start = split_points[b_i, scan_k_idx]
                            kv_size = split_points[b_i, scan_k_idx + 1] - kv_start
                            prefix_len_b = prefix_lens[b_i]

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            if kv_start < prefix_len_b:
                                cache_block_idx = kv_start // block_N
                                physical_block = block_table[b_i, cache_block_idx]
                                block_offset_start = kv_start % block_N
                                T.copy(
                                    key_cache[physical_block, block_offset_start : block_offset_start + kv_size, h_kv, :],
                                    k_l1[:, :],
                                )
                            else:
                                kv_packed_start = q_seq_starts[b_i] + kv_start - prefix_len_b
                                T.copy(
                                    K[kv_packed_start : kv_packed_start + kv_size, h_kv, :],
                                    k_l1[:, :],
                                )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a_s[side, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b_s[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)

                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # ---------------------------------
                        # GEMM2: O = P * V (写入 workspace_3)
                        # ---------------------------------
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            scan_count = 0
                            scan_k_idx = -1
                            target = k_outer * num_stages + i
                            for k_scan in T.serial(kv_iter_end):
                                kv_scan_start = split_points[b_i, k_scan]
                                kv_scan_end = split_points[b_i, k_scan + 1]
                                process_cond_scan = (
                                    ((rule == 0) & (kv_scan_start <= max_row_pos))
                                    | ((rule == 1) & (kv_scan_start < seg_end))
                                    | ((rule == 2) & ((kv_scan_start < seg_start) | ((kv_scan_start < q_end) & (kv_scan_end > q_start))))
                                )
                                scan_k_idx = T.if_then_else(process_cond_scan & (scan_count == target), k_scan, scan_k_idx)
                                scan_count = T.if_then_else(process_cond_scan, scan_count + 1, scan_count)
                            kv_start = split_points[b_i, scan_k_idx]
                            kv_size = split_points[b_i, scan_k_idx + 1] - kv_start
                            prefix_len_b = prefix_lens[b_i]

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            if kv_start < prefix_len_b:
                                cache_block_idx = kv_start // block_N
                                physical_block = block_table[b_i, cache_block_idx]
                                block_offset_start = kv_start % block_N
                                T.copy(
                                    value_cache[physical_block, block_offset_start : block_offset_start + kv_size, h_kv, :],
                                    v_l1[:, :],
                                )
                            else:
                                kv_packed_start = q_seq_starts[b_i] + kv_start - prefix_len_b
                                T.copy(
                                    V[kv_packed_start : kv_packed_start + kv_size, h_kv, :],
                                    v_l1[:, :],
                                )
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            if side == 0:
                                T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o_0[:, :], init=True)
                            else:
                                T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o_1[:, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            if side == 0:
                                T.copy(l0c_o_0[:, :], workspace_3[cid, i, :, :])
                            else:
                                T.copy(l0c_o_1[:, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)

                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)
                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                # 回收初始化的 Signal
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # =========================================================================
            # Scope V: Vector 核心 (负责生成 Mask、Softmax 和累加降级)
            # =========================================================================
            with T.Scope("V"):
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for core_index in T.serial(my_iters):
                    pid = cid + core_index * core_num
                    tile_id = pid // heads
                    h_i = pid % heads
                    h_kv = h_i // kv_group

                    # 重新解析位置（必须与 Scope C 完全对齐）
                    b_i = 0
                    for _b in T.serial(batch):
                        b_i = T.if_then_else(tile_id >= tiles_prefix_sum[_b + 1], _b + 1, b_i)
                    s_local = tile_id - tiles_prefix_sum[b_i]
                    q_start = split_points[b_i, s_local]
                    q_end = split_points[b_i, s_local + 1]
                    q_tile_size = q_end - q_start

                    seg_id = 0
                    for _seg in T.serial(max_segs - 1):
                        seg_id = T.if_then_else(q_start >= segment_offsets[b_i, _seg + 1], _seg + 1, seg_id)

                    rule = segment_rules[seg_id]
                    prefix_len_b = prefix_lens[b_i]
                    q_start_live = T.if_then_else(q_start >= prefix_len_b, q_start, prefix_len_b)
                    q_tile_size_live = q_end - q_start_live
                    q_tile_size_live = T.if_then_else(q_tile_size_live > 0, q_tile_size_live, 0)
                    q_packed_start = q_seq_starts[b_i] + q_start_live - prefix_len_b
                    seg_end_offset = segment_offsets[b_i, seg_id + 1]

                    tiles_this_batch = tiles_prefix_sum[b_i + 1] - tiles_prefix_sum[b_i]
                    next_seg_first_tile = tiles_this_batch
                    for _k in T.serial(max_splits):
                        next_seg_first_tile = T.if_then_else(
                            (_k > s_local)
                            & (_k <= tiles_this_batch)
                            & (split_points[b_i, _k] >= seg_end_offset)
                            & (_k < next_seg_first_tile),
                            _k,
                            next_seg_first_tile,
                        )
                    kv_iter_end = T.if_then_else(rule == 1, next_seg_first_tile, s_local + 1)

                    valid_k_total = 0
                    for k_i in T.serial(kv_iter_end):
                        kv_start = split_points[b_i, k_i]
                        kv_end = split_points[b_i, k_i + 1]
                        seg_start = segment_offsets[b_i, seg_id]
                        seg_end = segment_offsets[b_i, seg_id + 1]
                        max_row_pos = q_start + q_tile_size - 1

                        process_cond = (
                            ((rule == 0) & (kv_start <= max_row_pos))
                            | ((rule == 1) & (kv_start < seg_end))
                            | ((rule == 2) & ((kv_start < seg_start) | ((kv_start < q_end) & (kv_end > q_start))))
                        )
                        if process_cond:
                            valid_k_indices[valid_k_total] = k_i
                            valid_k_total += 1

                    T.pipe_barrier("v")
                    T.tile.fill(acc_o, 0.0)
                    T.pipe_barrier("v")
                    T.tile.fill(sumexp, 0.0)
                    T.pipe_barrier("v")
                    T.tile.fill(neg_sm, 2**30)

                    num_outer = T.ceildiv(valid_k_total, num_stages)
                    for k_outer in T.serial(num_outer):
                        _remaining = valid_k_total - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- Softmax Batch 处理 ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur

                            k_idx = valid_k_indices[k_outer * num_stages + i]
                            kv_start = split_points[b_i, k_idx]
                            kv_size = split_points[b_i, k_idx + 1] - kv_start
                            seg_start = segment_offsets[b_i, seg_id]
                            seg_end = segment_offsets[b_i, seg_id + 1]
                            

                            # 【核心优化】计算掩盖 (Hiding Computation)：在等 Cube 前利用算力资源提前生成 MASK
                            T.pipe_barrier("v")
                            T.tile.fill(buf_2d, NEG_INF)
                            for row in T.serial(half_M):
                                row_abs_pos = q_start + vid * half_M + row
                                if rule == 0:
                                    raw_len = row_abs_pos - kv_start + 1
                                    fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        # T.pipe_barrier("v")
                                        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
                                elif rule == 1:
                                    raw_len = seg_end - kv_start
                                    fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        # T.pipe_barrier("v")
                                        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
                                elif rule == 2:
                                    raw_len = seg_start - kv_start
                                    fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        # T.pipe_barrier("v")
                                        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
                                    diag_col = row_abs_pos - kv_start
                                    
                                    if (diag_col >= 0) & (diag_col < kv_size):
                                        T.set_flag("v", "s", SIG_V_S_READY)
                                        T.wait_flag("v", "s", SIG_V_S_READY)
                                        buf_2d[row, diag_col] = 0.0

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)

                            T.copy(workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.pipe_barrier("v")
                            T.copy(io_buf, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            # 施加 Mask 到矩阵 S
                            T.pipe_barrier("v")
                            T.tile.add(work_ub, work_ub, buf_2d)
                            T.pipe_barrier("v")
                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.pipe_barrier("v")
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.pipe_barrier("v")
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                            T.pipe_barrier("v")
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.pipe_barrier("v")
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.pipe_barrier("v")
                            T.tile.exp(work_ub, buf_2d)

                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.pipe_barrier("v")
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(acc_s_half, workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)

                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                            T.pipe_barrier("v")
                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.pipe_barrier("v")
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- O 累加 Batch ---
                        for i in T.serial(batch_iters):
                            T.pipe_barrier("v")
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.pipe_barrier("v")
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.pipe_barrier("v")
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.pipe_barrier("v")
                            T.tile.broadcast(bcast_buf, r_factors[i, :, :])
                            T.pipe_barrier("v")
                            T.tile.mul(acc_o, acc_o, bcast_buf)

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :], o_io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.pipe_barrier("v")
                            T.copy(o_io_buf, o_work_buf)
                            T.set_flag("V", "MTE2", SIG_IO_UB)
                            T.pipe_barrier("v")
                            T.tile.add(acc_o, acc_o, o_work_buf)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)
                    
                    for sub in T.serial(2):
                        row_start = sub * sub_M
                        T.pipe_barrier("v")
                        T.tile.max(sumexp[row_start : row_start + sub_M, :], sumexp[row_start : row_start + sub_M, :], 1.0)
                        T.pipe_barrier("v")
                        T.tile.broadcast(bcast_buf[row_start : row_start + sub_M, :], sumexp[row_start : row_start + sub_M, :])
                        T.pipe_barrier("v")
                        T.tile.div(acc_o[row_start : row_start + sub_M, :], acc_o[row_start : row_start + sub_M, :], bcast_buf[row_start : row_start + sub_M, :])
                        T.pipe_barrier("v")
                        T.copy(
                            acc_o[row_start : row_start + sub_M, :],
                            o_acc_half[row_start : row_start + sub_M, :],
                        )
                        T.set_flag("V", "MTE3", SIG_O_READY + sub)

                        T.wait_flag("V", "MTE3", SIG_O_READY + sub)
                        valid_rows_sub = T.if_then_else(
                            q_tile_size_live >= vid * half_M + row_start + sub_M,
                            sub_M,
                            T.if_then_else(
                                q_tile_size_live > vid * half_M + row_start,
                                q_tile_size_live - vid * half_M - row_start,
                                0,
                            ),
                        )
                        h_i_out = (cid + core_index * core_num) % heads
                        output_packed_start = q_packed_start + vid * half_M + row_start

                        if valid_rows_sub > 0:
                            T.copy(
                                o_acc_half[row_start : row_start + valid_rows_sub, :],
                                Output[
                                    output_packed_start : output_packed_start + valid_rows_sub,
                                    h_i_out,
                                    :,
                                ],
                            )

                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


# ---------------------------------------------------------------------------
# Host 端 Wrapper（集成动态长度推导与 Workspace 分配）
# ---------------------------------------------------------------------------
def compute_split_points_batch(segment_offsets_i32, block_M):
    B = segment_offsets_i32.size(0)
    seg_lengths = segment_offsets_i32[:, 1:] - segment_offsets_i32[:, :-1]
    seg_starts = segment_offsets_i32[:, :-1]

    num_splits_per_seg = torch.clamp((seg_lengths + block_M - 1) // block_M, min=0)
    max_splits_per_seg = num_splits_per_seg.max().item()

    if max_splits_per_seg == 0:
        return (
            torch.zeros((B, 1), dtype=torch.int32, device=segment_offsets_i32.device),
            torch.zeros(B, dtype=torch.int32, device=segment_offsets_i32.device),
        )

    k = torch.arange(1, max_splits_per_seg + 1, dtype=torch.int32, device=segment_offsets_i32.device)
    k_block = (k * block_M).view(1, 1, -1)
    capped = torch.minimum(k_block, seg_lengths.unsqueeze(-1))
    split_points_3d = seg_starts.unsqueeze(-1) + capped

    valid = k.view(1, 1, -1) <= num_splits_per_seg.unsqueeze(-1)

    SENTINEL = 0x7FFFFFFF
    split_points_flat = torch.where(
        valid, split_points_3d, torch.full_like(split_points_3d, SENTINEL)
    ).view(B, -1)

    split_points_sorted, _ = split_points_flat.sort(dim=1)

    num_valid = valid.view(B, -1).sum(dim=1)
    max_num_valid = num_valid.max().item()

    valid_part = split_points_sorted[:, :max_num_valid]
    valid_part = torch.where(
        valid_part == SENTINEL, torch.zeros_like(valid_part), valid_part
    )

    zeros = torch.zeros((B, 1), dtype=torch.int32, device=segment_offsets_i32.device)
    split_points_i32 = torch.cat([zeros, valid_part], dim=1)

    return split_points_i32, num_valid.to(torch.int32)


def high_perf_sparse_attn_wrapper(
    query,
    key,
    value,
    segment_offsets_i32,
    segment_rules_i32,
    q_seq_starts_i32,
    sm_scale,
    num_blocks,
    key_cache,
    value_cache,
    block_table_i32,
    matched_prefix_lens_i32,
    block_M=128,
    block_N=128,
    core_num=24,
    num_stages=4,
    kv_group=1,
    cross_interval=2,
):
    B = segment_offsets_i32.size(0)
    H = query.size(1)
    D = query.size(2)
    max_segs = segment_offsets_i32.size(1) - 1

    assert segment_offsets_i32.size(1) == max_segs + 1, \
        f"segment_offsets_i32 shape {segment_offsets_i32.shape} != [batch={B}, max_segs+1={max_segs + 1}]"
    assert segment_rules_i32.size(0) == max_segs, \
        f"segment_rules_i32 shape {segment_rules_i32.shape} != [max_segs={max_segs}]"

    split_points_i32, num_tiles_per_batch = compute_split_points_batch(
        segment_offsets_i32, block_M
    )

    max_splits = split_points_i32.size(1)
    total_seq_tiles = num_tiles_per_batch.sum().item()

    tiles_prefix_sum_i32 = torch.zeros(
        B + 1, dtype=torch.int32, device=query.device
    )
    tiles_prefix_sum_i32[1:] = torch.cumsum(num_tiles_per_batch, dim=0).to(torch.int32)

    ws1 = torch.empty((core_num, num_stages, block_M, block_N), dtype=torch.bfloat16, device=query.device)
    ws2 = torch.empty((core_num, num_stages, block_M, block_N), dtype=torch.bfloat16, device=query.device)
    ws3 = torch.empty((core_num, num_stages, block_M, D), dtype=torch.bfloat16, device=query.device)
    output = torch.empty_like(query)

    func = high_perf_mtgr_sparse_attn_kernel(
        heads=H,
        dim=D,
        num_blocks=num_blocks,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_M=block_M,
        block_N=block_N,
        core_num=core_num,
        num_stages=num_stages,
        max_splits=max_splits,
        cross_interval=cross_interval,
    )

    # print(func.get_kernel_source())

    func(
        query,
        key,
        value,
        output,
        q_seq_starts_i32.to(query.device),
        split_points_i32,
        tiles_prefix_sum_i32,
        segment_offsets_i32,
        segment_rules_i32,
        ws1,
        ws2,
        ws3,
        key_cache,
        value_cache,
        block_table_i32,
        matched_prefix_lens_i32,
        total_seq_tiles
    )

    torch.npu.synchronize()
    return output


def test(config, block_M=128, core_num=24, num_stages=14, cross_interval=2):
    data = prepare_data(config)
    block_N = config.get("block_N", 128)

    torch.npu.synchronize()
    print("init successful!")

    output_snd = high_perf_sparse_attn_wrapper(
        data["query_snd"].npu(),
        data["key_snd"].npu(),
        data["value_snd"].npu(),
        data["segment_offsets_i32"].npu(),
        data["segment_rules_i32"].npu(),
        data["q_seq_starts_i32"].npu(),
        data["sm_scale"],
        num_blocks=data["num_cache_blocks"],
        key_cache=data["key_cache"].npu(),
        value_cache=data["value_cache"].npu(),
        block_table_i32=data["block_table_tensor"].npu(),
        matched_prefix_lens_i32=data["matched_prefix_lens_i32"].npu(),
        block_M=block_M,
        block_N=block_N,
        core_num=core_num,
        kv_group=data["kv_group"],
        num_stages=num_stages,
        cross_interval=cross_interval,
    )

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
    torch.testing.assert_close(ref_output.npu(), output_snd, rtol=1e-2, atol=1e-2)
    print("Test Passed!")


if __name__ == "__main__":
    test_configs = [
        {
            "H": 8,
            "D": 128,
            "seg_lengths": [[1600, 8, 200, 1200]],
            "rules": [0, 1, 2, 2],
            "matched_prefix_arr": [0],
        },
    ]

    for config in test_configs:
        test(config)
