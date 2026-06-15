# sparse_fa_v5_fix_by_xrb 实现说明

> 基于 `sparse_fa_v5.py`，在保持 kernel 内计算 mask 的前提下，通过收紧 rule 1 的 KV 迭代上界和裁剪 rule 2 的无效 KV tile，减少进入 Cube/Vector 流水线的无效计算。

## 1. 结论

`sparse_fa_v5_fix_by_xrb.py` 基于 `sparse_fa_v5.py` 演进，继续保持 kernel 内计算 mask，不在 host 侧预生成 mask。默认 D=128 配置下精度通过，最终 `msprof op` 性能为：

```text
Task Duration(us): 3966.64
Block Dim: 24
Mix Block Dim: 48
```

同环境重跑 `sparse_fa_v5.py` 的基线为 `4650.55 us`，v5_fix_by_xrb 相对提升约 **14.71%**，约 **1.17x**。

## 2. 背景：基线结构与关键概念

v5 是一个 C/V 分离的 fused attention kernel。C 指 Cube scope（负责矩阵乘），V 指 Vector scope（负责 mask、softmax 和输出累加）。两者通过三个 workspace 数组和 cross-core flag 组成流水线。

### 2.1 关键概念

| 名称 | 含义 |
| --- | --- |
| Q tile | 一个 `block_M` 行的 query 块，默认 `block_M = 128` |
| KV tile | 一个 `block_N` 行的 key/value 块，默认 `block_N = 128` |
| valid KV tile | 根据 `segment_rules` 和当前 Q tile 位置，可能参与 attention 的 KV tile |
| `valid_k_total` | 当前 Q tile 在同一个 request batch 内需要处理的有效 KV tile 总数 |
| `num_stages` | kernel 内部一次流水分组最多处理的 KV tile 数，默认 `14` |
| `k_outer` | 第几个 KV tile 流水分组（不是 request batch），只是把 `valid_k_total` 分块处理的循环变量 |
| `batch_iters` | 当前 `k_outer` 分组实际处理的 KV tile 数，最后一组可能小于 `num_stages` |
| workspace slot `i` | 当前分组里第 `i` 个有效 KV tile 对应的中转位置，`i in [0, batch_iters)` |

### 2.2 数据流

每个 KV slot 走完整条 C/V 流水线：

```text
Cube:  Q, K_i  ---> [QK GEMM] ---> workspace_1[i]
Vector:                             workspace_1[i] ---> [mask + online softmax] ---> workspace_2[i]
Cube:                                                                                workspace_2[i], V_i ---> [PV GEMM] ---> workspace_3[i]
Vector:                                                                                                              workspace_3[i] ---> [scale + accumulate]
```

因此，如果一个 KV tile 最终会被 mask 成全无效，它仍然会经历：

```text
K/V 搬运 → QK GEMM → workspace_1 写/读 → mask+softmax → workspace_2 写/读 → PV GEMM → workspace_3 写/读 → output accumulate
```

v5_fix_by_xrb 的核心思路不是"让 mask 算得更快"，而是**更早地把确定全无效的 KV tile 排除在 valid KV 列表之外**，让它不进入这条流水线。

### 2.3 三种 rule 的语义

| rule | 含义 | 允许访问的 K |
| --- | --- | --- |
| 0 | causal | `K <= current_row` |
| 1 | full | `K < segment_end` |
| 2 | prefix + diagonal | `K < seg_start` 或 `K == current_row` |

## 3. 优化点 A：rule 2 的无效 KV tile 剪枝

### 3.1 问题分析

`rule == 2` 的语义是 prefix + diagonal：

```text
对某个 query row r:
  允许访问:
    1. 所有 K < 当前 segment 起点    (prefix)
    2. K == r                        (diagonal)
  不允许访问:
    当前 segment 内除 diagonal 外的其他 K
```

图示：

```text
K 轴:
|<----------- prefix ----------->|<----------- 当前 diagonal segment ----------->|
                                  ^ seg_start

当前 Q tile:
                                  |............. Q tile .............|

允许:
|<----------- prefix ----------->|                 diagonal only

不允许:
                                  | before Q tile | off-diagonal | after Q tile |
```

基线 v5 的有效 KV tile 条件为：

```python
(kv_start <= seg_start) | (kv_start <= max_row_pos)
```

其中 `max_row_pos = q_start + q_tile_size - 1` 是当前 Q tile 最后一行的绝对位置。

**问题**：`kv_start <= max_row_pos` 会把当前 diagonal segment 内、位于当前 Q tile 之前的 KV tile 也纳入计算。这些 tile 的起点确实满足 `<= max_row_pos`，但它们整个区间 `[kv_start, kv_end)` 都在 `q_start` 之前，不包含任何 diagonal 行——根据 rule 2 语义，它们会被完整 mask 掉。

对 diagonal segment 内第 `j` 个 Q tile：

```text
K tiles:
prefix tiles      same-segment past tiles      current overlap      future tiles
[P0][P1]...[Pm]   [D0][D1]...[D(j-1)]          [Dj]                  [D(j+1)]...
 有效 prefix       全块被 mask（浪费！）         diagonal 有效         不参与
```

基线实际计算：`[P0..Pm] + [D0..D(j-1)] + [Dj]`
真正需要计算：`[P0..Pm] + [Dj]`

被浪费的 `[D0..D(j-1)]` 每个都会经历完整的 QK/PV GEMM + workspace 读写链路。

### 3.2 修改方案

将 rule 2 的有效 KV tile 条件改为区间重叠判断：

```python
# v5（旧）
(kv_start_k <= seg_start) | (kv_start_k <= max_row_pos)

# v5_fix_by_xrb（新）
(kv_start_k < seg_start) | ((kv_start_k < q_end) & (kv_end_k > q_start))
```

含义：

| 条件 | 作用 |
| --- | --- |
| `kv_start_k < seg_start` | 保留所有 prefix tile |
| `(kv_start_k < q_end) & (kv_end_k > q_start)` | 仅保留与当前 Q tile 区间有重叠的 KV tile，覆盖 diagonal |

图示：

```text
K tiles:
| prefix | same-segment past | current overlap | future |
|  keep  |       skip        |      keep       | skip   |

Q tile:
                            |------ q_start ... q_end ------|

保留条件:
  prefix:   kv_start < seg_start
  diagonal: [kv_start, kv_end) 与 [q_start, q_end) 相交
```

注意新的条件不再使用 `max_row_pos`。`max_row_pos` 仍保留用于 rule 0 的 causal 判断，但在 rule 2 中已被区间重叠条件取代——这是本次优化的核心。

### 3.3 正确性证明

对跳过的 tile，有：

```text
kv_start >= seg_start   （不在 prefix 内）
且 kv_end <= q_start     （与 Q tile 不相交）
```

因此：
1. 它不属于 prefix，因为 `kv_start >= seg_start`
2. 它不包含任何 diagonal 行，因为整个 tile 的 K 行范围 `[kv_start, kv_end)` 都在 Q tile 起始位置之前
3. 根据 rule 2 语义，该 tile 内所有元素都会被 mask 为 `-inf`

跳过该 tile 不改变 softmax 的有效集合，不改变输出。

### 3.4 修改位置

这个条件变更必须同步应用于 kernel 内 **4 处**，否则 C/V 两侧对 workspace slot 的理解会错位：

| # | 位置 | 行号 (v5_fix_by_xrb) | 作用 |
| --- | --- | --- | --- |
| 1 | Scope C `valid_k_total` 计数 | 187–202 | 决定进入流水线的有效 KV tile 数量 |
| 2 | Scope C QK GEMM 扫描 | 222–237 | 定位第 `target` 个有效 KV tile 的 split_points 索引 |
| 3 | Scope C PV GEMM 扫描 | 277–292 | 同上，PV 阶段的 KV tile 定位 |
| 4 | Scope V `valid_k_indices` 生成 | 386–406 | Vector 侧定位有效 KV tile 以做 mask/softmax/accumulate |

每处的具体改动：

**位置 1 — Scope C 计数**（v5:183–191 → v5_fix_by_xrb:187–202）：

```diff
 for k_i in T.serial(kv_iter_end):
     kv_start_k = split_points[b_i, k_i]
+    kv_end_k = split_points[b_i, k_i + 1]
     process_cond = (
         ((rule == 0) & (kv_start_k <= max_row_pos))
         | ((rule == 1) & (kv_start_k < seg_end))
-        | ((rule == 2) & ((kv_start_k <= seg_start) | (kv_start_k <= max_row_pos)))
+        | (
+            (rule == 2)
+            & (
+                (kv_start_k < seg_start)
+                | ((kv_start_k < q_end) & (kv_end_k > q_start))
+            )
+        )
     )
```

**位置 2 — Scope C QK 扫描**（v5:211–218 → v5_fix_by_xrb:222–237）：

```diff
 for k_scan in T.serial(kv_iter_end):
     kv_scan_start = split_points[b_i, k_scan]
+    kv_scan_end = split_points[b_i, k_scan + 1]
     process_cond_scan = (
         ((rule == 0) & (kv_scan_start <= max_row_pos))
         | ((rule == 1) & (kv_scan_start < seg_end))
-        | ((rule == 2) & ((kv_scan_start <= seg_start) | (kv_scan_start <= max_row_pos)))
+        | (
+            (rule == 2)
+            & (
+                (kv_scan_start < seg_start)
+                | ((kv_scan_start < q_end) & (kv_scan_end > q_start))
+            )
+        )
     )
```

**位置 3 — Scope C PV 扫描**（v5:259–266 → v5_fix_by_xrb:277–292）：与位置 2 完全相同的改动。

**位置 4 — Scope V 索引生成**（v5:356–370 → v5_fix_by_xrb:386–406）：

```diff
 for k_i in T.serial(kv_iter_end):
     kv_start = split_points[b_i, k_i]
+    kv_end = split_points[b_i, k_i + 1]
     seg_start = segment_offsets[b_i, seg_id]
     seg_end = segment_offsets[b_i, seg_id + 1]
     max_row_pos = q_start + q_tile_size - 1

     process_cond = (
         ((rule == 0) & (kv_start <= max_row_pos))
         | ((rule == 1) & (kv_start < seg_end))
-        | ((rule == 2) & ((kv_start <= seg_start) | (kv_start <= max_row_pos)))
+        | (
+            (rule == 2)
+            & (
+                (kv_start < seg_start)
+                | ((kv_start < q_end) & (kv_end > q_start))
+            )
+        )
     )
```

### 3.5 理论收益

对一个 diagonal segment，如果它被切成 `T` 个 Q tile，prefix tile 数为 `P`：

| | tile 0 | tile 1 | tile 2 | ... | tile T-1 |
| --- | --- | --- | --- | --- | --- |
| v5 处理 KV tile 数 | P + 1 | P + 2 | P + 3 | ... | P + T |
| v5_fix_by_xrb 处理 KV tile 数 | P + 1 | P + 1 | P + 1 | ... | P + 1 |

节省 KV tile 数：`0 + 1 + 2 + ... + (T - 1) = T * (T - 1) / 2`

每跳过一个 KV tile，减少：
1. 一次 K 搬运 + QK MMA
2. 一次 `workspace_1` 写/读
3. 一次 mask/softmax 局部处理
4. 一次 `workspace_2` 写/读
5. 一次 V 搬运 + PV MMA
6. 一次 `workspace_3` 写/读

节省会乘上 batch 数和 head 数，并反映在 Cube 和 Vector 两侧。

## 4. 优化点 B：rule 1 的 KV 迭代上界收紧

### 4.1 问题分析

`rule == 1` 的语义是 full attention 到当前 segment 结束位置之前：

```text
允许访问: K < segment_end
```

有效条件：`kv_start < seg_end`。因此最小且正确的 exclusive end 是：**第一个满足 `split_points[k] >= seg_end` 的 `k`**。

基线 v5 的 `next_seg_first_tile` 计算有两个问题：

#### 问题 1：短 full segment 退化到扫描整个 batch

v5 寻找的是"最后一个 `split_points[_k] < seg_end_offset` 的 `_k`"。当 rule 1 segment 很短（如默认样例中的 8 token，远小于 `block_M=128`）时，找不到任何满足条件的 `_k`，于是 `next_seg_first_tile` 保持 `s_local`，触发 fallback 到 `tiles_this_batch`。

结果：一个 8-token 的 full segment 本来只需要扫描 `k = 0, 1`，却退化到扫描 batch 内全部 tile。虽然后续 `process_cond: kv_start < seg_end` 会过滤掉无效 tile（结果正确），但 Scope C 和 Scope V 两侧的标量扫描都多做了大量无用的循环迭代。

#### 问题 2：长 full segment 会漏掉最后一个合法 tile

v5 找的是最后一个 `_k` 满足 `split_points[_k] < seg_end`。但 `range(kv_iter_end)` 是左闭右开，如果 `seg_end = 512` 且 `split_points[3] = 384`（最后一个合法 tile 的起点），则 `next_seg_first_tile = 3`，`range(3)` 只覆盖 `[0,1,2]`，漏掉了 `k=3`（即 `[384, 512)`）。

正确的 `kv_iter_end` 应该是 `4`（第一个 `split_points[k] >= 512` 的索引）。

### 4.2 修改方案

将 `next_seg_first_tile` 重新定义为"当前 segment 后第一个 tile 的索引"——即第一个满足 `split_points[k] >= seg_end_offset` 的边界：

```python
# v5（旧）
next_seg_first_tile = s_local
for _k in T.serial(max_splits):
    next_seg_first_tile = T.if_then_else(
        (_k > s_local) & (split_points[b_i, _k] < seg_end_offset), _k, next_seg_first_tile
    )
tiles_this_batch = tiles_prefix_sum[b_i + 1] - tiles_prefix_sum[b_i]
next_seg_first_tile = T.if_then_else(next_seg_first_tile == s_local, tiles_this_batch, next_seg_first_tile)
kv_iter_end = T.if_then_else(rule == 1, next_seg_first_tile, s_local + 1)

# v5_fix_by_xrb（新）
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
```

改动要点：

| 要点 | v5 | v5_fix_by_xrb |
| --- | --- | --- |
| 初始值 | `s_local`（需要 fallback） | `tiles_this_batch`（天然上界） |
| 查找目标 | 最后一个 `< seg_end_offset` | 第一个 `>= seg_end_offset` |
| 越界保护 | fallback 到 `tiles_this_batch` | `_k <= tiles_this_batch` |
| 取最小 | 无（取最后） | `_k < next_seg_first_tile`（取最小） |

图示：

```text
split_points index:
0    1    2    3    4    5    6
|----|----|----|----|----|----|
          ^ q tile (s_local=1)
                    ^ seg_end_offset
                    ^ next_seg_first_tile = 4

rule 1:
  kv_iter_end = 4
  range(4) → 0, 1, 2, 3  ← 恰好所有合法 tile
```

短 full segment 例子（8 token）：

```text
seg_end = 136
split_points[2] = 136 ≥ seg_end

next_seg_first_tile = 2
range(2) → 0, 1  ← 不再退化到 tiles_this_batch
```

### 4.3 修改位置

这个改动必须应用于 C/V 两侧的 **2 处**：

| # | 位置 | 行号 (v5_fix_by_xrb) | 作用 |
| --- | --- | --- | --- |
| 5 | Scope C `next_seg_first_tile` 计算 | 169–180 | 决定 `valid_k_total` 和 QK/PV 扫描的 `kv_iter_end` |
| 6 | Scope V `next_seg_first_tile` 计算 | 372–383 | 决定 `valid_k_indices` 生成的 `kv_iter_end` |

两侧代码完全一致，保证 C/V 对有效 KV tile 范围的理解对齐。

### 4.4 正确性说明

因为 rule 1 的有效条件是 `kv_start < seg_end`，只需要扫描所有起点小于 `seg_end` 的 KV tile。

`next_seg_first_tile` 是第一个满足 `split_points[k] >= seg_end` 的 tile 边界，因此 `range(next_seg_first_tile)` 恰好覆盖所有可能满足 `kv_start < seg_end` 的 tile——不会遗漏（修复问题 2），也不会多余扫描（修复问题 1）。

如果当前 segment 是 batch 内最后一个 segment，找不到后续边界，`next_seg_first_tile` 保持初始值 `tiles_this_batch`，自然扫描到 batch 末尾。

这个优化不改变 mask 语义——Vector 侧仍然按 `rule == 1` 做 `fill_len = min(seg_end - kv_start, kv_size)` 来 mask。它只是把明显不可能满足条件的 tile 从外层扫描范围里排除。

### 4.5 理论收益

主要减少 `rule == 1` 路径上的标量循环和索引选择开销。不直接减少 GEMM（不像 rule 2 剪枝那样），但缩短了 C/V 两侧进入真正流水计算前的准备路径。

对短 full segment（如默认样例中 8 token），优化幅度最大——原来退化到扫描整个 batch，现在只扫描到 segment 边界。

## 5. 完整修改映射表

```text
sparse_fa_v5.py  →  sparse_fa_v5_fix_by_xrb.py
```

| # | 行号 (v5_fix_by_xrb) | Scope | 位置 | 改动摘要 | 优化点 |
| --- | --- | --- | --- | --- | --- |
| 1 | 169–180 | C | `next_seg_first_tile` | 初始值改为 `tiles_this_batch`；查找第一个 `>= seg_end_offset` 而非最后一个 `< seg_end_offset` | **B** |
| 2 | 187–202 | C | `valid_k_total` 计数 | 新增 `kv_end_k`；rule 2 条件从 `max_row_pos` 近似改为区间重叠 | **A** |
| 3 | 222–237 | C | QK 扫描 `process_cond_scan` | 新增 `kv_scan_end`；rule 2 条件同步替换 | **A** |
| 4 | 277–292 | C | PV 扫描 `process_cond_scan` | 新增 `kv_scan_end`；rule 2 条件同步替换 | **A** |
| 5 | 372–383 | V | `next_seg_first_tile` | 与 #1 完全一致，保证 C/V 对齐 | **B** |
| 6 | 386–406 | V | `valid_k_indices` 生成 | 新增 `kv_end`；rule 2 条件同步替换 | **A** |

**核心约束**：

```text
C 侧计数 (#2) == C 侧 QK 扫描 (#3) == C 侧 PV 扫描 (#4) == V 侧索引生成 (#6)
C 侧 kv_iter_end (#1) == V 侧 kv_iter_end (#5)
```

任何一个位置漏改，都会产生 workspace slot 语义不一致——Cube 认为第 `i` 个有效 KV tile 是 A，Vector 认为第 `i` 个有效 KV tile 是 B，导致 mask 和 score 错位。

### 未修改的部分

Host 端 wrapper（`high_perf_sparse_attn_wrapper`）、`golden_attention`、`test` 函数以及 `__main__` 测试入口均与 v5 完全一致。调度参数 `num_stages=14`、`cross_interval=2`、`block_M=128`、`block_N=128` 均保持不变。

## 6. 验证

### 6.1 语法检查

```bash
python -m py_compile sparse-fa/sparse_fa_v5_fix_by_xrb.py
```

### 6.2 默认大样例精度

```bash
source tilelang-ascend/set_env.sh
cd sparse-fa
python sparse_fa_v5_fix_by_xrb.py
```

结果：

```text
Test Passed!
```

### 6.3 rule 1 跨 tile 回归

为覆盖 full segment 长度超过 128 的场景（验证优化点 B 不遗漏最后一个合法 tile）：

```bash
source tilelang-ascend/set_env.sh
cd sparse-fa
python - <<'PY'
import sparse_fa_v5_fix_by_xrb as m
m.test(
    H=8,
    D=128,
    seg_lengths=[[256]],
    rules=[1],
    matched_prefix_arr=[0],
)
PY
```

结果：

```text
Test Passed!
```

### 6.4 性能 profile

```bash
source tilelang-ascend/set_env.sh
cd sparse-fa
msprof op --kernel-name=main_kernel --output=./OPPROF_sparse_fa_v5_fix_by_xrb_final python sparse_fa_v5_fix_by_xrb.py
```

核心结果：

```text
main_kernel,mix,3966.64,24,48,0,545565,1850,1850,
```

同环境 v5 基线：

```text
main_kernel,mix,4650.55,24,48,0,392929,1850,1850,
```

相对提升：**14.71%**，约 **1.17x**。

