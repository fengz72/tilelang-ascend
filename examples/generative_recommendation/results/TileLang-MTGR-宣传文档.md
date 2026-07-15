# TileLang × 生成式推荐：Python DSL 驱动美团 MTGR 稀疏注意力算子昇腾适配

## 背景介绍

生成式推荐（Generative Recommendation）是推荐系统领域的新范式，它将推荐问题建模为序列生成任务，通过大模型对用户行为序列进行自回归建模，生成个性化的推荐结果。美团提出的 MTGR（Meituan Generative Recommendation）框架是该方向的代表性创新，其核心在于对用户行为序列进行多段式建模——将序列划分为历史行为（history）、上下文（context）、实时行为（realtime）、候选目标（target）等语义段，每段采用不同的注意力可见性规则，以精准刻画推荐场景中"历史因果、上下文全可见、目标独立评估"的业务语义。

这一多段异构掩码机制带来了一个关键的计算挑战：**标准注意力算子无法直接支持**。在昇腾 NPU 上，CANN 原生的 `FusedInferAttentionScore`（FIA）仅支持单一的 Causal 或 Full 掩码，面对 MTGR 的多段混合掩码需求，开发者不得不先调用 `Mask` 算子生成完整的掩码矩阵，再将其传入 FIA 进行计算。这种"两算子串联"方案存在两大问题：

1. **额外访存开销**：掩码矩阵需要物化到全局内存，引入大量中间数据读写
2. **丧失融合优化机会**：掩码生成与注意力计算分离执行，无法利用算子融合消除中间结果落盘

与此同时，若采用 Ascend C 原生开发定制算子，开发者需要直面 Cube/Vector 双核协作、L0C/L1/UB 多级内存管理、跨核信号量同步等底层硬件细节，开发门槛高、调优周期长。

为此，我们基于 TileLang-Ascend 路线重写了 MTGR 的核心注意力算子 `mtgr_ragged_segment_attention`，以 Python DSL 表达多段掩码逻辑，由编译器自动处理底层硬件映射，在保持开发效率的同时，实现了相对 CANN 两算子串联基线 **最高 2.48x 的 Kernel 加速**。

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     生成式推荐推理引擎                           │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    推理调度层                              │  │
│  │   多段序列构建 → 段偏移计算 → 算子调用 → 结果回收            │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │ 运行时调度                            │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │              TileLang 编译产物 (.so)                       │  │
│  │   mtgr_ragged_segment_attention (融合多段掩码注意力)        │  │
│  └───────────────────────▲───────────────────────────────────┘  │
│                          │ AOT 编译                              │
│  ┌───────────────────────┴───────────────────────────────────┐  │
│  │              TileLang Python Kernel                       │  │
│  │   @T.prim_func → generate_source() → Ascend C 源码         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

TileLang 的编译流程将开发者用 Python 表达的 Tiling 逻辑与计算流水线，自动降级为 Ascend C 源码，再由毕昇编译器编译为动态库。开发者全程无需手写 C++ Kernel 代码，也无需手动计算 L1/UB/L0C 的内存偏移量——这些繁琐工作由编译器自动完成。

## 核心技术挑战与 TileLang 解决方案

MTGR 多段掩码注意力的实现面临五大技术挑战。以下展示 TileLang 如何以 Python DSL 逐一化解。

### 挑战一：多段异构掩码的统一表达

MTGR 将序列划分为多段，每段采用不同掩码规则：

| 规则 | 语义 | 可见范围 | 业务含义 |
|------|------|----------|----------|
| Causal | 下三角因果 | `q_pos + 1` | 历史行为序列，仅可见过去 |
| Full | 全可见 | `seg_end` | 上下文/属性信息，全局共享 |
| Diagonal | 对角线 | `seg_start` + 自身 | 候选目标，独立评估 |

在 TileLang 中，开发者可以用标准的 Python 控制流直接表达这一异构逻辑，无需关心底层向量化指令的生成：

```python
# 逐行施加段掩码（TileLang 自动映射为高效的向量化填充指令）
for row in T.serial(half_M):
    row_abs_pos = q_start + vid * half_M + row
    _row_rule = row_rule_buf[row]

    if _row_rule == 0:          # Causal：下三角
        fill_len = row_abs_pos - kv_start + 1
        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
    elif _row_rule == 1:        # Full：全可见
        fill_len = _row_seg_end - kv_start
        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
    elif _row_rule == 2:        # Diagonal：对角线
        fill_len = _row_seg_start - kv_start
        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
        buf_2d[row, diag_col] = 0.0            # 对角线自身可见
```

**核心优势**：掩码生成与注意力计算融合在同一个 Kernel 内，掩码直接在片上 UB 中生成并施加到 Score 矩阵，**彻底消除了 CANN 两算子方案中掩码矩阵的物化与全局内存读写**。这是性能提升的根本来源。

**二分查找定位段规则**：MTGR 场景下段数可达上千（实测最多 1027 段），若对每行线性扫描定位所属段，复杂度为 O(max_segs)，在多段场景下性能急剧下降。算子采用**二分查找**（`bin_iters = max_segs.bit_length()`），将复杂度降至 O(log(max_segs))，确保段数增长时性能不退化：

```python
# 二分查找：定位每行所属段，O(log N) 复杂度
_lo = 0
_hi = max_segs
for _bi in T.serial(bin_iters):          # bin_iters = log2(max_segs)
    _mid = (_lo + _hi) // 2
    _le = segment_offsets[b_i, _mid] <= row_abs_pos
    _gt = segment_offsets[b_i, _mid] > row_abs_pos
    _lo = T.if_then_else(_active & _le, _mid + 1, _lo)
    _hi = T.if_then_else(_active & _gt, _mid, _hi)
seg_id = _lo - 1

# 查表获取该行的掩码规则与段边界
row_rule_buf[row] = segment_rules[seg_id]
row_seg_start_buf[row] = segment_offsets[b_i, seg_id]
row_seg_end_buf[row] = segment_offsets[b_i, seg_id + 1]
```

在 1027 段的极端场景下，二分查找仅需 11 次迭代即可定位，相比线性扫描的 1027 次比较，**段定位开销降低近 100 倍**，是算子在超多子段场景下仍保持 2.42x 加速的关键。

### 挑战二：Cube/Vector 双核协作流水线

注意力计算包含矩阵乘（QK^T、PV）和逐元素运算（Mask、Softmax），前者适合 Cube 核，后者适合 Vector 核。在 Ascend C 原生开发中，开发者需要手动管理双核的内存排布转换（如 Fractal_NZ 格式）、跨核信号量同步、workspace 分配等底层细节。

TileLang 提供了 `T.Scope("C")` / `T.Scope("V")` 抽象，让开发者以声明式方式调度计算：

```python
with T.Scope("C"):
    # Cube 核：矩阵乘法
    T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
    T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])    # C→V 传输
    T.set_cross_flag("FIX", SEM_WS1_C2V)                    # 通知 Vector

with T.Scope("V"):
    # Vector 核：掩码 + Softmax
    T.wait_cross_flag(SEM_WS1_C2V)                          # 等待 Cube
    T.copy(workspace_1[cid, i, ...], io_buf)
    T.tile.add(work_ub, work_ub, buf_2d)                    # 施加掩码
    T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)        # 在线 Softmax
    T.tile.exp(work_ub, buf_2d)
    T.copy(work_ub, workspace_2[cid, i, ...])               # V→C 传输
    T.set_cross_flag("MTE3", SEM_WS2_V2C)                   # 通知 Cube
```

**核心优势**：开发者只需关注"什么计算在哪个核上执行"以及"核间数据依赖关系"，TileLang 编译器自动处理 Fractal 格式转换、L0A/L0B/L0C 双缓冲分配、跨核信号量编排。算子采用 14 级 KV tile 流水深度（`num_stages=14`），配合 `cross_interval=2` 的跨核同步聚合，实现 Cube 与 Vector 的高效重叠。

**框架接管 VS 同步，释放流水线性能**：在 Expert 模式下，算子关闭了毕昇编译器的自动同步（`--cce-auto-sync=off`），改为由 TileLang 框架精确插入同步点。其中关键的一步是保留 `TL_ASCEND_AUTO_SYNC_VS: True`——将 Vector 核内部的 Scalar 与 Vector 流水线同步交由框架自动管理，而非毕昇编译器的保守策略：

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,   # 手动分离 Cube/Vector
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC:    False,   # 手动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC:       False,   # 手动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,   # 手动地址分配
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS:    True,    # 框架接管 VS 同步
}
```

毕昇编译器在缺乏全局流水线信息时，倾向于插入保守的 VS 屏障以确保正确性，这会打断 Vector 核内 Scalar 计算与 Vector 计算的重叠。TileLang 框架拥有完整的流水线视图，能够精确判断哪些 VS 依赖可以延迟、哪些需要立即同步，从而**最大化 Scalar 与 Vector 的执行重叠**。这一优化带来了巨量的性能收益，是算子达到 1.80x 平均加速的关键因素之一。

### 挑战三：计算掩盖（Hiding Computation）优化

在双核流水线中，Vector 核在等待 Cube 核的 GEMM 结果期间存在算力空闲。TileLang 的 Python DSL 让开发者能够灵活插入"掩盖计算"，利用空闲算力提前生成后续 tile 的掩码：

```python
# 【核心优化】在等待 Cube 结果前，利用空闲算力提前生成 MASK
_vid_first_rule = row_rule_buf[0]
_first_row_pos = q_start + vid * half_M
_is_full_kv = kv_size == block_M

# 基于可见性单调递增原理：首行完全可见，则全块必然完全可见
_r0_valid = (_vid_first_rule == 0) & (_first_row_pos + 1 >= kv_start + kv_size)
_r1_valid = (_vid_first_rule == 1) & (kv_start + kv_size <= _vid_first_seg_end)
_r2_valid = (_vid_first_rule == 2) & (kv_start + kv_size <= _vid_first_seg_start)

_all_valid = T.if_then_else(_is_full_kv & _r0_valid, 1, ...)

if _all_valid == 1:
    T.tile.fill(buf_2d, 0.0)          # 快速路径：整块全可见，跳过逐行掩码
else:
    T.tile.fill(buf_2d, NEG_INF)      # 慢速路径：逐行生成掩码
    for row in T.serial(half_M):
        ...
```

**核心优势**：这一优化基于"可见性单调递增"原理——若首行完全可见，则整块必然完全可见，可直接跳过逐行掩码生成。**快速路径不仅跳过了逐行的 `T.tile.fill` 指令，更重要的是避免了每行的 Scalar 计算**（二分查找段定位、规则查表、`fill_len` 计算、对角列判断等），将整个 half_M×block_N 的掩码生成缩减为一次 `T.tile.fill(buf_2d, 0.0)`。在推荐场景中，history 段和 context 段的大量 tile 命中快速路径，显著减少了 Vector 核的 Scalar 开销。这类需要结合业务语义的优化，在 TileLang 的 Python DSL 中表达自然直观，而在 Ascend C 中实现则极为繁琐。

### 挑战四：Paged KV Cache 与 Ragged Batch 的灵活数据流

生成式推荐场景下，不同用户的序列长度各异（ragged batch），且需要复用前缀缓存（Paged KV Cache）以加速推理。这要求算子支持：

- **Ragged Batch**：各 request 长度不等，TND packed 布局
- **Paged KV Cache**：通过 `block_table` 查询物理块，前缀缓存命中与 live KV 混合访问
- **动态段结构**：段数从 4 到 1027 不等，段长度从 4 到 3200 token 跨度极大

TileLang 的 Python DSL 让这些复杂的数据流逻辑以直观的方式表达：

```python
# Ragged batch：通过 q_seq_starts 定位 request，线性扫描解析任务
for _b in T.serial(batch):
    _total_seq_len = segment_offsets[_b, max_segs]
    num_tiles_b = T.ceildiv(_total_seq_len, block_M)
    _is_this = (tile_id >= cum_tiles) & (tile_id < cum_tiles + num_tiles_b)
    b_i = T.if_then_else(_is_this, _b, b_i)
    s_local = T.if_then_else(_is_this, tile_id - cum_tiles, s_local)

# Paged KV Cache：前缀部分走 cache，live 部分走原始 KV
if kv_start < prefix_len_b:
    cache_block_idx = kv_start // block_N
    physical_block = block_table[b_i, cache_block_idx]
    T.copy(key_cache[physical_block, ...], k_l1[:, :])
else:
    kv_packed_start = q_seq_starts[b_i] + kv_start - prefix_len_b
    T.copy(K[kv_packed_start:..., h_kv, :], k_l1[:, :])
```

**核心优势**：Paged KV Cache 的块表查询、前缀/live 混合访问、ragged batch 的任务解析，全部在 Python 层以标准控制流表达，编译器自动映射为高效的 GM 访问指令。开发者无需关心物理块地址计算、非连续访存的向量化处理等底层细节。

### 挑战五：单一 Kernel 适配多 Head Dim 的地址规划

生成式推荐模型在不同层和不同配置下可能使用不同的 Head Dim（32/64/128），传统开发方式需要为每个 Dim 维护独立的内存布局代码。TileLang 通过 `T.annotate_address` 的**符号化地址表达式**，让一份 Kernel 源码同时适配三种 Dim：

```python
# 地址以 dim、half_M、block_N 等符号参数表达，编译后自动适配不同 Dim
T.annotate_address({
    acc_o: 0,
    r_factors: half_M * dim * 4,                                    # 随 dim 缩放
    sumexp_is: half_M * dim * 4 + num_stages * half_M * 4,
    io_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12,
    acc_s_half: half_M * dim * 4 + ... + half_M * block_N * 2,     # bf16 缓冲
    work_ub: half_M * dim * 4 + ... + half_M * block_N * 4,        # fp32 缓冲
    buf_2d: half_M * dim * 4 + ... + half_M * block_N * 8,         # 掩码缓冲
    ...
})
```

**核心优势**：所有 UB buffer 的地址均以 `dim`、`half_M`、`block_N`、`num_stages` 等符号参数的算术表达式定义，而非硬编码常量。这意味着：

1. **一份源码，三种 Dim**：D=32/64/128 共用同一份 Kernel 实现，地址布局在编译时根据 Dim 自动推导，无需为每个 Dim 手写独立的内存管理代码
2. **精确的地址复用**：生命周期不重叠的 buffer（如 `io_buf` 与 `o_io_buf`、`acc_s_half` 与 `o_acc_half`）被显式分配到相同地址，最大化 UB 空间利用率，使算子在 192KB UB 限制下仍能容纳 14 级流水所需的全部缓冲区
3. **可维护性**：新增或调整 buffer 时，只需修改符号表达式，编译器自动验证地址不冲突，避免了手动计算 offset 的灾难性维护负担

这一能力使得算子在覆盖 D=32/64/128 三种规格的 462 组精度测试和 63 组性能测试中，均以同一份代码实现，充分体现了 TileLang DSL 的表达力与可维护性。

## 性能收益

### 测试方法

- **对比基线**：CANN 原生 `FusedInferAttentionScore` + `Mask` 两算子串联（因 CANN 不支持多段掩码，需先用 Mask 算子生成掩码矩阵，再传入 FIA）
- **测试工具**：`msprof op` 采集 Kernel 级耗时
- **测试覆盖**：63 组对比用例，覆盖 Head Dim 32/64/128、Batch 1-8、段数 4-1027

### 核心性能数据

| 指标 | 结果 |
|------|------|
| 对比用例数 | 63 |
| **Kernel 平均加速比** | **1.80x** |
| Kernel 加速范围 | 1.25x ~ **2.48x** |
| Kernel 加速 ≥ 1.5x 的用例占比 | **88.9%（56/63）** |
| Kernel 加速 ≥ 2.0x 的用例占比 | 25.4%（16/63） |

### 按 Head Dim 分组

| Head Dim | 用例数 | 平均 Kernel 加速比 |
|----------|--------|-------------------|
| D=128 | 33 | **1.91x** |
| D=64 | 15 | **1.79x** |
| D=32 | 15 | **1.57x** |

> Head Dim 越大，GEMM 计算密度越高，融合算子消除中间访存的优势越明显。

### 代表性场景性能

| 场景 | 配置 | TileLang Kernel | CANN 基线 Kernel | 加速比 |
|------|------|-----------------|------------------|--------|
| 单用户基础场景 | B=1, D=128, 4段 | 228.86 us | 378.89 us | 1.66x |
| 多用户并发 | B=8, D=128, 4段 | 2687.67 us | 5934.73 us | **2.21x** |
| 超多子段复杂场景 | B=1, D=128, 1027段 | 1420.08 us | 3438.23 us | **2.42x** |
| 多用户复杂段结构 | B=8, D=128, 7段 | 2577.04 us | 6392.94 us | **2.48x** |

### 精度验证

| 指标 | 结果 |
|------|------|
| 精度用例数 | 462 |
| 累计运行次数 | 11,480 |
| 通过率 | **100%（0 失败）** |
| 精度标准 | atol=1e-2, rtol=1e-2（bfloat16） |
| 覆盖范围 | D32/64/128、B1-8、段数 4-1027、三种掩码规则、前缀匹配三模式 |

## 总结

TileLang 为 MTGR 多段掩码注意力算子在昇腾 NPU 上的实现提供了一条高效路线：

- **性能高**：通过算子融合消除掩码矩阵物化与中间访存，Kernel 相比 CANN 两算子串联基线平均加速 **1.80x**，最高 **2.48x**，88.9% 用例达到 1.5x 以上加速。框架接管 VS 同步、二分查找段定位、可见性单调递增快速路径等多项优化共同贡献了性能收益。
- **开发便利**：多段异构掩码、Cube/Vector 双核协作、Paged KV Cache、Ragged Batch 等复杂逻辑，全部以 Python DSL 直观表达，编译器自动处理 Fractal 格式转换、双缓冲分配、跨核信号量同步等底层细节。通过 `T.annotate_address` 的符号化地址表达式，一份 Kernel 源码即可适配 D=32/64/128 三种 Head Dim，开发者专注于算法逻辑与业务语义优化。
- **精度可靠**：462 组用例、11,480 次运行全部通过，覆盖极端 ragged、超多子段、非对齐边界等场景，满足 bfloat16 精度要求。
- **可复用**：多段掩码注意力的 TileLang 实现模式（融合掩码生成 + 双核流水线 + 计算掩盖优化 + 符号化地址规划）可推广至其他需要异构掩码的注意力变体，为生成式推荐及更多创新模型架构的快速落地奠定基础。
