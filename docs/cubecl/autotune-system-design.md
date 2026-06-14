# CubeCL 的 Autotune 系统：策略枚举、优先级剪枝与量化缓存

## 为什么需要 Autotune

矩阵乘法 `C = A × B`。A 的形状是 [1, 4096]，B 是 [4096, 4096]——这是一次 matvec。同样的乘法，A 是 [4096, 4096]，B 是 [4096, 4096]——这是一次 gemm。同一个操作、同一份代码、同一块 GPU，最优的 tile size、workgroup 大小、向量化宽度完全不同。

更复杂的场景：gemm + bias + relu 被 fusion 合并成一个 kernel 后，最优参数又变了——因为寄存器压力不同，memory bound vs compute bound 的平衡点也不同。

**不存在一套参数在所有场景下最优**。Autotune 就是在运行时为具体的 (shape, dtype, hardware, fusion_combination) 搜索最佳 kernel 实现。

但 autotune 本身有代价：每个候选方案都需要在 GPU 上真正跑几次来测量性能。如果候选太多，首次执行延迟不可接受——用户不会等 30 秒来做 kernel tuning。

CubeCL 的 autotune 设计核心是在**搜索精度**和**搜索代价**之间找平衡。

---

## 设计哲学：策略枚举 vs 参数网格

这是 CubeCL 和 Triton autotuner 最根本的差异。

### Triton 的路：参数网格

Triton 的 `@triton.autotune` 让用户定义一个参数网格：`BLOCK_SIZE_M = [16, 32, 64, 128]`、`BLOCK_SIZE_N = [32, 64, 128]`、`num_warps = [2, 4, 8]`……编译器对每个组合生成 kernel、做 exhaustive search。简单、参数可正交、覆盖面广，但候选数爆炸——5×4×3×3 就是 180 个候选。

### CubeCL 的路：策略枚举

CubeCL 走的是完全不同的路线。Kernel 作者**枚举一组具体的实现策略**，每个策略是一个闭包，包含自己的 tile 大小、向量化策略、双缓冲配置等所有参数：

```rust
// 示意：不是实际代码，但反映结构
TunableSet::new(key_gen, input_gen)
    .with(Tunable::new("simple_unit",    |i| simple_unit_matmul(i)))
    .with(Tunable::new("double_unit_max", |i| double_unit_max_tile(i)))
    .with(Tunable::new("gemm_no_stage",  |i| gemm_no_stage(i)))
    .with(Tunable::new("cmma_multi_rows",|i| cmma_multi_rows(i)))
    .with(Tunable::new("tma_cmma",       |i| tma_cmma(i)))
    // ... 20+ more
```

这不是"调参数"——是**从一组已知可行的实现中选最快的那个**。每个候选都是手写的、验证过的 kernel，不是一个参数组合自动生成出来的。

这个设计选择的后果：

**优点**：候选数少（6-35 个），质量可控（每个候选是手写优化过的），不会出现参数组合生成的 kernel 无法运行的情况。

**代价**：覆盖面由 kernel 作者决定。如果作者没有为某种 shape 注册合适的策略，autotune 也拯救不了。CubeCL 通过**优先级分组（TuneGroup）**部分补偿——同一策略可以在不同硬件/shape 下被不同的优先级控制跳过或优先。

**`burn/crates/burn-cubecl/src/kernel/matmul/tune/base.rs`** 中 matmul 注册了 30+ 个候选：

- Naive fallback（始终在第一位，保证了至少有一个可执行的 kernel）
- GEMV 变体（matvec 场景）：DoubleVecMat、SimpleVecMat、Gemm、GemvUnitPerpendicular
- Unit 变体（小矩阵）：SimpleUnit、DoubleUnit，各自带 Max/Min tile size
- Gemm no-stage
- 加速变体：Cmma vs Mma（硬件矩阵乘法指令），multi_rows 开关，双缓冲配置
- TMA 变体（Tensor Memory Accelerator，H100 特性）：SimpleTma、SpecializedTma

用枚举而非网格的深层原因是：**GPU kernel 优化参数不是正交的**。double buffering 只在配合分块 gemm 时有效，TMA 只在 H100 上有效，向量化宽度受 shared memory 大小限制。正交网格会产生大量无效组合；枚举只包含验证过的组合，零无效候选。

---

## 缓存密钥设计：Anchor 量化

如果每种 shape 都单独 autotune，缓存永远无法命中——shape 的取值空间是连续的。CubeCL 用 **anchor 量化** 将 shape 空间映射到离散桶：

```rust
// cubecl/crates/cubecl-runtime/src/tune/util.rs:16
pub fn anchor(value: usize, base: f64) -> usize
```

anchor 函数将值**向上取整 (ceil)** 到最近的 `base^n`（不是除以，是舍入）。

| AutotuneLevel | anchor 因子 | 效果 |
|---------------|------------|------|
| Minimal | 1.25× | 粗桶——更少缓存条目，更快命中 |
| Balanced | 1.0× | 默认 |
| Extensive | 0.75× | 细桶——更多条目，更精确 |
| Full | 无 anchor | 每种 shape 独立条目 |

例如 shape [1024, 2048] 在 Balanced 级别可能和 [1024, 2050] 映射到同一个桶——因为 2050 落在 2048 的 anchor 范围内。在 Full 级别则它们是不同的桶。`#[derive(AutotuneKey)]` 宏对标记了 `#[autotune(anchor)]` 的字段自动应用此量化。

缓存密钥中还包括精度、layout、fusion 的组合信息。以融合 matmul 为例：

```rust
// burn/crates/burn-cubecl-fusion/src/optim/matmul/tune.rs:21
pub struct FusedMatmulAutotuneKey {
    matmul_key: MatmulAutotuneKey,  // shape、dtype、layout
    num_out_buffers: u32,      // #[autotune(anchor)]
    num_ops: u32,              // #[autotune(anchor)]
}
```

这意味着 matmul + 3 个 fused op 和 matmul + 4 个 fused op 在 Balanced 级别可能共享缓存（锚定到同一桶），在 Extensive 级别则分开。

### 缓存架构：内存 + 持久化 + checksum

```
TuneCache<K>
  ├── in_memory: HashMap<K, CacheEntry>
  │     ├── Done { checksum: Match, fastest_index: usize }
  │     └── Pending   (其他线程正在对该 key 做 autotune 中)
  └── persistent: Cache<(key, checksum), (fastest_index, results)>
        └── {cache_root}/autotune/{cargo_version}/{device_id}/{tuner_name}.json.log
        （`cubecl/crates/cubecl-common/src/cache.rs:303`，cache_root 由 `AutotuneConfig.cache` 配置为 Local/Target/Global）
```

checksum 是所有 tunable 名称拼接后的 MD5。如果 kernel 作者增加或删除了一个候选策略，checksum 不匹配，全部持久化缓存自动失效——保证不会使用过时的"最快第 3 个"索引。

持久化缓存独立于进程生命周期。首次运行冷启动需要 autotune（6-35 个候选的 benchmark），后续运行直接命中磁盘缓存。

**缓存粒度**：
- 每个 `static TUNER` 实例独立命名空间（matmul 和 reduce 不共享缓存）
- 每个设备独立（CPU、GPU#0、GPU#1 各有自己的缓存，因为硬件不同最优参数不同）
- 每个 (key + checksum) 一个条目

---

## 搜索策略：优先级分组 + 批量提前终止

如果对 30 个候选逐一 benchmark，每个 warmup 3 次 + 测量 10 次 = 13 次 kernel launch × 30 = 390 次 kernel launch。不可接受。

CubeCL 的方案是 **优先级分组（TuneGroup）+ 批量提前终止**。

### 优先级分组

Kernel 作者为不同的硬件特性或 shape 特征分配优先级（`cubecl/crates/cubecl-runtime/src/tune/base.rs:62`）：

```rust
// 示意
let accelerated_group = TuneGroup::new("accelerated", |key: &MatmulKey| -> i8 {
    match key.kind {
        MatmulKind::Large | MatmulKind::General => 3,  // 优先尝试
        _                                        => 1,  // 低优先级
    }
});
```

- **高优先级组先执行**：rank 3 的候选在所有 rank 1 的候选之前 benchmark
- **组内子优先级**：同一个组内，每个候选可以有自己的子优先级
- **负数跳过**：优先级为 -1 的候选直接跳过（比如 TMA 在不支持的硬件上）
- **提前终止**：只要某一批中至少有一个候选成功运行，后续所有分组不再执行

### 具体例子：Matmul 的优先级层级

| 组 | 适用场景 | 优先级 |
|----|---------|--------|
| accelerated | Large/General matmul | 3（最先试） |
| tma | Large matmul + 良好 stride（H100） | 3 或 2 |
| unit | 小矩阵 | 2 |
| gemv | MatVec/VecMat | 3 或 2 |
| （无组） | naive fallback | 始终最后 |

对于 Large matmul：accelerated 组先跑，找到最快的立即停止，naive 根本不跑。对于 MatVec：gemv 组先跑，accelerated 组（优先级=1）排在后面——如果 gemv 成功了提前终止，accelerated 也不跑。

### 批量 benchmark

每次 `plan.next()` 返回**一批**候选（`cubecl/crates/cubecl-runtime/src/tune/base.rs:186`）。批内所有候选都被 benchmark：3 次 warmup + 10 次采样。

测量使用 **GPU 时间戳**（`cubecl/crates/cubecl-wgpu/src/compute/timings.rs`）而非 CPU 墙钟时间——避免了 CPU-GPU 同步开销和系统抖动的影响。Metal 上受限于最多 28 个活跃时间戳查询集，用 round-robin 溢出。

### 评分函数

```rust
// cubecl/crates/cubecl-common/src/benchmark.rs:124
fn score(&self) -> u64 {
    const ALPHA: f64 = 0.8;
    let base = min * ALPHA + median * (1.0 - ALPHA);
    let cv = 1.0 + sqrt(variance) / (1.0 + mean);
    (base * cv) as u64
}
```

**不是简单的取最大值或中位数**。它混合了最小值（80%权重）和中位数（20%权重），再乘以变异系数惩罚项。不稳定的 kernel（方差大）即使最快也会被惩罚——在 GPU 计算中，方差大通常意味着容易触发 throttle 或 cache miss，不是一个可靠的选择。

---

## Fusion 场景下的 Autotune

融合 kernel 的 autotune 有一个额外挑战：benchmark 时需要一个可执行的 context（tensor handles、shapes、dtypes），而这些在 autotune 阶段可能还没完全确定。

解决方案在 `burn/crates/burn-cubecl-fusion/src/tune.rs`：

```rust
pub struct TuneInput<'a, R, O> {
    context: &'a mut Context<CubeFusionHandle<R>>,
    optimization: O,
}
```

`TuneInput` 包装了对 fusion context 的借用引用和优化对象。benchmark 时创建一个 **Fork context**——独立的 handle 空间，不影响原始 context。Fork 中的 benchmark 运行产物通过 `HandleCollector` 收集，在 `Drop` 时仅将原始 context 中缺失的新 handle 提升回去。

关键设计：**autotune 后的最优索引被缓存，但实际的 kernel 执行仍然走正常的 fusion pipeline。** 这意味着 autotune 只是"选中一个策略"，不改变 fusion 的执行路径——只是改变了"选中哪个"。

`FusedMatmulAutotuneKey` 中 `num_out_buffers` 和 `num_ops` 的 anchor 量化意味着：相似的融合模式（比如都是 matmul + 2-3 个 elemwise op）映射到同一个缓存条目——这是合理的，因为融合的 elemwise op 对 register pressure 和 memory 的影响在大体相同的数量级。

---

## 容错与 Fallback

### Fallback 契约：第一个候选永远可用

```rust
// burn/crates/burn-cubecl/src/kernel/matmul/tune/base.rs:164
// First entry should always work, since it is considered the fallback.
```

这不是运行时检查——是**代码约定**。kernel 作者保证 `.with()` 的第一个 tunable 在任何 shape/dtype 下都正确但不一定最快。对于 matmul，这是一个简单的无优化分块乘法；对于 reduce，是 `sum_chained`。

### 配置级 Fallback

`ConvStrategy` 在 `cfg(feature = "autotune")` 为假时，`default()` 返回 `Direct`（直接执行），跳过整个 autotune 管线。这使得 autotune 可以作为编译期 feature flag 关闭，用于不需要 warmup 的场景。

### Fusion 的 fallback 执行

融合 matmul 的 autotune 如果在实际执行时失败（benchmark 阶段测量通过但实际运行时的形状或资源不同），`tune_fused` 函数在原始 context 上调用 `opt.execute_fallback(ctx)`，回退到融合的默认执行路径，保证正确性。

### 显式正确性检查

`autotune-checks` feature 开启后，autotune 会**运行所有候选**（不仅仅是选中的那批）并验证输出是否一致。用于开发阶段确保新添加的候选策略确实计算正确，生产环境关闭。

---

## 与 Triton Autotuner 对比

| 维度 | CubeCL | Triton |
|------|--------|--------|
| 搜索空间定义 | 枚举手写策略（闭包） | 参数网格（Block size、num_warps 等） |
| 候选数 | 6--35 | 数十到数百（网格爆炸） |
| 搜索算法 | 优先级分组 + 批量提前终止 | exhaustive 或启发式/遗传 |
| 首次延迟 | 低（候选少 + 早停） | 可能很高（全网格搜索） |
| 覆盖面 | 由 kernel 作者决定 | 由网格大小决定 |
| 无效组合 | 零（手写验证） | 可能存在（编译失败或 OOM） |
| 缓存 key | anchor 量化的 shape/type/fusion combo | 输入 shape 和 dtype |
| key 空间控制 | `#[autotune(anchor)]` + AutotuneLevel 四级 | 无（形状直接作为 key） |
| 缓存失效 | checksum（tunable 名称变化 → 自动失效） | 手动清理或版本变更 |
| 评分 | min×0.8 + median×0.2，变异系数惩罚 | 中位数或均值 |
| 测量 | GPU 时间戳（3 warmup + 10 sample） | CUDA events（可配置 warmup/reps） |
| 硬件支持 | CUDA + Metal + Vulkan + WebGPU + WASM | 仅 CUDA（ROCm 实验性） |
| Fusion 感知 | 原生：共享 context，fork handle 空间 | 无内置 fusion autotune |
| 正确性验证 | `autotune-checks` feature：运行所有候选并比对 | 无内置 |
| 策略表达力 | 低（需手写每个候选 kernel） | 高（改参数即可生成） |

最根本的哲学差异：**CubeCL 信任 kernel 作者能枚举高质量策略**——用人力换搜索范围缩小。**Triton 信任编译器能对任何参数生成正确代码**——用搜索换人力投入减少。

---

## 限制

1. **覆盖面由作者决定**：如果 kernel 作者没有为某种 shape/硬件注册合适的策略，autotune 选出来的也是次优的。"选最快的"的上限是"有人提供了对的候选"。

2. **首次执行代价**：即使有优先级剪枝，warmup 3 + sample 10 的测量仍然需要 13 次 kernel launch 乘上候选数。对延迟敏感的场景（如首次推理请求），这个开销可能不可接受。缓存命中后无额外测量开销，但冷启动的代价是实实在在的。

3. **参数正交性假设的局限性**：策略枚举假设 kernel 参数不是正交的（否则应该用网格搜索）。对于某些简单的 kernel（如 element-wise），参数确实几乎正交——此时枚举不是最优方案。

4. **单候选场景**：如果只有一个候选，benchmark 被跳过——这是对的，但也意味着如果那个候选不是最优的，autotune 无法改善。

5. **weight quantization**：anchor 量化减少了缓存条目，但 ground truth 是量化掉的那部分 shape 差异——精确 shape 的最优策略和锚定 shape 的最优策略可能不同。

---

## 关键源码入口

- Autotune 框架：`cubecl/crates/cubecl-runtime/src/tune/`
- 优先级与候选：`cubecl/crates/cubecl-runtime/src/tune/base.rs`
- Benchmark 测量：`cubecl/crates/cubecl-runtime/src/tune/tune_benchmark.rs`
- 缓存机制：`cubecl/crates/cubecl-runtime/src/tune/tune_cache.rs`
- Anchor 量化：`cubecl/crates/cubecl-runtime/src/tune/util.rs`
- GPU 时间戳（Metal）：`cubecl/crates/cubecl-wgpu/src/compute/timings.rs`
- Matmul 30+ 候选策略：`burn/crates/burn-cubecl/src/kernel/matmul/tune/base.rs`
- Fusion 场景 autotune：`burn/crates/burn-cubecl-fusion/src/tune.rs`
