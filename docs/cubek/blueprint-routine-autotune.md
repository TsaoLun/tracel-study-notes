# CubeK 的架构纪律：Blueprint、Routine 与 Autotuner

> CubeK 不是在 CubeCL 上再写一遍算子——它用 Blueprint-Routine-Autotuner 三层纪律解决"如何在不产生 kernel 组合爆炸的前提下，让一个 matmul kernel 适配所有硬件"。与 CUTLASS 的模板参数爆炸不同，CubeK 将 JIT 编译 key 限制在离散的 Blueprint 空间内。

## CubeK 在技术栈中的位置

CubeCL 提供了 `#[cube]` 宏和 JIT 编译管线——写 kernel 的能力。CubeK 在此基础上提供了**成品算子和架构纪律**：

```
Burn 后端 ──→ CubeK 成品算子（matmul/attention/convolution/...）
                    ↓
                CubeCL（#[cube] + IR + JIT + autotune）
                    ↓
                CUDA / Metal / Vulkan / WebGPU / CPU
```

CubeK 不只是在 CubeCL 上写几个 kernel 函数。它的核心资产是**一套防止 kernel 爆炸的架构约束**——Blueprint-Routine-Autotuner 三层纪律。

---

## 问题：Kernel 组合爆炸

一个生产级 matmul kernel 需要面对的选择空间：

- 硬件加速器：NVIDIA Tensor Core (CMMA)、AMD Matrix Core (MMA)、纯寄存器 (Register)、plane 向量化 (PlaneVec)、交错分片 (Interleaved)——5 种 TileMatmulKind
- 加载策略：同步全循环 (SyncCyclic)、同步全跨步 (SyncStrided)、同步全分块 (SyncTilewise)、异步全循环 (AsyncCyclic)、异步全跨步 (AsyncStrided)、异步全 TMA (AsyncTma)——6 种加载策略
- 分块缓冲：单缓冲 vs 双缓冲 (Single/Double)
- 分块大小：`TileSize`、`PartitionSize`、`StageSize`、`GlobalPartitionSize`——每个维度可独立选择

如果把所有这些参数放进 JIT 编译 key（即 CubeCL 的 `KernelId`），组合数爆炸：5 × 6 × 2 × (数十种大小组合) = **数百到上千个独立的 JIT 编译产物**。每个产物对应一份完整的 WGSL/SPIR-V/MSL 代码，编译耗时和缓存压力线性增长。

CUTLASS 用 C++ 模板参数做同样的事——它的"模板爆炸"是编译期二进制体积的爆炸。CubeK 的"kernel 爆炸"是 JIT 缓存 key 空间的爆炸。两种爆炸的根因相同：**把不该进 key 的参数放进了 key**。

---

## 解决方案：Blueprint-Routine-Autotuner 三层纪律

CubeK 将 kernel 参数按**决策时机**分为三层（`cubek/GUIDE.md`）：

```
Blueprint   ← JIT 编译时（进 KernelId）   — 只放结构性的、离散的选择
Routine     ← JIT 编译时（不进 KernelId） — 根据硬件和问题推断 Blueprint
Autotuner   ← 首次执行时                  — 在有限的 Blueprint 候选间选最快
```

### 第 1 层：Blueprint —— 进 JIT key 的结构性选择

```rust
// cubek/crates/cubek-matmul/src/definition/blueprint.rs:19
pub trait Blueprint: Debug + Clone + Eq + PartialEq + Hash {
    fn tiling_scheme(&self) -> TilingScheme;
    fn swizzle_modes(&self) -> SwizzleModes;
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn out_global_layout_config(&self) -> GlobalLayoutConfig;
}
```

`Blueprint` 必须实现 `Hash + Eq`——它的哈希值进入 CubeCL 的 `KernelId`。**Blueprinte 中放什么、不放什么，是 CubeK 最关键的纪律约束**：

**可以放**：算法变体（Cmma vs Register）、分块方案形状（`TilingScheme`）、swizzle 模式、安全检查策略（mask vs branch）。这些是离散的、有限的。

**不能放**：向量化因子（已在 JIT key 中）、`CubeDim`（已在 JIT key 中）、硬件属性（kernel 运行时获取）、问题大小（kernel 运行时参数）。

违反这条纪律的后果：如果 Blueprint 包含 `M`（问题大小的行数），每个不同的 `M` 值产生一个独立的编译产物——上千个 kernel。`TilingBlueprint`（`blueprint.rs:30`）遵守了这一纪律：它的字段全部是离散的结构性选择，不包含运行时变量。

### 第 2 层：Routine —— 硬件适配

```rust
// cubek/crates/cubek-matmul/src/routines/base.rs:21
pub trait Routine<RC: RuntimeConfig>: Sized {
    type Strategy: Default + Display + Clone;
    type Blueprint: Blueprint;

    fn expand_blueprint<R: Runtime>(problem, device_settings, strategy) -> ExpandInfo<Self::Blueprint>;
    fn prepare<R: Runtime>(problem, device_settings, expand_info) -> LaunchInfo<Self::Blueprint>;
    fn validate_blueprint<R: Runtime>(client, blueprint, ...) -> ...;
}
```

`Routine` 的职责是在 **JIT 编译 key 生成之前**，根据硬件和问题特征推断出最佳的 Blueprint。关键方法 `expand_blueprint` 做的是"探测 + 离散化"：

以 `SimpleAlgorithm` 为例（`routines/simple.rs:105`），它在 `infer_blueprint_plane` 中：

1. 探测 GPU 的 `features.matmul.cmma` 支持哪些 tile 大小。尝试固定优先序列表：16×16×16 → 8×8×8 → 32×8×16 → ...
2. 根据可用的 plane 数量计算 `rows_per_plane`、`stage_size_m`、`partition_shape_n`
3. 根据 stage 尺寸和元素大小选择 swizzle 模式
4. 组装为离散的 `TilingBlueprint`

**Routine 探测了连续空间，但输出离散的 Blueprint**。"试试 16×16×16 行不行"是一个二元判定——可行就选它，不行就试下一个——不是"把所有可能的 tile 大小都生成一份 Blueprint"。

### 第 3 层：Autotuner —— 有限候选间选最快

前两层保证了到达 Autotuner 的 **Blueprint 候选数在可控范围内**。CubeK 的 `Strategy` 枚举（`launch/strategy.rs:72`）有 ~34 个变体，覆盖了 5 种 TileMatmulKind × 多种加载策略的组合。但通过 `BlueprintStrategy` 的分层，实际 benchmark 的候选数远小于 34：

```rust
// launch/strategy.rs:542 — Strategy::Auto 的默认行为
fn auto<R: Runtime>(...) {
    // 先试 CMMA（tensor core 路径）
    if let Err(err) = Strategy::SimpleCyclicCmma(Default::default()).launch_ref(...) {
        match err {
            MatmulSetupError::Unavailable(_) => {
                // CMMA 不可用 → 退到纯标量路径
                Strategy::SimpleUnit(Default::default()).launch_ref(...)?;
            }
        }
    }
}
```

`Strategy::Auto` 不是 exhaustive search——它是**级联回退**：CMMA 可用时选 CMMA，硬件不支持时退到 Register/Unit。这是生产级的行为。需要 exhaustive autotune 的场景可以通过显式枚举 `Strategy` 变体 + CubeCL 的 `AutotuneKey` 系统实现。

CubeK 的 autotune key（`launch/tune_key.rs`）比 burn-cubecl 的 key 更丰富：

```rust
pub struct MatmulAutotuneKey {
    pub definition: MatmulProblemDefinition,  // m, n, k, stride_factor, layout
    pub analysis: MatmulAutotuneAnalysis,      // GlobalScale { Large/Medium/Small } + MatmulKind
}
```

`stride_factor` 和 `pow2_factor` 捕获了向量化的可行性——一个 128 字节对齐的矩阵和 16 字节对齐的矩阵在 autotune 层面是不同的 key。`MatmulAutotuneAnalysis` 做 scale 分桶——Large 问题优先试双缓冲，Small 问题跳过双缓冲。

---

## TileMatmulKind：五种硬件路径

`TileMatmulKind` 枚举（`cubek-matmul/src/components/tile/mod.rs:94`）是 CubeK 对硬件矩阵乘法的抽象：

| 变体 | 硬件 | 作用域 | 约束 |
|------|------|--------|------|
| `Cmma` | NVIDIA Tensor Core | Plane | `features.matmul.cmma` 支持的 tile 大小 |
| `Mma` | AMD/CDNA Matrix Core | Plane | `features.matmul.mma` 支持的 tile 大小 |
| `Register` | 纯标量 FMA | Unit | 无特殊约束 |
| `PlaneVec` | Plane 级向量规约 | Plane | k == plane_dim × vector_size，LHS RowMajor，RHS ColMajor |
| `Interleaved` | k 维按 plane 交错分片 | Plane | k 能被 plane_dim 整除 |

每个变体实现 `TileVariant` trait（`tile/variant.rs:20`）：`requires_accelerator()`（是否需要硬件加速器）、`is_supported()`（当前 GPU 是否支持）、`supported_sizes()`（支持的 tile 大小集合）、`expand()`（生成 IR 操作）、`validate()`（验证 Blueprint 对硬件是否有效）。

`Cmma` 和 `Mma` 的分离体现了 CubeK 的跨平台意识——它们在同一层抽象下处理 NVIDIA 和 AMD 的不同 tensor core 指令集。`Register` 是通用回退——任何 GPU 上都能跑，只是速度慢。`PlaneVec` 和 `Interleaved` 是在"没有 tensor core 但想做向量化"时的高效替代。

---

## 与 CUTLASS 的对比

CubeK 显式受了 CUTLASS（NVIDIA 的 CUDA C++ 模板库）的设计影响。两者的共性和差异：

### 共性

- **分层分块模型**：CUTLASS ThreadBlock → Warp → Thread。CubeK Global → Stage → Tile。`TilingScheme` 的四层大小（`TileSize` → `PartitionSize` → `StageSize` → `GlobalPartitionSize`）直接对应 CUTLASS 的层级分解
- **数据移动策略的参数化**：CUTLASS 模板参数化 `Mma`、`SmemLayout`。CubeK trait 参数化加载策略（`SyncFullCyclicLoading`、`AsyncFullTmaLoading` 等）
- **Epilogue 分离**：CUTLASS 的 "mainloop + epilogue" 对应 CubeK 的 `GlobalRead → TileMatmul → GlobalWrite`
- **多阶段双缓冲**：CUTLASS 的软件流水线对应 CubeK 的 `PartitionBuffering::Double` + `multi_stage`

### 差异

| 维度 | CUTLASS | CubeK |
|------|---------|-------|
| 语言 | C++ 模板，库编译期实例化 | Rust `#[cube]`，应用运行时 JIT |
| 组合爆炸位置 | 二进制体积（模板实例化 N 份） | JIT 缓存（KernelId 哈希 N 个） |
| 爆炸管理 | 限制模板组合 | Blueprint 纪律 + Routine 离散化 |
| Autotune | `cutlass_profiler` 离线或启动时 | CubeCL `AutotuneKey`，线上首次执行 |
| 平台 | 仅 NVIDIA | CUDA + Metal + Vulkan + WebGPU + CPU |
| Tile 种类 | 隐式（通过模板参数组合） | 显式 `TileMatmulKind` 枚举（5 种） |
| Swizzle | 可选优化 | 嵌入 Blueprint 的一等公民 |
| 计算域 | 隐式（warp vs thread） | 显式 `TileScope`（Unit/Plane/Cube）类型标记 |

最根本的设计差异：**CUTLASS 用编译期模板参数生成 kernel 变体，CubeK 用 JIT key 的哈希生成 kernel 变体**。两种方案都面临组合爆炸，但爆炸发生的位置不同。CubeK 的 Blueprint 纪律等同于 CUTLASS 的"只允许某些模板参数组合"——只是它用 trait 边界和离散枚举来执行，而非模板特化。

---

## 一个 matmul 的实际执行路径

以 `Strategy::SimpleCyclicCmma` 为例，f32 精度：

```
1. 用户代码: tensor.matmul(&other)
   ↓
   经过 Burn 后端 → CubeK::launch_ref()

2. 展开 Blueprint（Routine::expand_blueprint）
   → 探测 GPU features.matmul.cmma 支持的 tile 大小
   → 尝试 16×16×16 → 支持 → 选定
   → 计算 Plane 映射、Partition 大小、Swizzle 模式
   → 输出 TilingBlueprint { tile_matmul: Cmma(16,16,16), swizzle: B32, ... }

3. 验证 Blueprint（Routine::prepare）
   → CmmaMatmul::validate(blueprint) → 验证 16×16×16 在 supported_sizes 中
   → 计算 CubeDim 和 CubeCountPlan
   → 输出 LaunchInfo

4. 编译 + 启动（Routine::launch）
   → PartitionedBatchMatmulFamily::launch_unchecked
   → CubeCL JIT key = (kernel_fn, cube_dim, ..., blueprint_hash)
   → 编译 WGSL/SPIR-V/MSL → GPU dispatch
   → 在 kernel 内部: GlobalRead → CmmaTileMatmul → GlobalWrite

5. [可选] Autotune
   → 若 Strategy::Auto → 先试 SimpleCyclicCmma → 硬件不支持则退回 SimpleUnit
   → 若显式 autotune → CubeCL 的 AutotuneKey 系统 benchmark 候选 Strategy
```

---

## 限制

1. **Routine 的启发式不是全局最优**：`find_instruction_size` 的固定优先序列表（16×16→8×8→32×8→...）是在常见硬件上的经验选择，不是 exhaustive search。在非常规硬件或未来架构上可能不是最优。

2. **Blueprint 的离散化可能遗漏中间配置**：Routine 将连续空间映射为离散 Blueprint——"16×16×16 和 8×8×8 哪个更好"能回答，"17×15×14 怎么样"不能问。硬件支持非均匀 tile 大小时，离散化可能跳过有用配置。

3. **Swizzle 选择是启发式的**：`select_swizzle` 基于 stage 尺寸和元素大小的经验公式选择，不是 benchmark。

4. **Autotune 候选数仍依赖 CubeCL 的优先级剪枝**：~34 个 Strategy 变体 × 多种 hardware/problem 组合，全枚举仍然不小。CubeK 依赖 CubeCL 的 `TuneGroup` 优先级提前终止来控制数量。

5. **非 matmul 算子的 Blueprint 设计不完整**：attention、convolution、reduce 有各自的 kernel，但 Blueprint 纪律的一致性和完整性在不同子 crate 之间不统一。

---

## 关键源码入口

- Blueprint 规范：`cubek/GUIDE.md`
- Blueprint trait：`crates/cubek-matmul/src/definition/blueprint.rs`
- Routine trait：`crates/cubek-matmul/src/routines/base.rs`
- Routine 选择器（启发式）：`crates/cubek-matmul/src/routines/selector/plane.rs`
- Strategy 枚举 + Auto 回退：`crates/cubek-matmul/src/launch/strategy.rs`
- Autotune key：`crates/cubek-matmul/src/launch/tune_key.rs`
- TileMatmulKind：`crates/cubek-matmul/src/components/tile/mod.rs`
- TilingScheme：`cubek-std/src/tile/`（`TileSize`、`PartitionSize`、`StageSize`、tiling walk orders）

---

← [Autotune 系统设计](../cubecl/autotune-system-design.md) | → 下一篇：[Autodiff 系统设计](../burn/autodiff-system-design.md)
