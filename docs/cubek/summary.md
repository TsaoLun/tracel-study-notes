# CubeK：Blueprint-Routine-Autotuner 三层纪律与成品内核

## 读前须知

- **CubeK 是什么**：Tracel 的基于 CubeCL 的高性能算子库——提供 matmul、attention、convolution、reduce、quant、FFT 等成品内核。它在 CubeCL 编译器之上，用 Blueprint-Routine-Autotuner 三层纪律组织内核开发和性能调优。
- **本文覆盖**：三层纪律的设计动机、TileKind 系统、kernel explosion 的预防策略、与 CubeCL compile/autotune 的分界线。本文不逐算子展开——各算子的具体内核变体在专题中展开。
- **前置**：[CubeCL 地图](../cubecl/summary.md)（#[cube]、comptime、autotune 概念）。与 CubeCL 的关系是：CubeCL 提供"怎么写 kernel"的编译器框架；CubeK 提供"写好的 kernel"的成品库 + 纪律约束。
- **机制基准**：cubek 仓库 `crates/cubek-std/src/tile/base.rs`（TileKind）、根目录 `GUIDE.md`（三层纪律）。源码行号为近似值。

系列分工与导航见 [README](../../README.md)。

---

## 架构一览

```
应用 / Burn
    ↓
CubeK（成品内核）
    ├── cubek-matmul    ← MatMul（CMMA / PlaneVec / Register tile）
    ├── cubek-attention ← 注意力（multi-head、masked、causal）
    ├── cubek-convolution ← 卷积（im2col / direct、mma / unit / tma）
    ├── cubek-reduce    ← 规约（sum、max、argmax、per-cube / per-plane）
    ├── cubek-quant     ← 量化（symmetric、per-block、q2-q8、fp4）
    ├── cubek-random    ← 随机数（bernoulli、normal、uniform）
    ├── cubek-fft       ← 快速傅里叶变换
    ├── cubek-pool      ← 池化
    └── cubek-interpolate ← 插值
    ↓
CubeCL（#[cube] + IR + JIT + autotune）
    ↓
CUDA / HIP / WGPU / CPU …
```

CubeK 不是 CubeCL 的替代，而是其上的**纪律层**——它不提供新的编译器机制，而是提供"如何组织内核使其可组合、可调优、不会组合爆炸"的规范。

---

## 核心结论

> CubeK 的三层纪律（Blueprint-Routine-Autotuner）把内核开发中三个容易混淆的决策维度——编译期结构选择、运行期参数适配、首次执行 benchmark——严格分层。Blueprint 只放改变控制流或指令选择的参数（防 kernel explosion）；Routine 从 launch 层接收约束后计算 `LaunchSettings`（不做硬件硬决策）；Autotuner 在同一 blueprint 下 benchmark 选最快实现。三层纪律的核心目标是：在 13 种 TileKind × 3×3×3 blueprint 空间的组合压力下，保持 JIT 缓存可控。

---

## 一、三层纪律

CubeK 的 [GUIDE.md](https://github.com/tracel-ai/cubek/blob/main/GUIDE.md) 规定了内核开发的三层分离：

| 层 | 职责 | 约束 | 决策时机 |
|----|------|------|----------|
| **Blueprint** | JIT 特化参数——决定 kernel 的**结构** | 只放改变控制流或指令选择的参数；排除 vectorization、cube dim、硬件属性、问题尺寸 | JIT 编译时（L2） |
| **Routine** | 每次 launch 计算 `LaunchSettings`（cube_dim、cube_count） | 不做硬件硬决策，从 launch 层接收约束后计算映射 | launch 调用时 |
| **Autotuner** | 首次遇 autotune key 时 benchmark 所有候选 | 在同一 blueprint 下选最快 Routine 组合 | 首次执行时（L3） |

### 为什么需要三层分离

在没有严格分层的情况下，内核开发者容易把"编译期参数"和"运行期参数"混在一起：

```
// 错误的做法：把所有参数塞进 blueprint
#[cube(launch_unchecked)]
fn matmul<F: Float>(
    lhs: &Tensor<Vector<F>>,
    rhs: &Tensor<Vector<F>>,
    out: &mut Tensor<Vector<F>>,
    #[comptime] blueprint: MatmulBlueprint {  // ← 放了太多东西
        algorithm: Algorithm,       // ✓ 改变控制流
        stage_size: u32,           // ✓ 改变指令
        vector_size: u32,          // ✗ 已在 JIT key 中
        cube_dim: CubeDim,         // ✗ 运行时参数
        problem_shape: (u32,u32,u32), // ✗ 运行时参数
    },
) { /* ... */ }
```

每次 `vector_size` 或 `cube_dim` 变化就生成一份独立 JIT 产物——组合爆炸。Blueprint 纪律就是把"不必要在编译期变化的"参数排除在外。

---

## 二、TileKind 系统：13 种 tile 策略

`crates/cubek-std/src/tile/base.rs` 定义了 13 种 `TileKind` 变体，代表三类策略：

### Tensor Core 路径（有 tensor core 时最快）

| TileKind | 含义 |
|----------|------|
| `Cmma` | NVIDIA tensor core WMMA intrinsic 片段 |
| `Mma` | 通用矩阵乘累加片段（Lhs/Rhs/Acc 三种角色） |
| `Pipelined` | 双缓冲流水线 tile（single/double-buffered） |
| `Bounce` | CMMA 片段 + shared memory scratch + whitebox 视图 |

### Warp-level 向量化（无 tensor core 时通常最优）

| TileKind | 含义 |
|----------|------|
| `PlaneVec` | warp-level 向量化 matmul |
| `Interleaved` | plane-interleaved-on-k matmul |
| `WhiteboxFragment` | plane 级暴露布局的片段 |
| `RowWise` | per-row 向量 tile（softmax max/sum state） |

### 通用路径

| TileKind | 含义 |
|----------|------|
| `Register` | 纯软件 tile——通用但无硬件加速 |
| `Stage` | whole-stage 视图，用于 partition 级 dispatch |
| `Partition` | per-tile 累加器序列 |
| `SharedTile` | shared memory 暴露为 tile（无计算） |
| `Unit` | per-unit 寄存器数组 |

`Tile` 结构体包装 `TileKind` + `ScopeMarker`（编译期追踪 tile 的作用域：Unit / Plane / Cube）。不同的核函数组合使用不同的 tile 子集——matmul 用 PlaneVec + Register + Cmma，attention 用 RowWise + Pipelined，convolution 用 Stage + Partition。

---

## 三、kernel explosion 的算术

Blueprint 的参数空间是 JIT 编译的维度——每个不同 blueprint 组合生成一份独立的 JIT 产物。TileKind 是 autotune 的维度——同一 blueprint 下 benchmark 不同 tile 策略。

**实际数字**（matmul 为例）：
- Blueprint 参数：algorithm（3 种：mma / unit / tma）× stage_size（3 种）× multi-stage（3 种）= **27 种编译变体**
- TileKind：**~13 种**
- 候选总数：27 × 13 = **351 个候选**

不严格控制哪些参数进 blueprint，组合爆炸会让 JIT 缓存不可接受。GUIDE.md 的排除规则——vectorization（已在 JIT key 中）、cube dim（运行时参数）、硬件属性（kernel 内 `comptime::device_properties()` 获取）、问题尺寸（runtime arguments）——精确地为这个算术服务。

---

## 四、Blueprint 编译流程

以 matmul 为例，一个完整的 launch 决策链：

```
用户调用 matmul(lhs, rhs, out, device)
    ↓
1. Routine::compute() 从 launch 层接收约束（vectorization、硬件属性）
    → 计算 LaunchSettings（cube_dim、cube_count）
    → 生成 Blueprint { algorithm, stage_size, multi_stage }
    ↓
2. Autotuner 查 autotune key: (M, N, K, dtype, layout)
    → hit: 直接用缓存的最优 (blueprint, tile_kind) 组合
    → miss: benchmark 候选 → 缓存最优
    ↓
3. kernel launch: 使用选定的 blueprint + tile_kind
    → JIT miss → expand + opt + codegen（CubeCL 管线）
    → 执行
```

Blueprint 在步骤 1 中由 Routine 生成（不暴露给用户选择）。用户只指定精度和布局偏好——框架负责选择"什么 blueprint + 什么 tile"。

---

## 五、与 CubeCL comptime/autotune 的分界线

CubeCL 本身提供 comptime 和 autotune 两种机制。CubeK 在此基础上加了第三层：

| 维度 | CubeCL comptime | CubeCL autotune | CubeK Blueprint | CubeK Autotuner |
|------|----------------|-----------------|-----------------|
| 决策内容 | 单个 kernel 的结构（`if plane { … }`） | 同一 kernel 的多种实现变体 | **kernels 的编译期组合**（算法选择、stage 大小） | 同一 blueprint 下的最优 (tile_kind, routine) 组合 |
| 粒度 | per-kernel | per-kernel per-shape | per-algorithm-family | per-autotune-key |
| 谁定义 | kernel 作者（`#[comptime]` 参数） | kernel 作者（`#[autotune]` 注解） | **CubeK Routine 层**（算法选择逻辑） | **CubeK Autotuner 层**（benchmark 框架） |
| 爆炸风险 | comptime 参数过多 | anchor 分桶控制 | **blueprint 参数空间有限**（纪律约束） | TileKind 数量有限 |

CubeK 的三层纪律是 CubeCL 机制上的**组织层**——它不提供新的编译能力，而是提供"如何把 CubeCL 的能力用在不炸的组合空间里"的纪律。

---

## 六、与 Burn 的边界

Burn 通过 `burn-cubecl` 调用 CubeK 内核：

```
Burn 用户代码
    ↓
Backend trait → Fusion → burn-cubecl
    ↓
CubeK kernel（matmul / attention / …）
    ↓
CubeCL JIT + autotune
    ↓
GPU
```

CubeK 对 Burn 暴露的是**算子级别的成品接口**——Burn 不关心 matmul 用的是 CmmaTile 还是 PlaneVecTile，只关心"这个 matmul 能在当前后端上跑"。CubeK 对 CubeCL 暴露的是**blueprint + tile_kind 的组合选择**——CubeCL 不关心 tile 策略怎么选的，只负责按给定的 blueprint 编译并 launch。

---

## 词汇说明表

| 术语 | 简要说明 |
|------|----------|
| **Blueprint** | CubeK 中的编译期特化参数集合：只放改变控制流或指令选择的参数，排除已在 JIT key 中的信息 |
| **Routine** | 从 launch 层接收约束（vectorization 等），计算 `LaunchSettings` 和生成 `Blueprint` 的适配层 |
| **Autotuner** | 首次遇 autotune key 时 benchmark 所有候选，缓存最优 (blueprint, tile_kind) 组合 |
| **TileKind** | 13 种 tile 实现策略的枚举：Cmma（tensor core）、PlaneVec（warp 向量化）、Register（纯软件）等 |
| **kernel explosion** | blueprint 参数空间过大 → JIT 编译产物指数增长，缓存与编译时间失控 |
| **Stage** | 软件流水线的级数——multistage 通过更大的 shared memory 占用换更高的计算/访存比 |
| **CMMA** | CUDA Matrix Multiply Accumulate——NVIDIA tensor core 的 WMMA intrinsic 包装 |
| **Plane** | CubeCL 的锁步协作单元（对应 CUDA warp / WebGPU subgroup） |

*Burn 底层机制 · CubeK 地图 · 导航见 [README](../../README.md)*
