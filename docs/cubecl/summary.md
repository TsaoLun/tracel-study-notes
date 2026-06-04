# CubeCL：`#[cube]` 如何把 Rust 函数变成六平台的 GPU 代码

## 读前须知

- **CubeCL 是什么**：Tracel 的 Rust GPU 计算库 + 编译器框架——`#[cube]` proc-macro 描述 kernel，首次 launch 时 JIT 到 CUDA/HIP/WGPU/CPU，不为每个平台手写 `.cu`/`.wgsl`/`.metal`。
- **本文覆盖**：`#[cube]` 宏展开、comptime/autotune 双机制、JIT 管线、CubeK 纪律——作为 Burn 底层机制系列的 GPU 篇。跟练走 [CubeCL 专题](index.md)。
- **机制基准**：cubecl 仓库 `cubecl-opt/src/lib.rs`（JIT pass）、cubek 仓库 `crates/cubek-std/src/tile/base.rs`（TileKind）。术语首次出现括号简注，完整释义见文末词汇表。

系列分工与导航见 [README](../../README.md)。

---

## 架构一览

```
Rust + #[cube] 源码                    ← 人写的 kernel 描述
    ↓ proc-macro 展开
expand 模块（IR 生成器）               ← 不执行浮点运算，只向 Scope 填 Operation
    ↓ 首次 launch
cubecl-opt（CFG → SSA → 定点循环）     ← 10 个 pass 反复执行至收敛
    ↓
编译器后端（CudaDialect / WgslCompiler / MLIR …）
    ↓
PTX / WGSL / SPIR-V / SIMD …          ← 磁盘缓存，后续命中
    ↓
GPU / CPU 执行
```

Comptime 在 JIT 时决定 kernel 结构（哪些指令）；Autotune 在首次执行时 benchmark 选最快实现变体。两者把"针对平台的决策"从"写代码时"推迟到"JIT 编译时"或"首次执行时"。

---

## 核心结论

> `#[cube]` proc-macro 保留你的 Rust 函数供类型检查，并生成以函数名命名的子模块——其中的 `expand` 在首次 launch 时向 IR 的 `Scope` 里填入 `Operation` 指令。你写的源码不直接编译为 GPU 机器码——它是一段在 JIT 时运行的 IR 生成器；再由 `cubecl-opt` 优化、各后端编译。

---

## 心智模型：你写的是 IR 生成器

传统 GPU 编程：人写 `.cu` → 编译器出 PTX。CubeCL 多了一步：

```
Rust 函数 + #[cube] → proc-macro 生成 expand 模块
    → 首次 launch 时（经 runtime define()）调用 expand，向 Scope 填 Operation
    → cubecl-opt 优化（CFG → SSA → pass 循环）
    → 编译器后端生成目标代码
```

`#[cube(launch)]` 做两层生成（`cubecl-macros/src/generate/launch.rs`）：

1. **原始函数**：保留在 crate，走 Rust 类型检查——保证 kernel 写法合法。
2. **Expand 模块**：proc-macro 以函数名创建子模块，内部函数固定命名为 `expand`。它不执行浮点运算——把 `+` 翻译为 `Operation::Arithmetic(Add, …)`，把 `if` 翻译为 `Operation::Branch(…)`，把 `ABSOLUTE_POS` 翻译为 `Variable::Builtin(AbsolutePos)`，全部填入 `cubecl_ir::Scope`。

`comptime! { … }` 里的代码在 expand 执行时作为普通 Rust 运行，结果烘焙进 IR 常量——与 Rust `const` 不同（后者在 `cargo build` 时固定）。

测试 kernel 的常见路径：CPU runtime launch（`--features cpu`），或对 `#[cube(launch)]` kernel 调用 **`{Name}Kernel::new` + `define()`** 打印 Scope（不 launch）。

---

## Comptime：JIT 时才做的决策

`#[comptime]` 参数参与 kernel 特化，但不作为 GPU 运行时参数传入。与 Rust `const` 的区别：`const` 在编你的 crate 时固定；`comptime` 在首次 JIT 该 kernel 时固定，结果写进 IR 常量，GPU 上不再分支。

GELU 里的 `comptime!(2.0f32.sqrt())` 在 expand 填 IR 时直接变成常量——GPU 每次 launch 不必算 √2。更关键的是结构级决策：

```rust
#[cube(launch)]
fn sum_plane<F: Float>(
    input: &[F], output: &mut [F],
    #[comptime] plane: bool,
) {
    if plane {
        output[UNIT_POS] = plane_sum(input[UNIT_POS]);
    } else {
        sum_basic(input, output, /* ... */);
    }
}
```

`plane: true/false` 生成两份不同的 JIT 产物——GPU 代码中不存在这个分支。这避免了"在不支持 subgroup 的硬件上编译包含 subgroup 指令的 kernel"（传统方案需 `#ifdef`）。

循环展开同样由 comptime 控制（`#[unroll]`），`end` 必须是 comptime 可确定的——防止性能悬崖。

CubeK 的 `GemmBlueprint` 是 comptime 的最高形态：决定 dot vs outer product 算法、stage 大小、是否 double-buffer、是否 plane 级协作。每种组合对应一份独立 JIT 产物，这正是 CubeK 要求 blueprint 保持极小的原因——防 kernel explosion。

---

## 自动向量化：launch 时注入

在 `launch` 时传入 vectorization factor（如 GELU 示例的 `vector_size: 4`），kernel 内写 `Vector<F, N>`。编译器据此生成合适宽度的 load/store，标量与向量混用时自动广播。

三种特化维度的分工：

| 维度 | 谁决定 | 何时固定 | 影响 |
|------|--------|----------|------|
| **Vectorization** | `launch` 参数（进入 JIT key） | 首次 JIT 该 vector 宽度 | 不同宽度的 load/store |
| **Comptime** | `#[comptime]` 参数 | 首次 JIT 该 blueprint | 控制流、循环展开、plane vs scalar |
| **Autotune** | benchmark | 首次执行该 shape key | 在已编译候选中选最快 launch |

---

## 四轴并行与统一拓扑

CubeCL 用四条正交轴描述硬件并行，好的 kernel 在 comptime 读取这些值自适应，不硬编码 `warpSize == 32`：

| 轴 | 含义 | 谁配置 |
|----|------|--------|
| **Vector** | 指令级 SIMD，一个 unit 一次处理 N 个 lane | launch 时指定 |
| **Plane** | lockstep 协作单元（warp/subgroup/simdgroup） | runtime 按硬件决定 |
| **CubeDim** | 一个 cube 内的并发 unit 数，共享内存与同步 | launch 时指定 |
| **CubeCount** | 启动多少个 cube | launch 时指定 |

CubeCL 的命名故意不与任何平台对齐：

| CubeCL | CUDA | WebGPU | Metal |
|--------|------|--------|-------|
| `CUBE_POS_X` | `blockIdx.x` | `workgroup_id.x` | `threadgroup_position_in_grid.x` |
| `UNIT_POS_X` | `threadIdx.x` | `local_invocation_id.x` | `thread_position_in_threadgroup.x` |
| `ABSOLUTE_POS_X` | 由后端合成* | `global_id.x` | `thread_position_in_grid.x` |
| `PLANE_DIM` | `warpSize` | `subgroup_size` | `threads_per_simdgroup` |

\* `ABSOLUTE_POS` 是 CubeCL 的合成抽象（`cubecl-core/src/frontend/topology.rs`），后端用 cube 维度与位置组合计算。写 elementwise kernel 只需 `output[ABSOLUTE_POS] = …`，不必手写 `blockIdx * blockDim + threadIdx`。

---

## 为什么是 proc-macro 而不是 trait-based 注册——CubeCL 的设计选择

一个常见的问题：如果 CubeCL 的目标是"写 Rust 然后在 GPU 上跑"，为什么不提供一个 Builder API 或者 trait 来注册 GPU 操作？比如：

```rust
// 假想的 trait-based 方案——实际不存在
struct MyKernel;
impl GpuKernel for MyKernel {
    fn build(scope: &mut Scope) {
        let a = scope.param::<f32>("a");
        let b = scope.param::<f32>("b");
        scope.register(Instruction::Add { lhs: a, rhs: b }, scope.output());
    }
}
```

CubeCL 选择 `#[cube]` proc-macro 而非 trait/builder，有三个结构性原因：

1. **与 Rust 类型检查的集成**：proc-macro 保留原始函数——任何类型错误（`f32 + bool`、数组边界越界）由 rustc 直接报告，不需要框架定义自己的类型系统。Builder API 必须在运行时做类型检查（或自己实现一套编译期检查）。

2. **comptime 的实现可行性**：`comptime!(2.0f32.sqrt())` 是**普通 Rust 代码**，在 expand 执行时由 Rust 运行时求值。如果使用 Builder API 在 proc-macro 中注册操作，`comptime!` 里的 Rust 代码就无法执行——proc-macro 运行的进程无权限执行任意 Rust 表达式。两阶段设计（proc-macro 生成 expand → expand 作为 Rust 函数执行）是 comptime 的前提条件。

3. **控制流的自然表达**：`if plane { plane_sum(...) } else { sum_basic(...) }` 作为 Rust 控制流写起来自然。trait-based 注册需要用 `scope.if_else(condition, |then| { ... }, |else| { ... })` 的 Builder 风格——不是不行，但阅读体验和写错概率不同。

代价是 proc-macro 的复杂度——`cubecl-macros` 需要实现一个完整的 Rust 子集解析器（AST → `Expression` 枚举），包括二元运算、方法调用、闭包、match、for 循环等。这是 CubeCL 最复杂的单个 crate。源码验证路径：`cubecl-macros/src/parse/expression.rs`（Expression 枚举）、`cubecl-macros/src/generate/expression.rs`（Expression::to_tokens 分发）。

> 跟练验证：[第二章](2-expand.md) 的 `define()` 方法可以打印 expand 生成的 Scope，直观感受 `a + b * c` 从 Rust 表达式变成哪些 IR 指令。

---

## JIT 管线：post-SSA 定点循环

当 `launch` 第一次遇到某个 `(kernel, comptime, vectorization, cube_dim, …)` 组合时，`expand` 被调用填入 IR 指令，然后进入 `cubecl-opt`。

**地图视角**：四个阶段——① 解析 Scope 为 CFG → ② SSA 变换 → ③ 定点循环（10 个 pass 反复执行至收敛）→ ④ 一次性重优化 + 后端代码生成。

<details>
<summary>完整 pass 清单（参考，可跳过）</summary>

`Function::run_opt()`（`cubecl-opt/src/lib.rs`）的流程：parse_graph → split_critical_edges → SSA 变换 + CompositeMerge 定点循环 → PointerSource 分析 → **apply_post_ssa_passes**（InlineAssignments / EliminateUnusedVariables / ConstOperandSimplify / MergeSameExpressions / ConstEval / RemoveIndexScalar / EliminateConstBranches / EmptyBranchToSelect / EliminateDeadBlocks / EliminateDeadPhi，循环至无变化）→ 一次性重优化（DisaggregateArray / GVN / ReduceStrength / CopyTransform）→ split_free → SharedLiveness → MergeBlocks → Captures → update_buffer_vis。

</details>

输出是 `petgraph::StableDiGraph<BasicBlock>`，后端编译器遍历此图生成目标代码。在非 SSA 目标语言中，phi 被模拟为在相应前驱块末尾赋值给可变变量。

各 runtime 编译路径：

| 平台 | Runtime | 编译器 | 编译产物 |
|------|---------|--------|----------|
| CUDA | `cubecl-cuda` | CppCompiler<CudaDialect> | NVRTC → PTX |
| ROCm | `cubecl-hip` | CppCompiler<HipDialect> | hipRTC → AMD GPU 码 |
| WGPU | `cubecl-wgpu` | WgslCompiler / MslDialect / SPIR-V | wgpu 驱动加载 |
| CPU | `cubecl-cpu` | MlirCompiler → MLIR → LLVM JIT | 本机 SIMD |

JIT 产物通过 `KernelId::stable_hash()` 缓存到 `CompilationCache`，同一组合的第二次 launch 直接命中。

---

## Autotune：同一 blueprint 下选最快实现

Comptime 决定"生成什么结构的 kernel"。Autotune 决定"用哪个已编译变体最快"：在真实 GPU + 真实 shape 上 benchmark，缓存最快方案的索引。

CubeK matmul 的 tile 系统有 13 种 `TileKind`（`cubek-std/src/tile/base.rs`），代表三类策略：

- **CmmaTile**：NVIDIA tensor core WMMA intrinsic——有 tensor core 时最快
- **PlaneVecTile**：warp-level 向量化——无 tensor core 的 GPU 上通常最优
- **RegisterTile**：纯软件 tile，通用但慢

`cubecl-runtime` 的 `Tuner<K: AutotuneKey>` 在首次遇到新 `(M,N,K,dtype,layout)` 时 benchmark 候选 kernel。`TuneGroup` 优先级让高分组的候选先试（CmmaMatmul > RegisterMatmul），避免不必要的 benchmark。结果可持久化到磁盘。

`#[autotune(anchor(...))]` 做指数分桶——`exp(min=16, max=1024, base=2)` 把 16–1024 按 2 的幂分 7 个桶，避免每个尺寸都触发完整 benchmark。

### Comptime × Autotune 分界线

| 维度 | Comptime / Blueprint | Autotune |
|------|----------------------|----------|
| 决策时机 | JIT 编译时 | 首次执行时 |
| 决策内容 | kernel **结构**（循环展开、plane vs scalar） | kernel **实现变体**（CMMA vs Register） |
| 缓存粒度 | 每个 blueprint 组合一份编译产物 | 每个 autotune key 缓存最快变体索引 |
| 爆炸风险 | blueprint 参数每多一个取值，JIT 产物翻倍 | 指数分桶控制粒度 |
| 失败形态 | JIT 编译失败（平台不支持） | 跑了次优 kernel（仍然能跑） |

---

## CubeK 的纪律

CubeK（[tracel-ai/cubek](https://github.com/tracel-ai/cubek)，与 cubecl 分仓）在 CubeCL 上提供成品内核——matmul、attention、convolution、reduce、quant、FFT。设计重心在 **Blueprint-Routine-Autotuner 三层纪律**（[GUIDE.md](https://github.com/tracel-ai/cubek/blob/main/GUIDE.md)）：

| 层 | 职责 | 约束 |
|----|------|------|
| **Blueprint** | JIT 特化参数 | 只放改变控制流或指令选择的参数，排除 vectorization、cube dim、硬件属性、问题尺寸 |
| **Routine** | 每次 launch 计算 `LaunchSettings` | 不做硬件硬决策，从 launch 层接收约束 |
| **Autotuner** | 首次遇 key 时 benchmark | 找到最快 Routine + Blueprint 组合 |

三层纪律的目标是防 kernel explosion：3×3×3 blueprint 参数空间 = 27 份 JIT 产物，×13 种 TileKind ≈ 351 个候选。不严格控制哪些参数进 blueprint（编译维度）vs autotune（运行维度），组合爆炸会让 JIT 缓存不可接受。

---

## CubeCL 与 Burn 的边界

CubeCL 是可独立使用的多平台 GPU 计算库。Burn 通过 `burn-cubecl` 调用 CubeK 内核：

```
用户模型/应用
    ↓
Burn（Autodiff + Fusion + Backend trait）     ← Burn 综合地图
burn-cubecl → CubeK 内核（cubek 仓库）
    ↓
CubeCL（#[cube] + IR + JIT + autotune）        ← 本篇
    ↓
CUDA / HIP / WGPU / CPU …
```

- **直接用 CubeCL**：自定义计算（稀疏 attention、科学模拟），跨平台可移植
- **用 Burn + CubeK**：端到端深度学习，需要 Backend trait 生态

---

## 诚实的局限

CubeCL README 标注 **alpha**：

1. 并非所有平台支持相同特性（如 WebGPU 尚无 tensor core 路径）。不支持的指令在 JIT 编译时明确失败，不静默降级。
2. 冷启动有成本——首次 launch 触发 JIT 编译 + autotune benchmark。磁盘缓存可缓解，但首次比热路径慢是 JIT 模型固有特性。

与成熟 CUDA 生态的差距在积累深度（cuDNN 等经多年硬件代际打磨），不在"能不能做"。

---

## 一次 launch 的完整旅程（GELU，CUDA）

```
gelu_array::launch_unchecked(client, cube_count, cube_dim, vectorization, buffers...)
    ↓
JIT 缓存 miss → KernelBuilder 调用 gelu_array::expand → 填入 Scope
    ↓
cubecl-opt: parse_graph → SSA → apply_post_ssa_passes（定点）→ GVN / 强度削减 …
    ↓
CppCompiler<CudaDialect> → CUDA C++ → NVRTC → PTX → CompilationCache 写入磁盘
    ↓
cudarc launch；ABSOLUTE_POS 写 output
```

换 `--features wgpu`，中间换成 WgslCompiler，无 NVRTC。`#[cube]` 函数体不变。跟练走 [CubeCL 专题 · 第一章](1-gelu-launch.md)。

---

## 词汇说明表

全文术语集中释义。按主题分组。

### 核心机制

| 术语 | 简要说明 |
|------|----------|
| **`#[cube]`** | proc-macro：保留原函数供类型检查，生成 `{fn}::expand` 子模块在 launch 时向 IR 填指令 |
| **Expand** | JIT 阶段在 host 执行的 IR 生成器，不直接做 GPU 浮点运算 |
| **Comptime** | `#[comptime]` / `comptime!`：JIT 该 kernel 时在 host 求值/分支，烘焙进 IR，非 GPU 运行时参数 |
| **Autotune** | 对多种已编译实现 benchmark，按 `(shape, dtype, device…)` 缓存最快候选索引 |
| **Blueprint** | CubeK 中描述 kernel 结构的 comptime 配置；防 kernel explosion 是它的核心约束 |
| **Kernel explosion** | comptime/blueprint 组合过多 → JIT 产物指数增长，缓存与编译时间失控 |

### 并行与拓扑

| 术语 | 简要说明 |
|------|----------|
| **Vector（轴）** | 指令级 SIMD；launch 时指定 vectorization factor |
| **Plane** | Warp/subgroup/simdgroup——锁步调度的协作单元 |
| **Cube / CubeDim** | 线程块（CUDA block / WebGPU workgroup） |
| **CubeCount** | grid 规模，启动多少个 cube |
| **`ABSOLUTE_POS`** | CubeCL 合成的全局元素索引，后端组合计算 |

### 编译链

| 术语 | 简要说明 |
|------|----------|
| **IR / Scope** | `Scope` 是单次 expand 收集指令的容器，`Operation` 是具体指令 |
| **JIT** | 首次 launch 某组参数组合时才编译 kernel；结果磁盘缓存 |
| **CFG** | Control Flow Graph，控制流图 |
| **SSA** | Static Single Assignment，每变量单次定义，合并点用 φ 节点 |
| **PTX** | NVIDIA 中间汇编，驱动进一步 JIT 为 SASS |
| **NVRTC** | NVIDIA 运行时 CUDA C++ 编译器 |

### 生态与缩写

| 缩写 | 全称 |
|------|------|
| GVN | Global Value Numbering |
| CSE | Common Subexpression Elimination |
| WGSL | WebGPU Shading Language |
| WMMA | Warp Matrix Multiply Accumulate |
| AOT | Ahead-of-Time（对比 JIT，见 ONNX 篇） |

*Burn 底层机制系列 · GPU 地图 · 导航见 [README](../../README.md)*

系列内相关：**[CubeK 地图](../cubek/summary.md)**（Blueprint-Routine-Autotuner 三层纪律） · **[Burn 地图](../burn/summary.md)**（类型栈 + 融合流） · **[跨项目架构](../architecture.md)**（决策推迟主线）
