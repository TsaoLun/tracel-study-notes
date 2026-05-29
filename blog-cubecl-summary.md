# CubeCL：`#[cube]` 如何把 Rust 函数变成六平台的 GPU 代码

## 读前须知

- **CubeCL 是什么**：Tracel 开源的 Rust **GPU 计算库 + 编译器框架**——语法仍是 Rust，外加 `#[cube]` 等 proc-macro 与前端类型（`Vector`、`ABSOLUTE_POS` 等），**不是**一门独立语言。你用这些扩展描述 kernel，在**首次 launch** 时 JIT 到 CUDA、ROCm/HIP、WebGPU/Metal/Vulkan、本机 CPU SIMD 等路径——而不是为每个平台各写一份 `.cu` / `.wgsl` / `.metal` 并做 FFI。
- **解决什么问题**：高性能算子若按平台手写，往往要维护多套 API、多套测试；换硬件或换激活函数（例如从 GELU 改成别的逐元素算子）要改多处。CubeCL 把「写一份 Rust 逻辑 + 运行时选后端与参数」当作默认路径。
- **本文示例 GELU**：**GELU**（Gaussian Error Linear Unit，高斯误差线性单元）是深度学习里常用的**激活函数**，对张量**逐元素**计算（实现里会用到 `erf` 等）。下文用官方示例 `cubecl/examples/gelu/` 演示；**`gelu_array`** 只是该示例里 kernel 的函数名，不是 Burn 里的模块名。
- **两种读法**：① **纵向**——作为 Burn 底层机制系列的 GPU 篇，接在 [Burn 综合地图（类型栈 + 融合流）](blog-burn-summary.md) 和 [ONNX AOT 编译器](blog-burn-onnx-summary.md) 之后；② **横向**——**只关心 CubeCL、不读 Burn** 时，把本文当**机制地图**查阅即可，**跟练路径**见 **[CubeCL 专题计划](blog-cubecl-plan.md)**（建议从 **[第一章 · GELU launch](blog-cubecl-1.md)** 跑通示例；需 clone `cubecl/` 到本目录）。
- **机制基准**：JIT pass 列表以 **cubecl** 仓库 `cubecl-opt/src/lib.rs` 为准；`TileKind` 以 **cubek** `crates/cubek-std/src/tile/base.rs` 为准（需 clone `cubek/`）。
- **术语**：*kernel*（设备上并行执行的计算）、*JIT*（首次 launch 才编译，非 `cargo build`）、*IR*（中间表示）、*expand*（JIT 时往 IR 填指令的生成器）等，下文**首次出现会括号简注**；完整释义见文末 **[词汇说明表](#词汇说明表)**。**不必先会 CUDA。**

### 三篇分工（与 Burn / ONNX 专文对齐）

| 层次 | 文档 | 本文覆盖 |
|------|------|---------|
| 构建期 | [ONNX 篇](blog-burn-onnx-summary.md) | —（生成 Rust 后进入 Burn 栈） |
| 编译期 | [Burn 地图](blog-burn-summary.md) | — |
| 运行期调度 | Burn 地图 §五 | Fusion drain 后才到本篇 |
| **GPU 代码生成** | **本文** | expand → SSA → NVRTC；autotune |

---

## 核心结论（读正文前的 spoiler）

> `#[cube]` **过程宏（proc-macro）** 保留你的 Rust 函数供类型检查，并生成以函数名命名的子模块——其中的 **`expand`** 在首次 launch、经 runtime **`define()`** 调用时，向 **IR** 的 **`Scope`** 里填入 **`Operation`** 指令。你写的源码不直接编译为 GPU 机器码——它是一段在 **JIT** 时运行的 **IR 生成器**；再由 `cubecl-opt` 与各后端（如 **NVRTC**→**PTX**）编译。这是 CubeCL 与「Rust 版 CUDA 源码」的关键区别。

---

## 动机：一份逻辑，为什么要六条编译路径

若不抽象，面向 NVIDIA / AMD / 浏览器 / Apple 往往意味着：各写一套 kernel 源码（CUDA、HIP、WGSL、MSL…），各配一套测试与调参。算法或算子一旦改动（例如把逐元素 **GELU** 换成另一种激活，或把 FFN 里的 **SwiGLU**——一种在 LLM 里常见的门控激活结构——换成别的实现），通常要在多套代码里同步修改。

CubeCL 的赌注是：**同一份 `#[cube]` Rust 逻辑**，在 launch 时由框架填 IR、选后端、缓存 JIT 产物。下文用 **GELU** 示例说明「改 Cargo feature 即可换 runtime，kernel 源码不变」。

同一段 GELU kernel，改一行 feature flag，即可切换 runtime（`examples/gelu` 内置 cpu / cuda / wgpu；同一源码也可经 `cubecl-hip` 或 wgpu 走到 HIP / Metal / Vulkan / WebGPU）：

```bash
# 在 cubecl 仓库根目录；每次只选一个 feature
cargo run --example gelu --features cpu   # MLIR → LLVM JIT → 本机 SIMD
cargo run --example gelu --features cuda  # NVRTC（NVIDIA 运行时编译器）→ PTX（GPU 中间码）
cargo run --example gelu --features wgpu  # WGSL 等（Metal / Vulkan 由 wgpu 选后端）
```

内核源码一字不改。`examples/gelu/src/lib.rs` 的核心逻辑如下（launch 辅助代码在同文件，此处略）：

```rust
#[cube(launch_unchecked)]
fn gelu_array<F: Float, N: Size>(input: &[Vector<F, N>], output: &mut [Vector<F, N>]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar(input[ABSOLUTE_POS]);
    }
}

#[cube]
fn gelu_scalar<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let sqrt2 = F::new(comptime!(2.0f32.sqrt())); // expand 执行时求值，写入 IR 常量
    let tmp = x / Vector::new(sqrt2);
    x * (Vector::erf(tmp) + Vector::one()) / Vector::new(F::new(2.0f32))
}
```

**没有手写 `.cu`，没有维护多份 `.wgsl` / `.metal` / `.hip`，没有 FFI 绑定。** CubeCL 做的就是：用 Rust 描述计算，在首次 launch 时生成目标平台的可执行代码，并缓存。

CubeCL 官方支持 **六条编译路径**（见 [README Supported platforms](https://github.com/tracel-ai/cubecl#supported-platforms)）：

| 平台 | Runtime | 编译器后端 | 典型硬件 |
|------|---------|-----------|---------|
| CUDA | `cubecl-cuda` | C++ (CUDA) → NVRTC | NVIDIA GPU |
| ROCm | `cubecl-hip` | C++ (HIP) → hipRTC | AMD GPU |
| Metal | `cubecl-wgpu` | C++ (Metal) | Apple GPU |
| Vulkan | `cubecl-wgpu` | SPIR-V | Linux / Windows GPU |
| WebGPU | `cubecl-wgpu` | WGSL | 浏览器与原生 wgpu |
| CPU | `cubecl-cpu` | MLIR → LLVM JIT | 本机 SIMD |

**与 Burn 系列的关系**：[Burn 综合地图](blog-burn-summary.md) 讲编译期类型栈与运行时融合流（v0.21.0 channel 重构，高并发下约 8.2×，[见该文 §五脚注](blog-burn-summary.md#融合反而更慢之后)）；[ONNX AOT 编译器](blog-burn-onnx-summary.md) 讲构建期模型导入；本文从 CubeCL 内部写透 proc-macro、SSA 与 autotune，与前两篇互补。

| 你想… | 建议 |
|--------|------|
| 理解 Burn 全栈 | [Burn 综合地图](blog-burn-summary.md) → [ONNX 篇](blog-burn-onnx-summary.md) → **本文** |
| 只学 CubeCL、能跑示例 | [专题计划](blog-cubecl-plan.md) → [专题第一章](blog-cubecl-1.md) → 需要全貌时回到本文 |
| 手写 reduction 练 kernel | 并行读 [cubecl-book](cubecl/cubecl-book/src/SUMMARY.md)（需 clone `cubecl/`） |

---

## 心智模型：你写的是 IR 生成器，不直接编译为 GPU 代码

传统 GPU 编程的流程：

```
CUDA C++ 源码 → nvcc 前端 → NVVM IR → PTX
WGSL 源码 → wgpu-native 前端 → SPIR-V → 驱动编译
```

**人写源码，编译器生成 GPU 代码。**

> **对比**：CUDA 是「人写 `.cu` → 编译器出 PTX」；CubeCL 是「人写 Rust → **expand 在 launch 时**往中间表示（IR）里填指令 → 再交给各平台编译器」。源码不等于最终 GPU 指令。

CubeCL 的流程多了一步：

```
Rust 函数 + #[cube] → proc-macro 生成 expand 模块
    → 首次 launch 时（经 runtime define()）调用 expand，向 Scope（IR 指令容器）里填 Operation
    → cubecl-opt 优化（CFG 控制流图 → SSA → pass 循环）
    → 编译器后端生成目标代码（CUDA C++ / WGSL / SPIR-V / MLIR …）
```

**人写的是一个在 JIT 编译时被调用、向 IR 里填指令的生成器。GPU 代码是在 JIT 时才产生的。**

这不是修辞。`#[cube(launch)]` 对函数做了**两层生成**（见 `cubecl-macros/src/generate/launch.rs`）：

1. **原始函数**：保留在 crate 中，走 Rust 类型检查与 borrow 检查——保证 kernel 写法本身合法、可组合。
2. **Expand 模块**（如 `gelu_array::expand`）：proc-macro 以函数名创建子模块，内部函数固定命名为 `expand`。语义等价于原始逻辑，但不做数值计算——它把 `+` 翻译为 `Operation::Arithmetic(Add, …)`，把 `if` 翻译为 `Operation::Branch(…)`，把 `ABSOLUTE_POS` 翻译为 `Variable::Builtin(AbsolutePos)`，全部填入 `cubecl_ir::Scope`。

> **Expand**：proc-macro 生成的同名子模块里的函数，名字固定为 `expand`；它不执行浮点运算，只把 `+`、`*`、`if` 等翻译成 IR 里的 `Operation`。

对 **trait 方法**，命名规则不同：`cubecl-macros` 的 `parse/cube_trait.rs` 将方法重命名为 `__expand_{method}`，以便在 trait 对象场景下区分。

`comptime! { … }` 宏把块内代码当作**普通 Rust**保留（见 `cubecl-macros` 的 `comptime` proc-macro）。在 `expand` 被调用、向 Scope 填 IR 时，这些块作为 Rust 代码执行，结果烘焙进 IR 常量——**不是在 `cargo build` 你的 crate 时固定**，也与 Rust `const` 不同。

测试 kernel 的常见路径是：用 **CPU runtime launch**（如 gelu 示例的 `--features cpu`），或用 `#[cube(create_dummy_kernel)]` 生成 IR 而不真正 launch——而不是把 `#[cube]` 函数当作普通 host 函数直接调用（函数体使用的是 CubeCL 前端类型 `Vector`、`ABSOLUTE_POS` 等）。

这解释了 CubeCL 与 "Rust 版 CUDA" 的差异：你的 Rust 代码**不直接被编译为 GPU 指令**。你的 Rust 代码是一段**在 JIT 编译时运行的逻辑**，输出是 IR，IR 再被编译为 GPU 指令。你拥有这个编译过程。

---

## 为什么需要这套机制：一次逻辑，六种硬件

做完上面的心智模型转换，CubeCL 的设计动机才说得通。

高性能计算的残酷等式：

```
最优性能 ≈ 不同硬件用不同 API × 不同形状用不同参数
```

不解决这个等式，你就需要：

- 为 NVIDIA 写一份 CUDA（WMMA tensor core intrinsic）
- 为 AMD 写一份 HIP（ROCm matrix intrinsic）
- 为浏览器写一份 WGSL（subgroup 操作语法不同）
- 为 Apple 写一份 MSL（simdgroup 操作语法又不同）

四份代码，四份测试矩阵。任何算法改动（例如把逐元素 GELU 换成另一种激活，或调整 LLM 里常用的 SwiGLU 门控结构）都往往要同步多套实现。

CubeCL 的方案是：**既然 expand 函数只是一个"向 Scope 里填指令"的过程，那用来填什么指令，可以在填的时候根据目标平台的能力决定——这个决策发生在 JIT 编译时，而非手写 kernel 时。**

这引出了 CubeCL 的两个核心机制：**comptime**（在 JIT 时决定生成什么结构）和 **autotune**（在首次执行时决定用哪个实现变体）。两者分工不同，但服务于同一个目标：把"针对平台的决策"从"写代码时"推迟到"JIT 编译时"或"首次执行时"。

---

## Comptime：推迟到 JIT 编译时才做的决策

`#[comptime]` 参数标记"这个值参与 kernel 特化，但**不作为 GPU 运行时参数传入**"。它与 Rust 的 `const` 不同：**Rust const 在编译你的 crate 时固定；comptime 在首次为某组参数 JIT 这个 kernel 时才固定。**

> **记忆法**：`const` = 编你的 crate 时就定死；`comptime` = **第一次为某组参数 JIT 这个 kernel** 时在 host 上算定，结果写进 IR 常量，GPU 上不再分支。

GELU 里的 `comptime!(2.0f32.sqrt())` 在 expand 执行、填 IR 时直接变成常量——GPU 每次 launch 不必算 `√2`。

更关键的是**结构级决策**。`cubecl-book` 的 `sum_plane` 模式展示了这个能力：

```rust
#[cube(launch)]
fn sum_plane<F: Float>(
    input: &[F], output: &mut [F],
    #[comptime] plane: bool,
    #[comptime] end: Option<u32>,
) {
    if plane {
        output[UNIT_POS] = plane_sum(input[UNIT_POS]);
    } else {
        sum_basic(input, output, end);
    }
}
```

> **Plane**：可理解为 NVIDIA 的 *warp*、WebGPU 的 *subgroup*、Apple 的 *simdgroup*——同一批线程步调一致、可做 shuffle/reduce 的协作单元。`plane: true/false` 会生成**两份不同的**编译产物，GPU 上不会出现「运行时 if 选路径」。

**这个 `if` 被 comptime 消解在 JIT 编译前——GPU 代码中不存在这个分支。** `plane: true` 和 `plane: false` 会 JIT 出**两个不同的 kernel**——一个包含 `plane_sum` 指令，一个走标量求和路径。这避免了"在不支持 subgroup 的硬件上编译包含 subgroup 指令的 kernel"——而传统方案要么需要 `#ifdef`，要么会在 JIT 编译阶段报错（且错误信息不直观）。

循环展开也是 comptime 控制的：

```rust
#[unroll(unroll)]
for i in 0..end { sum += input[i]; }
```

`end` 必须是 comptime 可确定的——运行时变量不能用于展开循环——这是刻意的：**不允许在运行时偷偷决定"展不展开循环"**，防止性能悬崖。

CubeK（独立仓库 [tracel-ai/cubek](https://github.com/tracel-ai/cubek)）的 `GemmBlueprint` 是整个生态中 comptime 用法的最高形态。`cubek-matmul` 的入口 kernel：

```rust
#[cube(launch_unchecked)]
pub fn matmul_entry<F: Float>(
    lhs: &Tensor<Vector<F>>, rhs: &Tensor<Vector<F>>, out: &mut Tensor<Vector<F>>,
    #[comptime] blueprint: GemmBlueprint,
) { ... }
```

`GemmBlueprint` 决定：dot product vs outer product 算法、stage 大小、是否 double-buffer、是否用 plane 级协作——每种组合对应**一份独立的 JIT 产物**。这解释了 CubeK 的 [`GUIDE.md`](https://github.com/tracel-ai/cubek/blob/main/GUIDE.md) 为何要求 blueprint 保持**极小**：防止 kernel explosion——comptime 参数每多一个取值，编译产物就翻一倍，磁盘缓存和编译时间可能失控。

> 上文 `matmul_entry` 签名已大幅简化。真实实现见 cubek 仓库 `crates/cubek-matmul/src/components/batch/gemm/matmul/base.rs`：含 `MatmulArgs`、`#[define]`、`cube_mapping`、`address_type = "dynamic"` 等。

---

## 自动向量化：launch 时注入，kernel 内写统一类型

高性能 kernel 应尽量使用 SIMD / subgroup 指令，但手写 `float4` 会让同一份逻辑复制多份。CubeCL 的做法是：**在 `launch` 时传入 vectorization factor**（如 GELU 示例里的 `vector_size: 4`），kernel 内写 `Vector<F, N>`——编译器据此生成合适宽度的 load/store，标量与向量混用时自动广播。

这与 comptime 是**不同维度**的特化：

| 维度 | 谁决定 | 何时固定 | 影响 |
|------|--------|----------|------|
| **Vectorization** | `launch` 参数（进入 JIT key） | 首次 JIT 该 vector 宽度 | 同一份 IR 生成不同宽度的 load/store |
| **Comptime** | `#[comptime]` 参数 | 首次 JIT 该 blueprint | 控制流、循环展开、plane vs scalar |
| **Autotune** | benchmark | 首次执行该 shape key | 在已编译候选中选最快 launch |

若算法依赖 vector 宽度（例如 reduction 边界），可在 kernel 内用 `input.vector_size()`——通过 comptime 读取，**零 GPU 运行时分支**。Autotune 则在多种 `(blueprint, tile, vectorization)` 组合里选赢家，而不是为每个具体元素个数各写一份 `.cu`。

---

## 四轴并行与统一拓扑

> **四轴**：从「一条指令算几个数」到「启动多少块」的四个独立旋钮；好 kernel 用 `PLANE_DIM` 等**运行时可读**的拓扑，而不是写死 `32`。

CubeCL 用 **四条正交轴**描述硬件并行（见 cubecl README），好的 kernel 在 comptime 读取这些值并自适应，而不是硬编码 `warpSize == 32`：

| 轴 | 含义 | 谁配置 | 映射示例 |
|----|------|--------|---------|
| **Vector** | 指令级 SIMD，一个 unit 一次处理 N 个 lane | launch 时指定 vectorization | AVX-512 lane、packed GPU load |
| **Plane** | lockstep 协作单元（warp / subgroup / simdgroup） | **runtime 按硬件决定** | CUDA warp、WebGPU subgroup |
| **CubeDim** | 一个 cube 内的并发 unit 数，共享内存与同步 | launch 时指定 | CUDA block、WebGPU workgroup |
| **CubeCount** | 启动多少个 cube | launch 时指定 | CUDA grid；CPU 上顺序调度 |

在此之上，CubeCL 用 **Cube**（线程块/workgroup）、**Unit**（线程/invocation）、**Plane**（warp/subgroup/simdgroup）命名执行层次。这些名字故意不与任何平台对齐——它们是 CubeCL IR 自己的抽象：

| CubeCL | CUDA | WebGPU | Metal |
|--------|------|--------|-------|
| `CUBE_POS_X` | `blockIdx.x` | `workgroup_id.x` | `threadgroup_position_in_grid.x` |
| `UNIT_POS_X` | `threadIdx.x` | `local_invocation_id.x` | `thread_position_in_threadgroup.x` |
| `ABSOLUTE_POS_X` | 由后端合成* | `global_id.x` | `thread_position_in_grid.x` |
| `PLANE_DIM` | `warpSize` | `subgroup_size` | `threads_per_simdgroup` |

\* **`ABSOLUTE_POS` 是 CubeCL 的合成抽象**，在 IR 里是 `Builtin::AbsolutePos`（`cubecl-core/src/frontend/topology.rs`）。CUDA 等后端并不直接映射到单一 intrinsic——`cubecl-cpp` 在 kernel 入口用 `AbsolutePosX/Y/Z` 与 cube 维度组合算出轴无关的全局索引（见 `crates/cubecl-cpp/src/shared/kernel.rs`）。这正是 CubeCL 相对原生 CUDA 的便利：写 elementwise kernel 你只关心"我是第几个元素"，不必手写 `blockIdx.x * blockDim.x + threadIdx.x`。

> **Elementwise 友好**：写 `output[ABSOLUTE_POS] = …` 时，你只需关心「我是全局第几个元素」，不必手写 `blockIdx * blockDim + threadIdx`。

Plane 抽象是 CubeK 的 Stage/Partition tile 系统的基础——矩阵乘法里的 warp-level 协作（`plane_sum`、`broadcast`）建立在这套统一拓扑上，不管底层是 NVIDIA warp 还是 Apple simdgroup。

---

## JIT 管线：post-SSA 定点循环（10 pass）+ 一次性重优化

当 `launch` 第一次遇到某个 `(kernel, comptime 参数, vectorization, cube_dim, …)` 组合时，`{fn_name}::expand` 被调用，向 `Scope` 填入 IR 指令。然后进入 `cubecl-opt` 优化器。

`Function::run_opt()`（`crates/cubecl-opt/src/lib.rs`，约 line 512）展示了完整流程：

> **SSA**：每条变量只赋值一次，合并点用 φ（phi）选「来自哪条前驱」的值，便于做死代码消除、公共子表达式消除等优化。**CFG** 即控制流图（基本块 + 分支）。

```
parse_graph(state, scope)           → 递归解析 Scope 为 ControlFlowGraph
split_critical_edges()              → 拆分关键边，准备 SSA
    ↓
transform_ssa_and_merge_composites  → SSA 变换（含 pre-SSA 的 InlineRef 等 pointer 处理）
    ├── ssa_transform: place_phi_nodes + version_program
    └── CompositeMerge 固定点循环：合并复合操作后重新 SSA
    ↓
analysis::<PointerSource>           → 指针源分析
    ↓
apply_post_ssa_passes(state)        → 定点循环执行 10 个 pass：
    ├── InlineAssignments           → 内联平凡赋值
    ├── EliminateUnusedVariables    → 死变量消除
    ├── ConstOperandSimplify        → 常量表达式简化
    ├── MergeSameExpressions        → 公共子表达式消除
    ├── ConstEval                   → 常量求值
    ├── RemoveIndexScalar           → 索引标量化
    ├── EliminateConstBranches      → 常量分支消除
    ├── EmptyBranchToSelect         → 平凡分支转 select
    ├── EliminateDeadBlocks         → 死代码块消除
    └── EliminateDeadPhi            → 死 φ 节点消除
    ↓ （循环直到无变化）
一次性重优化：
    ├── DisaggregateArray           → 复合数组分解为标量（如有变化，重新 SSA + post-ssa 循环）
    ├── GvnPass                     → 全局值编号
    ├── ReduceStrength              → 强度削减
    └── CopyTransform               → 复制传播
    ↓ （如有变化，重新 post-ssa 循环）
split_free()                        → 拆分 free 操作
analysis::<SharedLiveness>          → 共享内存活性分析（分配共享内存偏移）
MergeBlocks                         → 合并相邻基本块
Captures analysis                   → 确定隐式/显式参数
update_buffer_vis                   → 标记每个 buffer 的读写可见性
```

> 上述 pass 名可对照文末「编译优化」词条；不必逐 pass 背，知道「IR 在进 CUDA/WGSL 前会被多轮简化」即可。

输出是一张 `petgraph::StableDiGraph<BasicBlock>`——每个 `BasicBlock` 包含一组 `Operation`，由 `ControlFlow` 终结。后端编译器遍历这张图，为 phi 节点生成对应代码（在非 SSA 目标语言中，phi 被模拟为在相应前驱块末尾赋值给一个可变变量）。

各 runtime 的编译路径：

| 平台 | Runtime | 编译器 | 编译产物 |
|------|---------|--------|----------|
| CUDA | `cubecl-cuda` | `CppCompiler<CudaDialect>` → CUDA C++（含 WMMA intrinsic） | NVRTC → PTX |
| ROCm | `cubecl-hip` | `CppCompiler<HipDialect>` → HIP C++ | hipRTC → AMD GPU 码 |
| WGPU | `cubecl-wgpu` | `WgslCompiler` / `CppCompiler<MslDialect>` / SPIR-V 编译器 | wgpu 驱动加载 |
| CPU | `cubecl-cpu` | `MlirCompiler` → MLIR → LLVM JIT | 本机 SIMD 指令 |

**JIT 编译产物**通过 `KernelId::stable_hash()` 作为缓存键，持久化到 `CompilationCache`（`cubecl-common/src/compilation_cache.rs`；各 runtime 如 `cubecl-cuda/src/compute/context.rs` 在 miss 时编译并写入）。**同一个 `(kernel id, comptime 参数, vectorization, cube_dim, …)` 组合的第二次 launch 直接命中缓存——不再编译。**

（Autotune 的 tunable 集合校验和另用 MD5，见 `cubecl-runtime/src/tune/operation.rs::compute_checksum`——与 kernel JIT 缓存是两套机制。）

---

## Autotune：同一 blueprint 下，13 种 tile 实现候选

Comptime 决定"生成什么结构的 kernel"。Autotune 决定"用哪个已编译变体在这块 GPU、这个形状上最快"。

> **Autotune**：同一逻辑、多种实现（如 tensor core vs 纯寄存器 tile），在**真实 GPU + 真实 shape** 上跑小 benchmark，把最快方案的索引缓存起来；与 comptime「改结构」不同，autotune「在已编译候选里选谁 launch」。

CubeK matmul 的 tile 系统有 **13 种 `TileKind` 变体**（另有 `None` 作零初始化哨兵，不参与 autotune），定义于 cubek 仓库 `crates/cubek-std/src/tile/base.rs`（**以你 clone 的 cubek 版本为准**）：

```
Stage → Partition → SharedTile → Cmma (tensor core)
  → Mma → Register → PlaneVec → Interleaved
  → Unit → WhiteboxFragment → RowWise → Pipelined → Bounce
```

`CmmaTile` 走 NVIDIA tensor core WMMA intrinsic——只在有 tensor core 的 GPU 上可用且最快（**WMMA / tensor core**：NVIDIA 等 GPU 上专用于矩阵乘的硬件单元）。`RegisterTile` 是纯软件 tile——任何硬件都能跑，但慢（**Register tile**：不用 tensor core，用寄存器里小块数据手工累加，通用但通常更慢）。`PlaneVecTile` 用 warp-level 向量化，在无 tensor core 的 GPU 上通常是最好的选择。`BounceTile` 让 tensor core 输出经过共享内存往返再广播——为 Flash Attention 的 online softmax 铺路。

`cubecl-runtime` 的 `Tuner<K: AutotuneKey>` 在**首次遇到一个新 `(M, N, K, dtype, layout)` 时**，用候选 kernel 在真实 buffer 上 benchmark。`TuneGroup` 优先级系统让高分组的候选先试——`CmmaMatmul` 优先级高于 `RegisterMatmul`，如果 tensor core 版本可用且正确，不会浪费时间 benchmark 软件实现。应用侧通过 `local_tuner!()` 宏创建 **`LocalTuner` 静态实例**（按 device id 懒初始化 `Tuner`），结果可持久化到磁盘 autotune 缓存（方便部署时预填）。

`#[derive(AutotuneKey)]` 的 `#[autotune(anchor(...))]` 属性做指数分桶——`exp(min=16, max=1024, base=2)` 把 16–1024 按 2 的幂分 7 个桶，避免每个具体尺寸都触发一次完整的 benchmark 流程。

---

## Comptime × Autotune 的分界线

这是 CubeCL 设计中最容易混淆的一点。两者的分工：

| 维度 | Comptime / Blueprint | Autotune |
|------|----------------------|----------|
| 决策时机 | JIT 编译时 | 首次执行时 |
| 决策内容 | kernel **结构**（循环展开、plane vs scalar、stage 大小） | kernel **实现变体**（CMMA vs Register、单缓冲 vs 双缓冲） |
| 缓存粒度 | 每个 blueprint 组合一份编译产物 | 每个 autotune key 缓存最快变体索引 |
| 爆炸风险 | blueprint 参数每多一个取值，JIT 产物翻倍 | autotune key 通过指数分桶控制粒度 |

Comptime 改变的是**编译出来的 PTX/WGSL 代码本身**。Autotune 改变的是**选哪个已编译 kernel 来 launch**。前者出错 = JIT 编译失败（平台不支持你请求的指令），后者出错 = 跑了次优的 kernel。

> **失败形态**：comptime 选了平台不支持的指令 → **JIT 编译失败**（明确报错）；autotune 选错 → 仍能跑，只是**慢**（次优 kernel）。

---

## CubeK 的纪律：防止 kernel explosion

CubeK（[tracel-ai/cubek](https://github.com/tracel-ai/cubek)，与 cubecl 分仓维护）在 CubeCL 上提供成品内核——matmul、attention、convolution、reduce、quant、FFT。它的设计重心不在算法本身（tiled matmul 是成熟的已知模式），而在 **Blueprint-Routine-Autotuner 的三层纪律**（[`GUIDE.md`](https://github.com/tracel-ai/cubek/blob/main/GUIDE.md)）：

| 层 | 职责 | 约束 |
|----|------|------|
| **Blueprint** | JIT 特化参数 | 只放会改变控制流或指令选择的参数。排除 vectorization（已在 JIT key 中）、cube dim（已在编译 key 中）、硬件属性（运行时可得）、问题尺寸（运行时参数） |
| **Routine** | 每次 launch 计算 `LaunchSettings` | 不做硬件相关的硬决策——从 launch 层接收 vectorization 等约束，据此计算 `cube_dim`、`cube_count`、stride 对齐策略 |
| **Autotuner** | 首次遇 key 时 benchmark | 找到最快 Routine + Blueprint 组合，缓存结果 |

这三层纪律的真实目的在 `GUIDE.md` 里写得很直白：**防止 kernel explosion。** 例如 3×3×3 的 blueprint 参数空间 = 27 份 JIT 产物；再 ×13 种 `TileKind` autotune ≈ **351 个候选（示意算术，非精确上界）**。若不严格控制哪些参数进 blueprint（编译维度）vs autotune（运行维度），组合爆炸会让 JIT 缓存大到不可接受。

---

## CubeCL 与 Burn 的边界

CubeCL 是**独立的多平台 GPU 计算库**，与 Burn 分仓维护，可以被 Burn 消费，也可以独立使用。Burn 通过 `burn-cubecl` crate（及 `burn-cuda`、`burn-wgpu` 等后端包装）调用 CubeK 内核。

```
用户模型/应用
    ↓
Burn（Autodiff + Fusion + Backend trait）     ← [Burn 综合地图](blog-burn-summary.md)
Burn ONNX（ONNX → Rust AOT）                   ← [ONNX 篇](blog-burn-onnx-summary.md)
burn-cubecl → CubeK 内核（cubek 仓库）
    ↓
CubeCL（#[cube] + IR + JIT + autotune）        ← 本篇
    ↓
CUDA / HIP / WGPU / CPU …
```

[Burn 综合地图](blog-burn-summary.md) 从 Burn 视角讲 fusion 调度与 channel 重构；本篇从 CubeCL 内部展开 CFG → SSA 的优化 pass、Blueprint-Routine-Autotuner 纪律与 13 种 `TileKind`。

**直接用 CubeCL 的场景**（不引入 `burn`）：你有自定义计算需求（稀疏 attention、科学模拟、特殊采样），需要跨平台可移植且不想维护多个 `.cu` / `.wgsl` 文件。

**用 Burn + CubeK 的场景**：你在做端到端深度学习——需要 `Backend` trait 的生态（ONNX 导入、训练管线、融合引擎、检查点保存），CubeK 作为后端 kernel 提供方。

---

## 诚实的局限

CubeCL README 标注 **alpha**。两层含义：

1. **并非所有平台支持相同特性**。WebGPU 上尚无 tensor core 路径。使用平台不支持的指令会在 JIT 编译时**明确失败**（不是静默降级）。
2. **冷启动有成本**。首次 launch 触发 JIT 编译 + autotune benchmark，用磁盘缓存和预填 autotune 数据可以缓解——但首次 launch 比热路径慢，这是 JIT 模型的固有特性。

与成熟 CUDA 生态的差距不在"能不能做"（同功能 matmul 已经做到），而在生态积累深度：cuDNN 等库经多年硬件代际打磨。CubeCL 的算子库（CubeK）和 autotune 覆盖范围正在快速增长，但不可能在短期内完全对齐这种历史积累。

---

## 一次 launch 的完整旅程（GELU，CUDA）

把前文收束为 `cargo run --example gelu --features cuda` 的真实顺序：

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

换 `--features wgpu`，中间换成 `WgslCompiler`（或 Metal/Vulkan 对应路径），无 NVRTC。**`#[cube]` 函数体与 GELU 的 Rust 源码不变。** 这与 [Burn 综合地图](blog-burn-summary.md) 调用链末尾的 `NVRTC → PTX` 是同一机制，本篇补全了中间的编译器 pass。

---

## 系列导航

### Burn 底层机制系列

| 文档 | 主题 | 适合 |
|------|------|------|
| [blog-burn-summary.md](blog-burn-summary.md) | Burn 地图：类型栈 + 融合流 + ONNX 入口 | 理解 Burn 全栈 |
| [blog-burn-onnx-summary.md](blog-burn-onnx-summary.md) | ONNX→Rust AOT 编译器 | 深入 ONNX 导入 |
| **本文** | CubeCL 编译器框架地图 | 理解 GPU 代码生成 |

### CubeCL 专题（跟练，可不读 Burn）

| 篇 | 主题 | 文档 |
|:---:|------|------|
| 计划 | 入门引导 + 章节目录 | [blog-cubecl-plan.md](blog-cubecl-plan.md) |
| 1 | GELU 走通 launch | [blog-cubecl-1.md](blog-cubecl-1.md) |
| 2–8 | expand、trait、comptime、拓扑、JIT、autotune、CubeK/Burn | 见 [计划表](blog-cubecl-plan.md#章节目录) |

---

## 词汇说明表

> 正文前 50 行出现的词，在此可查完整释义。读正文时见括号简注即可，不必先背本表。

按主题分组；首次阅读可先看 **粗体** 词条，其余作查阅用。

### 核心概念（本篇最重要）

| 术语 | 简要说明 |
|------|----------|
| **CubeCL** | Tracel 的多平台 GPU 计算框架：Rust 语法 + `#[cube]` proc-macro 与前端类型，JIT 生成 CUDA / HIP / WGSL / MLIR 等目标代码；非独立语言。 |
| **`#[cube]`** | 过程宏：保留原函数供 Rust 类型检查，并生成 `{函数名}::expand`，在 launch 时向 IR 填指令。 |
| **Expand** | JIT 阶段在 host 上执行的「IR 生成器」；不直接做 GPU 浮点运算。 |
| **IR / Scope** | 中间表示；`Scope` 是单次 expand 收集指令的容器，`Operation` 是具体指令（算术、分支、内置变量等）。 |
| **JIT** | Just-In-Time：首次 launch 某组参数组合时才编译 kernel；结果可磁盘缓存。 |
| **Kernel** | 在设备上并行执行的计算函数；CubeCL 里由 expand 产出 IR，再经后端编译。 |
| **Launch** | 在指定 `cube_count`、`cube_dim`、vectorization 等下，把 kernel 提交给 runtime 执行。 |
| **Comptime** | `#[comptime]` 或 `comptime!`：在 **JIT 该 kernel 时**在 host 求值/分支，烘焙进 IR；不是 GPU 运行时参数。 |
| **Autotune** | 对多种已编译实现做 benchmark，按 `(shape, dtype, device…)` 缓存最快候选索引。 |
| **Blueprint** | CubeK 中描述 kernel **结构**的 comptime 配置（算法、stage、是否 double-buffer 等）。 |
| **Kernel explosion** | comptime/blueprint 组合过多 → JIT 产物数量指数增长，缓存与编译时间失控。 |
| **GELU** | Gaussian Error Linear Unit，常用激活函数；本文与 `examples/gelu` 仅作最小演示，非 CubeCL 内置专有名词。 |

### 平台与编译链

| 术语 | 简要说明 |
|------|----------|
| **CUDA** | NVIDIA GPU 编程栈；CubeCL 经 C++ 方言 → **NVRTC** → **PTX** → 驱动加载。 |
| **PTX** | NVIDIA 中间汇编；由驱动进一步 JIT 为具体 GPU 的 SASS。 |
| **NVRTC** | NVIDIA 运行时 CUDA C++ 编译器，把生成的 CUDA C++ 编成 PTX。 |
| **ROCm / HIP** | AMD GPU 生态；**hipRTC** 类似 NVRTC。 |
| **WGSL** | WebGPU 着色器语言；浏览器与 wgpu 原生路径可用。 |
| **SPIR-V** | Vulkan 常用的中间表示（二进制 IR）。 |
| **MSL** | Metal Shading Language，Apple GPU。 |
| **WGPU** | Rust 的 WebGPU 实现；CubeCL 经它覆盖 Metal / Vulkan / WebGPU。 |
| **MLIR → LLVM JIT** | CPU 路径：多级 IR lowering 到 LLVM，再 JIT 为本机 SIMD 指令。 |
| **Runtime** | 如 `cubecl-cuda`、`cubecl-wgpu`：负责编译、缓存、与设备通信。 |
| **Feature flag** | Cargo `features`（如 `cpu` / `cuda` / `wgpu`）切换链接的后端，源码可不变。 |

### 并行与拓扑（CubeCL 命名）

| 术语 | 简要说明 |
|------|----------|
| **Vector（轴）** | 指令级 SIMD：一次处理 N 个 lane；由 launch 的 vectorization 决定。 |
| **Plane** | Warp / subgroup / simdgroup：锁步调度的协作单元；`plane_sum` 等建立在此之上。 |
| **Cube / CubeDim** | 一个线程块（CUDA block / WebGPU workgroup）；块内共享内存与同步。 |
| **CubeCount** | 启动多少个 cube（CUDA grid 规模）。 |
| **Unit** | 块内一个执行线程（thread / invocation）。 |
| **`CUBE_POS` / `UNIT_POS`** | 块在 grid 中的索引 / 块内线程索引。 |
| **`ABSOLUTE_POS`** | CubeCL 合成的全局元素索引；后端用块维与位置组合计算。 |
| **`PLANE_DIM`** | 当前硬件 plane 大小（如 warp 32）；可用 comptime 读取，避免硬编码。 |

### 类型与特化

| 术语 | 简要说明 |
|------|----------|
| **`Vector<F, N>`** | CubeCL 前端向量类型；launch 注入 N，生成对应宽度的 load/store。 |
| **Vectorization factor** | Launch 时指定的向量宽度（如 4），进入 JIT 缓存键。 |
| **JIT key** | `(kernel id, comptime 参数, vectorization, cube_dim, …)` 唯一标识一份编译产物。 |
| **`launch_unchecked`** | 跳过部分 launch 安全检查的宏变体；示例与高性能路径常用。 |
| **`#[unroll]`** | 要求循环在 comptime 可知边界上展开，避免运行时决定是否展开。 |

### 编译优化（读 JIT 管线时用）

| 术语 | 简要说明 |
|------|----------|
| **CFG** | Control Flow Graph，控制流图（基本块 + 边）。 |
| **SSA** | Static Single Assignment：每变量单次定义，合并点用 φ 节点。 |
| **Phi (φ) 节点** | 多条前驱汇合时「选哪条前驱的值」的 SSA 占位；非 SSA 后端常模拟为赋值。 |
| **GVN** | Global Value Numbering，全局值编号，利于消除冗余计算。 |
| **CSE** | Common Subexpression Elimination，公共子表达式消除（文中 MergeSameExpressions）。 |
| **强度削减** | 用更便宜运算替代昂贵运算（如乘常数改移位）。 |
| **死代码消除** | 去掉不可达块、无用变量、无用 φ 等。 |
| **Liveness / 共享内存** | 分析哪些共享内存槽仍活跃，用于分配 offset。 |
| **petgraph** | Rust 图库；优化后的 IR 以 `StableDiGraph<BasicBlock>` 形式交给后端。 |

### 矩阵乘与 CubeK

| 术语 | 简要说明 |
|------|----------|
| **CubeK / cubek** | 基于 CubeCL 的成品算子库（matmul、attention 等），与 cubecl 分仓。 |
| **Tile / TileKind** | 分块矩阵乘的数据布局与实现策略（Stage、Partition、Cmma、Register…）。 |
| **WMMA / CMMA / MMA** | Warp/协作矩阵乘加速指令族；NVIDIA tensor core 常用 WMMA。 |
| **Tensor core** | NVIDIA GPU 上专用于矩阵/tensor 运算的硬件单元。 |
| **Stage / Partition** | 多级 tile：全局块 → 块内子块 → 寄存器/共享内存层次。 |
| **Double-buffer** | 流水线重叠：一块数据计算时预取下一块，隐藏内存延迟。 |
| **Flash Attention** | 分块 attention；`BounceTile` 等配合 online softmax。 |
| **Routine** | CubeK 每次 launch 算 `LaunchSettings`（grid、stride 等），不做硬件硬编码。 |
| **AutotuneKey / anchor** | 把连续 shape 指数分桶，避免每个尺寸都 full benchmark。 |

### 生态与边界

| 术语 | 简要说明 |
|------|----------|
| **Burn** | Rust 深度学习框架；通过 `burn-cubecl` 使用 CubeK/CubeCL。 |
| **Fusion** | Burn 运行时把多算子合并调度（[Burn 地图 §五](blog-burn-summary.md)）；与 CubeCL JIT 是不同层。 |
| **Backend trait** | Burn 抽象「张量落在哪、如何执行」的接口；类型即后端（系列第一篇）。 |
| **CompilationCache** | 按 `KernelId` 等键持久化 JIT 结果，第二次 launch 命中。 |
| **Alpha** | README 标注的成熟度：特性因平台而异，冷启动含 JIT + autotune 成本。 |

### 缩写速查

| 缩写 | 全称 / 含义 |
|------|-------------|
| IR | Intermediate Representation |
| JIT | Just-In-Time compilation |
| SSA | Static Single Assignment |
| CFG | Control Flow Graph |
| GVN | Global Value Numbering |
| CSE | Common Subexpression Elimination |
| PTX | Parallel Thread Execution（NVIDIA） |
| WGSL | WebGPU Shading Language |
| SPIR-V | Standard Portable Intermediate Representation-V |
| MSL | Metal Shading Language |
| HIP | Heterogeneous-compute Interface for Portability（AMD） |
| WMMA | Warp Matrix Multiply Accumulate |
| AOT | Ahead-of-Time（对比 JIT；系列 ONNX 篇） |
| FFI | Foreign Function Interface（手写绑定时常见） |

*Burn 底层机制系列 · GPU 地图 · [系列索引](README.md) · [Burn 地图](blog-burn-summary.md) · [ONNX 篇](blog-burn-onnx-summary.md)*
