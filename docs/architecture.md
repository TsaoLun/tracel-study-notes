# Tracel 生态的架构主线：决策推迟

> Burn 把后端选择推迟到编译期，CubeCL 把 GPU 指令生成推迟到首次 launch，CubeK 把最优实现推迟到首次 benchmark——三个项目共享一条设计哲学，各解决不同层次的问题。

## 读前须知

- **本文覆盖**：Burn、CubeCL、CubeK、Burn-ONNX 四个项目共享的设计哲学——将决策从"写代码时"推迟到"编译时 / JIT 时 / 首次执行时"。本文是跨项目的地图，解释四个项目如何通过同一原则解决不同层次的问题。
- **本文定位**：阅读路径的第一步。在深入具体系统之前建立跨项目坐标系——理解 Burn/CubeCL/CubeK/Burn-ONNX 共享的设计哲学（决策推迟）。不需要先读其他文章。读完后可以解释"为什么 Tracel 生态的组件可以自由组合而不互相冲突"。
- **读完后**：按 [阅读路径](../README.md) 继续全景篇和各系统文章。

---

## 核心主张

> Tracel 生态的四个项目——Burn（框架）、CubeCL（编译器）、CubeK（算子库）、Burn-ONNX（AOT 导入）——解决的是不同层次的问题，但它们共享一条设计主线：**将决策从代码作者手里推迟到更合适的时机**。Burn 把后端选择推迟到编译期单态化；CubeCL 把 GPU 指令生成推迟到首次 launch；CubeK 把 tile 策略选择推迟到首次执行时 benchmark；Burn-ONNX 把模型导入推迟到构建期 AOT 代码生成。

四个项目共享这条主线，是因为它们都构建在同一组基础设施上：Rust 的类型系统提供编译期计算能力，proc-macro 提供代码生成能力，CubeCL 的 JIT 编译管线提供运行时编译能力。三者叠加，整个生态可以在不牺牲性能的前提下保持组件间的正交性。

---

## 一、决策推迟的三个层次

Tracel 生态的决策分布在三个时间层次上。每个层次的推迟机制、成本和收益不同：

| 层次 | 时机 | 机制 | 推迟成本 | 代表 |
|------|------|------|----------|------|
| **L1：编译期** | `cargo build` | Rust 泛型单态化 + proc-macro | 编译时间增长（单次） | Burn 类型栈、Burn-ONNX AOT |
| **L2：JIT 编译时** | 首次 launch miss | expand → SSA → 后端 codegen | 首次 launch 延迟（一次性） | CubeCL JIT、comptime |
| **L3：首次执行时** | 首次遇 shape key | benchmark 候选实现 → 缓存最优 | 首次执行延迟（一次性） | CubeK autotune |

三层推迟的方向一致：**越靠近硬件，推迟得越远**。Burn 的决策在 L1（离硬件最远），CubeCL 在 L2，CubeK autotune 在 L3（直接与硬件对话）。这种分层意味着每一层都可以独立演进——CubeK 换一种 tile 策略不影响 Burn 的类型栈，Burn 换一种融合策略不影响 CubeCL 的 JIT 管线。

---

## 二、Burn：正交能力的编译期组合

PyTorch 的后端选择是运行时的：`device = torch.device("cuda:0")` → Dispatch Key 查表 → 找到对应的 kernel 实现。这个方案的灵活性代价是运行时开销（虚表查找）和能力耦合（autograd 和 CUDA 后端在 C++ 层紧耦合）。

Burn 把这个问题挪到编译期：

```rust
// 用户代码中不出现泛型——通过 DispatchDevice 枚举消解
let device = Device::wgpu(0);
let tensor = Tensor::<2>::from_data(data, &device);

// 但框架内部，类型栈在编译期完全展开：
// Autodiff<Fusion<CubeBackend<WgpuRuntime>>>
```

[详细分析见 Burn 地图](burn/summary.md) | [系统设计：Kernel Fusion](burn/kernel-fusion-system-design.md) | [系统设计：Autodiff](burn/autodiff-system-design.md)

**推迟了什么**：后端选择、autodiff 的启用/禁用、融合的启用/禁用——从运行时字符串匹配推迟到编译期 trait 单态化。

**代价**：编译时间。每个后端组合生成一份独立的机器码。但这是单次成本——运行时没有虚表，dispatch 在编译期消解完毕。

**这个代价在什么场景下不可接受**：后端极多且频繁切换时。Tracel 针对这个问题提供了 `DispatchDevice` 枚举——用户侧不需要泛型传染，只有框架内部的 match 需要展开为具体类型。

---

## 三、CubeCL：GPU 指令生成的 JIT 推迟

传统 GPU 编程：人写 `.cu` → nvcc → PTX。平台变了，重写 `.wgsl` 或 `.metal`。

CubeCL 的方案：人写一份 `#[cube]` Rust 函数 → proc-macro 生成 expand 模块 → 首次 launch 时调用 expand 填入 IR → cubecl-opt 优化 → 各后端生成目标代码。

[详细分析见 CubeCL 地图](cubecl/summary.md) | [系统设计：Autotune](../cubecl/autotune-system-design.md) | [系统设计：JIT 编译管线](../cubecl/jit-compilation-pipeline.md)

**推迟了什么**：GPU 指令的选择和优化——从"写代码时针对特定平台"推迟到"首次 launch 时由编译器决定"。

**comptime 的特殊角色**：`#[comptime]` 参数在 JIT 编译时固定，不在 GPU 运行时分支。这意味着：

```rust
#[cube(launch)]
fn sum_plane<F: Float>(input: &[F], output: &mut [F], #[comptime] plane: bool) {
    if plane {
        output[UNIT_POS] = plane_sum(input[UNIT_POS]);  // 使用 subgroup 指令
    } else {
        sum_basic(input, output, /* ... */);              // 不使用 subgroup
    }
}
```

`plane: true` 和 `plane: false` 生成**两份不同的 JIT 产物**。不支持 subgroup 的硬件上，编译器不会遇到包含 subgroup 指令的 kernel——因为 `plane: false` 那份产物不包含这些指令，硬件的支持检查在 kernel 选择阶段已完成。

**代价**：首次 launch 的冷启动延迟。JIT 编译 + 代码优化的时间在首次 launch 时支付。磁盘缓存缓解了后续 launch，但首次比热路径慢是 JIT 的固有特性。

---

## 四、CubeK：最快的实现由 benchmark 决定

CubeCL 的 comptime 决定了 kernel 的**结构**（哪些指令、什么控制流）。但同一结构下，可以有多种实现策略——NVIDIA tensor core (CMMA)、warp-level 向量化 (PlaneVec)、纯软件 tile (Register)。

CubeK 的 autotune 在首次遇到新 `(M, N, K, dtype, layout)` 组合时，benchmark 所有候选实现，缓存最快者的索引。

[详细分析见 CubeK 地图](cubek/summary.md)（待写）

**推迟了什么**：同一算法结构下的最优实现选择——从"写代码时凭经验选"推迟到"首次执行时让硬件 benchmark 决定"。

**与 comptime 的分界线**：

| 维度 | Comptime / Blueprint | Autotune |
|------|----------------------|----------|
| 决策时机 | JIT 编译时（L2） | 首次执行时（L3） |
| 决策内容 | kernel **结构**（循环展开、plane vs scalar） | kernel **实现变体**（CMMA vs PlaneVec） |
| 失败形态 | JIT 编译失败（平台不支持的指令） | 跑了次优 kernel（仍能跑，只是慢） |
| 爆炸风险 | blueprint 参数每多一个取值，JIT 产物翻倍 | 指数分桶（`#[autotune(anchor(...))]`）控制粒度 |

**Blueprint 纪律是 CubeK 的核心设计约束**：只放改变控制流或指令选择的参数。vectorization、cube dim、硬件属性、问题尺寸一律排除——它们已由 JIT key 或 runtime 参数覆盖。违反这条纪律的代价是 kernel explosion——3×3×3 blueprint 参数空间 × 13 种 TileKind ≈ 351 个组合，每个组合独立 JIT 编译。

---

## 五、Burn-ONNX：模型导入的构建期推迟

ONNX Runtime 在运行时加载模型、解析 protobuf、按图执行。Burn ONNX 把这个过程挪到构建期：

```
model.onnx → build.rs → model.rs + model.bpk → 普通 Burn 代码
```

[详细分析见 Burn ONNX 地图](burn/onnx-summary.md)

**推迟了什么**：模型导入——从"运行时解析 ONNX protobuf"推迟到"构建期生成 Rust 源码"。

**这个推迟使能了什么**：模式匹配。运行时 loader 只能按图执行——`MatMul → Scale → Softmax → MatMul` 五个 kernel launch。AOT 编译器可以识别 SDPA 分解模式，替换为单一 `Attention` 节点——生成代码调用 Burn 原生注意力，穿过融合流和 CubeK/CubeCL 内核。

**代价**：`build.rs` 的执行时间。模型越大、IR 流水线越复杂，构建越慢。但这是单次成本——运行时二进制不依赖 ONNX Runtime。

---

## 六、一次 matmul 穿过四层：完整的决策推迟之旅

抽象的"决策推迟"在具体操作中是什么样子？以 Burn 用户写 `tensor.matmul(&other)` 为例，追踪它依次经过的推迟层：

```
用户代码: tensor.matmul(&other)
    │
    ├─ [L1 编译期] Burn 类型栈
    │   rustc 单态化: Autodiff<Fusion<CubeBackend<CudaRuntime>>>
    │   → tensor 的类型在编译期固定，matmul dispatch 消解为函数指针
    │
    ├─ [L1 编译期] Autodiff 层
    │   Autodiff::float_matmul:
    │     1. 调用 inner.matmul() → 交给 Fusion 层
    │     2. 注册 MatmulBackward 到梯度图
    │   → 前向和反向图的注册是编译期决定的，但梯度图本身是运行时构造的
    │
    ├─ [运行时-惰性] Fusion 层
    │   FusionBackend::float_matmul:
    │     1. 将 OperationIr::Matmul { lhs, rhs, out } 入队到当前流的 OperationQueue
    │     2. 不执行 → 等到 drain 时才处理
    │   → 融合决策推迟到入队后：Explorer 扫描 OperationQueue 决定哪些操作可以合并
    │
    ├─ [运行时-drain] Fusion drain
    │   MultiStream::drain → Processor::process:
    │     1. Policy 决定 Explore/Execute/Defer
    │     2. Explorer 用 StreamOptimizer 注册 op 到 Block
    │     3. Block::optimize 用 find_best_optimization_index 选最佳 builder
    │     4. 若与前后操作可融合→ 生成 FuseTrace；否则 ExecutionStrategy::Operations
    │   → 多操作融合成 FuseTrace: Clone + ScalarMul + ScalarAdd + Tanh → elemwise_fuse
    │
    ├─ [运行时] CubeK 层
    │   Fusion drain 后的操作交给 CubeK:
    │     1. matmul 走到 cubek-matmul → 查当前 (M,N,K,dtype) 是否有 autotune 缓存
    │     2. 若缓存命中: 直接用最快 blueprint + tile 配置
    │     3. 若缓存缺失: autotune benchmark 候选 Strategy
    │   → Strategy::Auto 先试 SimpleCyclicCmma，硬件不支持则退到 SimpleUnit
    │
    ├─ [L2 JIT] CubeCL JIT 编译
    │   KernelLauncher::launch → CubeCL runtime:
    │     1. 计算 JIT key = (kernel, comptime, vectorization, cube_dim, ...)
    │     2. 缓存 miss → 调用 kernel.define() → expand 填入 Scope
    │     3. cubecl-opt: parse_graph → SSA → 定点循环(10 passes)
    │     4. CppCompiler<CudaDialect> → NVRTC → PTX → 写入 CompilationCache
    │   → matmul 的计算逻辑从 Rust #[cube] 变成 PTX 机器码
    │
    └─ [L3 首次执行] Autotune（若 matmul 未缓存）
        1. 对每个候选 TileKind (CMMA, PlaneVec, Register) → 各 benchmark 一次
        2. 缓存最快候选索引，后续 launch 直接用
        → autotune 回答的是"在这个特定 GPU 上，对 (M,N,K) 这个大小，
           是 CMMA tile 快还是 PlaneVec tile 快"
```

每一步的决策时机不同，但顺序一致：编译期 → 惰性入队 → drain 融合 → JIT 编译 → autotune。每一步只对下一步可见，不穿透多层——Fusion 不知道 CubeCL 的 SSA 定点循环在做什么，CubeCL 不知道 Burn 的梯度图怎么记录的。

### 数据流中的关键交接

| 层间切换 | 交接物 | 内部表示变化 |
|----------|--------|-------------|
| Burn → Autodiff | `FloatTensor<Autodiff<B>>` | `AutodiffTensor { primitive: ..., node: NodeRef }` |
| Autodiff → Fusion | `FloatTensor<Fusion<B>>` | `FusionTensor { id: TensorId, stream: StreamId }` |
| Fusion → CubeK | `OperationIr` + `TensorId → Handle` 映射 | `FuseTrace { blocks, resources }` |
| CubeK → CubeCL | `KernelLauncher` + `KernelSettings` | `KernelDefinition { body: Scope }` |
| CubeCL → GPU | `KernelId` + `CompiledKernel` | PTX / WGSL / SPIR-V |

每次交接在源代码中都对应一次 trait 方法调用——`Backend::float_matmul` → `FusionBackend::float_matmul` → `FusionRuntime` trait 的实现 → `CubeKernel::define()`。trait 是各层之间的硬边界。

---

## 七、推迟的边界：什么不能推迟

四层推迟的共同前提是：**推迟后的决策条件必须在推迟到的时机仍然可用**。

- **Burn**：编译期需要知道所有可能的后端组合。如果后端是运行时动态加载的（如插件系统），编译期单态化就不适用——`DispatchDevice` 枚举需要提前列出所有变体。
- **CubeCL**：JIT 时需要 GPU 编译器可用（NVRTC、wgpu 等）。在没有编译器的环境（如某些嵌入式 GPU），JIT 不可行——需要 AOT 预编译。
- **CubeK**：autotune 需要实际硬件执行 benchmark。在没有目标硬件的 CI 环境，autotune 只能跳过——退化为使用默认候选。
- **Burn-ONNX**：AOT 编译需要 `build.rs` 能访问 ONNX 模型文件。模型路径在构建期确定——运行时换模型需要重新构建。

### 推迟的隐性成本

每层推迟机制都有"首次支付"特征——第一次遇到某组合时慢，后续快。这对生产部署的影响取决于调用模式：

| 推迟层 | 首次成本 | 触发条件 | 缓解手段 |
|--------|----------|----------|----------|
| L1 编译期 | 编译时间（分钟级） | 每次 `cargo build` | 增量编译缓存 |
| L2 JIT | JIT 编译 + SSA 优化（秒级） | 首次 launch 每个 (kernel, comptime, vec, cubdim) 组合 | `CompilationCache` 磁盘缓存 |
| L3 Autotune | benchmark N 个候选（秒~分钟级） | 首次执行每个 (M,N,K,dtype,layout) 组合 | 指数分桶减少 key 空间；持久化 autotune 结果 |

对于推理服务（固定模型 + 固定 batch size），三层推迟的成本只在首次请求时支付。对于训练（每次 batch 大小可能不同），autotune 的分桶策略（`#[autotune(anchor(...))]`）将连续的 `M` 值映射到离散桶，减少了需要 benchmark 的 unique key 数量。

---

## 八、为什么推迟能正交组合

Tracel 生态的组件可以像积木一样组合（`Autodiff<Fusion<CubeBackend<CudaRuntime>>>`），是因为每层推迟的决策是**正交的**：

- Burn 的类型栈在 L1 展开——它对 L2/L3 的 JIT 和 autotune 无感知
- CubeCL 的 JIT 在 L2 编译——它不关心 kernel 是被 Fusion 合并来的还是直接 launch 的
- CubeK 的 autotune 在 L3 benchmark——它不关心 kernel 是用 `#[cube]` 手写的还是 ONNX AOT 生成的

每层只看到下层提供的 trait 接口，不穿透到下层内部。这种分层约束由 `Backend` trait 的拆分设计（1 + 8 个超 trait）和 CubeK 的 Blueprint-Routine-Autotuner 三层纪律共同保证。

---

## 九、与其他生态的对比

| 决策 | PyTorch | TensorFlow/XLA | Tracel |
|------|---------|---------------|--------|
| 后端选择 | 运行时 Dispatch Key | 编译期（XLA 编译整个图） | 编译期（trait 单态化） |
| 算子融合 | 运行时（Dynamo / inductor） | 编译期（XLA HLO fusion） | 运行时增量融合（L2 层：入队推迟） |
| GPU 代码生成 | AOT（nvcc 预编译 CUDA kernel）+ Triton JIT（Python 层） | AOT（XLA → PTX/Metal） | JIT（L2：首次 launch） |
| kernel 实现选择 | 手写 CUDA + Triton autotune | XLA 后端固定实现 | autotune（L3：首次执行 benchmark） |
| 模型导入 | 运行时（ONNX Runtime / torch.onnx） | 运行时（TF Serving）+ AOT（XLA 图导出） | AOT（`build.rs` 生成 Rust 源码） |

Tracel 的独特之处在于：**三种推迟机制共享一个宿主语言（Rust）**——编译期单态化、proc-macro 代码生成、JIT 编译都在 Rust 生态内完成。这意味着：
- Comptime 代码和 kernel 代码写在同一份源码里（不是 Python 生成 Triton kernel）
- 类型栈的所有组合在 `cargo build` 时检查（不是运行时 crash）
- AOT 生成的 Rust 代码穿过同一套类型栈和融合流（生成代码和手写代码等价）

---

## 相关文档

### 系统设计文章
| 项目 | 文章 |
|------|------|
| 全栈 | [全景篇](burn/burn-systems-architecture.md) |
| Burn | [Fusion](burn/kernel-fusion-system-design.md)、[Autodiff](burn/autodiff-system-design.md) |
| CubeCL | [Autotune](cubecl/autotune-system-design.md)、[JIT 编译管线](cubecl/jit-compilation-pipeline.md) |

### 导航与教程
| 项目 | 地图 | 专题计划 |
|------|------|----------|
| Burn | [summary.md](burn/summary.md) | [fusion/index.md](burn/fusion/index.md) |
| Burn ONNX | [onnx-summary.md](burn/onnx-summary.md) | [onnx/index.md](burn/onnx/index.md)（待写） |
| CubeCL | [summary.md](cubecl/summary.md) | [index.md](cubecl/index.md) |
| CubeK | [cubek/summary.md](cubek/summary.md)（待写） | — |

*Tracel 底层机制系列 · 跨项目架构地图 · 导航见 [README](../README.md)*

---

→ 下一篇：[全景篇](burn/burn-systems-architecture.md) — 一行代码穿行四个核心系统

[概念索引](concept-index.md) · [源码版本管理](SOURCE-VERSION.md)
