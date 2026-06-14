# 正交分层：四个项目的交互边界

> Burn、CubeCL、CubeK 解决的是不同层次的问题。但它们能像 `Autodiff<Fusion<CubeBackend<WgpuRuntime>>>` 一样自由叠加，是因为每个项目都有清晰的边界和独立的核心理念——不侵入上层的职责，不依赖下层的实现。

## 读前须知

- **本文性质**：跨项目导航。用每个项目**自己的话**解释它的核心设计（附带原文档出处），然后展示它们如何通过 trait 边界正交组合。
- **本文定位**：阅读路径的第一步。不需要先读其他文章——读完你能解释为什么 Burn 的类型栈组件可以像积木一样叠加。
- **读完后**：按 [阅读路径](../README.md) 继续全景篇和各系统文章。

---

## 四个项目，各自的核心

四个项目共享这条主线，是因为它们都构建在同一组基础设施上：Rust 的类型系统提供编译期计算能力，proc-macro 提供代码生成能力，CubeCL 的 JIT 编译管线提供运行时编译能力。三者叠加，整个生态可以在不牺牲性能的前提下保持组件间的正交性。

---

### Burn

Burn 的核心概念来自 `burn/README.md` line 72：

> Autodiff is actually a backend *decorator*. This means that it cannot exist by itself; it must encapsulate another backend.

Backend Decorator 是 Burn 的核心理念——Autodiff 不是独立的后端，是包裹在其他后端外面的装饰器。Fusion 同理。这意味着用户可以写 `Autodiff<Fusion<CubeBackend<WgpuRuntime>>>`——通过编译期泛型嵌套叠加正交能力，而不是运行时 switch 后端。

为什么这是核心：因为 Burn 选择了 **trait 单态化而不是运行时 dispatch**。PyTorch 的后端切换是 `tensor.to(device)` 加查表；Burn 的后端切换是 `cargo build` 时 rustc 为每种类型组合生成独立机器码。代价是编译时间，收益是运行时零 dispatch 开销，以及推理时可以直接编译排除整个 Autodiff crate。

[系统设计：Kernel Fusion](burn/kernel-fusion-system-design.md) | [系统设计：Autodiff](burn/autodiff-system-design.md)

### CubeCL

CubeCL 的核心概念来自 `cubecl/README.md` line 34：

> [Kernels are] type-checked, borrow-checked, composable, and testable, and you do not have to context-switch into another language or build shader sources by string concatenation at runtime.

CubeCL 的核心理念是 **kernel 不离开 Rust 的编译期安全边界**。不是用 C++ 写 CUDA 再链接，也不是用 Python 字符串拼接 WGSL——而是用 `#[cube]` 标注的纯 Rust 函数，依靠 proc-macro 在编译期展开为 IR，JIT 时再翻译为目标平台代码。

为什么这是核心：因为保持了 Rust 的类型安全和工具链（rust-analyzer、cargo test、cargo expand），同时通过 JIT 编译支持多平台。代价是首次 launch 的冷启动编译延迟，通过 `CompilationCache` 磁盘缓存缓解。

[系统设计：JIT 编译管线](../cubecl/jit-compilation-pipeline.md) | [系统设计：Autotune](../cubecl/autotune-system-design.md)

### CubeK

CubeK 的核心概念来自 `cubek/GUIDE.md` line 8：

> The core philosophy of cubek is the strict separation of kernel structure (Compile Time) from execution parameters.

CubeK 用 **minimal Blueprint** 实现这一哲学。Blueprint 只放结构性信息（算法变体、分块方案形状、swizzle 模式），不放运行时数据（问题大小、硬件属性）。它必须实现 `Hash + Eq`，因为它的哈希值进入 CubeCL 的 JIT 编译 key。如果 Blueprint 包含了 M（问题大小的行数），每种不同的 M 都产生一份独立的 JIT 编译产物——kernel 爆炸。

为什么这是核心：因为它是 CubeK 的"纪律"约束。与 CUTLASS 的模板参数爆炸不同，CubeK 用 Blueprint 的 `Hash` 边界来限制 JIT key 的维度。

[系统设计](cubek/blueprint-routine-autotune.md)

### Burn-ONNX

Burn-ONNX 的核心来自 `burn-onnx/README.md`——它的设计定位是 build-time code generation 而不是 runtime model loader。ONNX Runtime 在运行时加载模型、解析 protobuf、按图执行。Burn ONNX 把这个过程挪到 `build.rs` 中：构建期把 ONNX 模型翻译为可调试的 Rust 源码，生成的代码穿过 Burn 的类型栈，享受与手写模型相同的融合和 autotune 优化。

[详细分析](burn/onnx-summary.md)

---

## 一个 matmul 穿过四层

```
用户代码: tensor.matmul(&other)
    │
    ├─ [编译期] Burn 类型栈
    │   rustc 单态化: Autodiff<Fusion<CubeBackend<CudaRuntime>>>
    │   → tensor 的类型在编译期固定，matmul dispatch 消解为函数指针
    │
    ├─ [编译期] Autodiff 层
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
    ├─ [JIT 编译] CubeCL JIT 编译
    │   KernelLauncher::launch → CubeCL runtime:
    │     1. 计算 JIT key = (kernel, comptime, vectorization, cube_dim, ...)
    │     2. 缓存 miss → 调用 kernel.define() → expand 填入 Scope
    │     3. cubecl-opt: parse_graph → SSA → 定点循环(10 passes)
    │     4. CppCompiler<CudaDialect> → NVRTC → PTX → 写入 CompilationCache
    │   → matmul 的计算逻辑从 Rust #[cube] 变成 PTX 机器码
    │
    └─ [首次执行] Autotune（若 matmul 未缓存）
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

## 各层的前提与限制

- **Burn**：编译期需要知道所有可能的后端组合。如果后端是运行时动态加载的（如插件系统），编译期单态化就不适用——`DispatchDevice` 枚举需要提前列出所有变体。
- **CubeCL**：JIT 时需要 GPU 编译器可用（NVRTC、wgpu 等）。在没有编译器的环境（如某些嵌入式 GPU），JIT 不可行——需要 AOT 预编译。
- **CubeK**：autotune 需要实际硬件执行 benchmark。在没有目标硬件的 CI 环境，autotune 只能跳过——退化为使用默认候选。
- **Burn-ONNX**：AOT 编译需要 `build.rs` 能访问 ONNX 模型文件。模型路径在构建期确定——运行时换模型需要重新构建。

### 首次支付

每个项目都有"首次遇到某组合时慢，后续快"的特征：

| 层 | 首次成本 | 触发条件 | 缓解 |
|----|----------|----------|------|
| Burn 类型栈 | 编译时间（分钟级） | 每次 `cargo build` | 增量编译缓存 |
| CubeCL JIT | JIT 编译 + 优化（秒级） | 首次 launch 每个 kernel+comptime 组合 | `CompilationCache` 磁盘缓存 |
| CubeK Autotune | benchmark N 个候选（秒~分钟级） | 首次执行每个 shape+dtype 组合 | anchor 分桶减少 key 空间；持久化 autotune 结果 |

对于推理服务（固定模型 + 固定 batch size），这些成本只在首次请求时支付。对于训练（每次 batch 大小可能不同），CubeK 的 anchor 分桶将连续的 shape 值映射到离散桶，减少 benchmark 次数。

---

## 组合的机制：trait 是硬边界

组件可以像积木一样组合（`Autodiff<Fusion<CubeBackend<CudaRuntime>>>`），是因为每个项目通过 trait 定义清晰的边界：

- Burn 的类型栈在编译期展开——它对 CubeCL 的 JIT 和 CubeK 的 autotune 无感知
- CubeCL 的 JIT 在首次 launch 时编译——它不关心 kernel 是被 Fusion 合并来的还是直接 launch 的
- CubeK 的 autotune 在首次执行时 benchmark——它不关心 kernel 是用 `#[cube]` 手写的还是 ONNX AOT 生成的

每层只看到下层提供的 trait 接口。`Backend` trait 的拆分设计和 CubeK 的 Blueprint 纪律各自保证了隔离。

## 与其他生态的对比

| 维度 | PyTorch | TensorFlow/XLA | Tracel |
|------|---------|---------------|--------|
| 后端选择 | 运行时 Dispatch Key | 编译期（XLA 编译图） | 编译期 trait 单态化 |
| 算子融合 | 运行时（Dynamo/inductor） | 编译期（XLA HLO fusion） | 运行时惰性入队 + 探索 |
| GPU 代码生成 | AOT（nvcc）+ Triton JIT | AOT（XLA→PTX/Metal） | JIT（首次 launch） |
| kernel 实现选择 | 手写 CUDA + Triton autotune | 后端固定实现 | autotune（首次 benchmark） |
| 模型导入 | 运行时（ONNX Runtime） | 运行时+AOT | AOT（`build.rs` 生成 Rust） |

Tracel 的独特之处：编译期单态化、proc-macro 代码生成、JIT 编译共享一个宿主语言（Rust）。Comptime 代码和 kernel 代码在同一份源码里；类型栈的所有组合在 `cargo build` 时检查；AOT 生成的 Rust 代码穿过与手写代码相同的类型栈。

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
