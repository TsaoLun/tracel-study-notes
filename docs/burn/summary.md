# Burn：类型栈、融合流与全栈地图

## 读前须知

- **Burn 是什么**：Tracel 的 Rust 深度学习框架。核心机制是 Backend trait 的嵌套装饰——每一层（Autodiff / Fusion / 具体后端）各自实现 `Backend` trait，编译期展开为扁平具体类型。`Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 是训练栈的一种组合，`CubeBackend<WgpuRuntime>` 是纯推理栈，`Flex` 是 CPU 零拷贝栈——不同场景选不同组合，编译期消解 dispatch。与 `device = "cuda:0"` 式的运行时字符串选择是不同路径。
- **本文覆盖**：编译期类型栈（Backend trait 拆分、选择性包装）、两种后端策略（CubeCL JIT 后端 vs Flex CPU 后端）、运行时融合流（v0.21.0 channel 重构）、框架开销。ONNX AOT 见 [专文](onnx-summary.md)，Autodiff 梯度图见 [Autodiff 地图](autodiff/summary.md)，GPU JIT 见 [CubeCL 篇](../cubecl/summary.md)。
- **机制基准**：融合 channel 重构以 burn v0.21.0 为叙述锚点；源码行号为近似值，以路径 + 符号名为准。

系列分工与导航见 [README](../../README.md)。

---

## 架构一览

```
用户代码: fn train<B: AutodiffBackend>(device)
              ↓ rustc 单态化
Autodiff<Fusion<CubeBackend<CudaRuntime>>>
    │          │              └─ CubeCL JIT → PTX
    │          └─ Fusion: 操作入队、增量融合、drain 执行
    └─ Autodiff: 只包装浮点张量，记录梯度图
              ↓ 运行时
tensor.matmul(&other) → dispatch → Autodiff::float_matmul
    → Fusion::float_matmul (入队) → drain → CubeBackend → CubeK → CubeCL
```

三层各自实现 `Backend` trait，编译期展开为扁平具体类型——运行时 dispatch 消解在编译期。

---

## 核心结论

> Burn 的 Backend trait 被设计为纯 eager 模式——所有 op 是无副作用的纯函数，全局没有可变图上下文。这使每个 decorator（Autodiff、Fusion）可以独立实现 trait 并选择性包装张量类型：Autodiff 只包装浮点（记录梯度图），Fusion 包装全部四种（但需后端额外实现 `FusionBackend` trait，不是所有后端都支持）。Fusion 的 channel 架构（v0.21.0）正是在 eager 约束上叠加惰性的自然方案——操作入队推迟到 drain 时才融合执行，worker channel 替换递归锁使 Fusion 与 GPU 执行可流水线并行。CubeCL 后端（WGPU/CUDA/CPU）和 Flex 后端（纯 Rust CPU，零拷贝视图）代表了两种不同的后端策略：前者靠 JIT 编译获得算子融合和 autotune，后者靠 Arc-COW 和 signed strides 获得零拷贝操作和嵌入部署能力。

---

## 一、Backend trait：1 + 8 的拆分

`Backend` trait（`crates/burn-backend/src/backend/base.rs`）：

```rust
pub trait Backend:
    BackendTypes
    + FloatTensorOps<Self> + BoolTensorOps<Self> + IntTensorOps<Self>
    + ModuleOps<Self> + ActivationOps<Self>
    + QTensorOps<Self> + TransactionOps<Self>
    + Clone + Default + Sized + Send + Sync + Debug + 'static
```

每个超 trait 拆分有精确动机。先看第一个：`BackendTypes`——浮点、整数、布尔、量化四种张量，四种独立的关联类型：

```rust
pub trait BackendTypes {
    type Device: DeviceOps;
    type FloatTensorPrimitive: TensorMetadata + 'static;
    type IntTensorPrimitive: TensorMetadata + 'static;
    type BoolTensorPrimitive: TensorMetadata + 'static;
    type QuantizedTensorPrimitive: TensorMetadata + QTensorPrimitive + 'static;
}
```

`type Device` 告诉框架"这个后端操作哪个设备"；四种张量 primitive 的独立关联类型使各层可以**选择性包装**不同类型的张量。

---

## 二、Autodiff：只包装浮点张量

```rust
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}
```

两个 `PhantomData`——运行时零大小。对 `BackendTypes` 的实现（`crates/burn-autodiff/src/backend.rs`）：

```rust
impl<B: Backend, C: CheckpointStrategy> BackendTypes for Autodiff<B, C> {
    type FloatTensorPrimitive = AutodiffTensor<B>;       // 只替换这个
    type IntTensorPrimitive   = B::IntTensorPrimitive;    // 透传
    type BoolTensorPrimitive  = B::BoolTensorPrimitive;   // 透传
    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive; // 透传
}
```

**只有浮点张量被包装**——整数、布尔、量化全部透传。若共用一种 `TensorPrimitive`，Autodiff 只能全包或全不包。

`AutodiffTensor<B>` 内部：数据在底层后端（`primitive: B::FloatTensorPrimitive`），梯度图在 CPU（`node: NodeRef`）。`tensor.matmul(&other)` 同时做两件事：GPU 执行矩阵乘法 + 记录 `MatmulBackward` 到计算图。

`name()` 暴露整个类型栈：`"autodiff<fusion<cubecl<cuda>>>"`。`burn-train` 据此判断是否允许 `.backward()`。

---

## 三、Fusion：四种张量全部包装

`Fusion<B>` 同样零大小。与 Autodiff 的一个关键差异：`Fusion<B>` 的泛型约束不是 `B: Backend`，而是 `B: FusionBackend`——一个独立 trait（`crates/burn-fusion/src/backend.rs`）。`FusionBackend` 要求后端额外实现 `BackendIr`（handle ↔ tensor 互转）和 `cast_float`（混合精度支持），并声明 `FullPrecisionBackend`（用于梯度累积）。不是所有后端都实现了这个 trait——`CubeBackend<R>` 实现了（在 `burn-cubecl/src/fusion.rs`），而 `Flex` 选择不实现（见下文）。

包装策略与 Autodiff 相反——四种张量全部换成 `FusionTensor`：

```rust
impl<B: FusionBackend> BackendTypes for Fusion<B> {
    type FloatTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type IntTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type BoolTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type QuantizedTensorPrimitive = FusionTensor<B::FusionRuntime>;
}
```

整数运算也能融合——融合只关心"连续操作能否合并为一个 kernel launch"。

`FusionRuntime::fusers(device)` 注册融合器。CubeCL 后端注册四种 fuser（`ElementWiseFuser`、`MatmulFuser`、`ReduceFuser`、`ReduceBroadcastedFuser`），见 `burn/crates/burn-cubecl/src/fusion.rs:144–151`。

### 为什么 Autodiff 和 Fusion 可以独立决定各自的包装策略

Autodiff 只包浮点——它只需要为浮点操作记录梯度图，整数索引、布尔 mask 不需要梯度。Fusion 全部包——整数运算也能从合并 kernel launch 中受益（减少 GPU 调度开销）。两种完全相反的策略共存，靠的是 `BackendTypes` 的四种独立关联类型——每个 decorator 对四种张量类型各自决定"替换还是透传"，编译期展开为具体类型，互不穿透。

这解释了为什么 Burn 用 trait 嵌套而非运行时 dispatch 来解决能力组合问题。PyTorch 的 autograd 和 CUDA 后端在 C++ 层紧耦合——训练栈开启 autograd 就意味着走 CUDA 后端，无法在框架层面自由组合"是否需要梯度 × 是否融合 × 跑什么硬件"。Burn 的每个能力维度是一个独立的 trait 实现，组合是泛型嵌套——但每个能力都是可选的门槛，不是所有后端都需要跨过。

### 两条后端路径：CubeCL JIT vs Flex 零拷贝

Burn 的后端生态有两条主要路径，各自适合不同的部署场景：

| 维度 | CubeCL 后端（WGPU/CUDA/CPU） | Flex 后端 |
|------|------------------------------|-----------|
| 执行模式 | JIT 编译 + GPU 加速（或 CPU MLIR） | eager 执行，纯 Rust |
| Fusion | 支持（`FusionBackend` 已实现） | 不实现 `FusionBackend` |
| 内存策略 | GPU 显存管理（MemoryManager/SlicedPool） | Arc-COW（`is_unique()` 启用原地修改） |
| 零拷贝操作 | 依赖 GPU 专有机制 | signed strides（`isize`）：flip/unfold/expand 均 O(1) |
| 核心收益 | 算子融合 + autotune | 无 JIT 冷启动、3x 更少分配、嵌入部署 |
| 适用场景 | 训练 + GPU 推理 | CPU 推理、WASM、no_std、嵌入式 |

Flex 选择不实现 `FusionBackend` 是经过权衡的设计决策——其 ARCHITECTURE.md 写得很直接："Without JIT compilation, fusion adds tracking overhead with no performance benefit. Deferred operations would still execute one-by-one with intermediate allocations." 没有 JIT 编译器的 CPU 后端，融合只增加入队开销而没有合并 kernel 的收益。

两种策略共享同一个 `Backend` trait 接口——用户代码在 `Tensor<B, D>` 层面不感知差异。`Flex` 和 `CubeBackend<WgpuRuntime>` 都可用于 `Autodiff`（`Autodiff<Flex>` 走 CPU 训练），都支持 ONNX 模型导入。

详细对比见 [burn-flex ARCHITECTURE.md](https://github.com/tracel-ai/burn/blob/main/crates/burn-flex/ARCHITECTURE.md) 和 [COMPARISON.md](https://github.com/tracel-ai/burn/blob/main/crates/burn-flex/COMPARISON.md)。

### Eager Mode 约束：为什么 Fusion 需要 channel

`Backend` trait 的设计注释（`crates/burn-backend/src/backend/base.rs`）明确了一个约束：

> "the backend trait is designed around kernel implementations that can be called without any mutable context or graph."

所有 tensor 操作都是无副作用的纯函数——没有全局计算图、没有可变 session 状态。这意味着：

- **不能**在 `float_matmul` 内部"把操作挂到图里等以后再执行"——trait 签名要求立即返回 tensor
- **不能**在 op 之间共享可变融合状态——每次调用是独立的

Fusion 层需要"拦截操作、推迟执行、批量提交"，这与 `Backend` trait 的 eager 语义直接冲突。解决方案是 **client-server channel 架构**——`Fusion<B>` 的 `float_matmul` 实现先按 eager 语义立即返回一个 `FusionTensor`（只分配 `TensorId`，不触发 GPU 计算），同时通过 channel 把操作闭包发给后台 `FusionServer`。Server 在自己的线程里积累操作、决定融合策略、批量提交给底层后端。前端满足 trait 的 eager 约定，后台做真正的惰性融合。

这解释了 v0.21.0 从递归锁迁移到 worker channel 不仅是性能优化——它是唯一能使 eager trait + 惰性执行共存的架构。

---

## 四、运行时：融合流与 channel 重构（v0.21.0）

> **逐机制跟练**：本节是融合运行时的宏观概述。若要照源码逐行追踪——从 `OperationQueue` 入队到 `elemwise_fuse` kernel 执行，见 [Burn Fusion 专题写作计划](fusion/index.md)（8 章，待写）。

### 问题：递归锁成瓶颈

0.20.1 的 `DeviceHandle` 内是递归互斥锁——`FusionServer` 与 CubeCL 运行时在同一把锁里串行。16 线程高负载下，`Fusion<CubeBackend>` 比裸 `CubeBackend` 慢（融合排队 + JIT 编译执行互相阻塞）。0.21.0 改为 worker 线程池上的 fire-and-forget channel——融合与执行可流水线并行。

> 具体性能数字来自 Burn 团队内部 benchmark，未收录公开 CI。机制变化（递归锁 → worker channel）见 `burn/` · `crates/burn-fusion/src/client.rs` 的 `submit()`/`submit_blocking()` 与 `DeviceServiceStage::Upstream`。

### 融合流：推迟的是"算子怎么合并"

在 `Fusion<B>` 下，`tensor.matmul(&other)` 生成 `OperationIr::Matmul`，进入当前流的队列——不立刻触发 GPU matmul。只有读张量（`.to_data()`）才 `drain_stream`，排空队列并提交执行。

`MultiStream`（`burn/` · `crates/burn-fusion/src/stream/multi.rs`）管理多流：每个 `StreamId` 有独立 `OperationQueue` + `Processor`。Processor 做**增量融合**——把 op 喂给 fuser，fuser 返回 Open 或 Closed；关闭后写入 `ExecutionPlanStore`，新 fuser 继续吃下一段。同一流上，前几段可能已执行，后几段仍在积累。

`submit()` 不阻塞——任务进 server 的 worker 队列，客户端继续入队。只有 `read_float()` 走 `submit_blocking()`，排空流并取回结果。

0.21.0 的 `DeviceServiceStage::Upstream` 让融合服务处在 CubeCL 上游——一批 op 融合完，下游立刻 JIT/launch，上游同时处理下一批。

### 跨流共享

`FusionTensor` 可 `Send + Clone` 到另一线程的另一 `StreamId`。`tag_shared_view` 先 drain 源流，再让目标流指向同一 `Arc<GpuBuffer>`。支撑条件：融合 IR 每个输出用新 `TensorId`，不复用输入 id（SSA-like），handle 不被覆盖，跨流只需在 handle 层共享。

### 与 CubeCL JIT 的边界

| 维度 | Fusion 流 | JIT + autotune（[CubeCL 篇](../cubecl/summary.md)） |
|------|-----------|------------------------------------------------------|
| 推迟什么 | 连续 op 如何合并、何时 drain | 某次 launch 用哪份 GPU 代码、哪种 tile |
| 决策粒度 | 操作序列 | 单次 kernel 的实现 |
| 决策时机 | 运行期（读张量前） | 首次 launch / 首次遇 shape |
| 0.21 的变化 | channel 消除锁竞争 | 行为不变，但被锁拖累的上下游打通 |

---

## 五、DispatchDevice：泛型从用户代码退场

`burn-dispatch` 用 `DispatchDevice` 枚举 + `dispatch_device!` 静态 match 消解用户侧泛型：

```rust
pub enum DispatchDevice {
    Cpu(CpuDevice), Cuda(CudaDevice), Rocm(AmdDevice),
    Metal(WgpuDevice), Vulkan(WgpuDevice), Wgpu(WgpuDevice),
    Flex(FlexDevice), LibTorch(LibTorchDevice), Remote(RemoteDevice),
    Autodiff(AutodiffDevice), // ...
}
```

热路径可内联到具体 `Backend` 实现，无虚表。用户写 `Tensor<Dispatch, 2>`，改后端配置不触发全工程级联重编译。类型栈留在框架内部。

---

## 六、ONNX 入口：构建期 AOT

Burn 的 ONNX 支持是在 `build.rs` 里运行的 AOT 编译器——把 ONNX 翻译为可调试的 Rust 源码与 `model.bpk` 权重，运行时二进制不依赖 ONNX Runtime。生成的 `model.rs` 是普通 Burn 代码，穿过本文的类型栈和融合流，首次遇具体形状时触发 CubeCL JIT。

完整 IR 流水线、注意力融合、分区编译、测试体系——见 [Burn-ONNX 专文](onnx-summary.md)。

---

## 完整调用链：三条路径

`tensor.matmul(&other)` 在不同类型栈上的行为：

**路径 A：CUDA 训练栈** `Autodiff<Fusion<CubeBackend<CudaRuntime>>>`

```
用户调用 tensor.matmul(&other)
         ↓
dispatch_device! → Autodiff<Fusion<Cuda>>::float_matmul
         ↓
Autodiff 层：取出 FusionTensor → 调用 Fusion::float_matmul → 记录 MatmulBackward
         ↓
Fusion 层：生成 OperationIr::Matmul → 入队 → fire-and-forget channel
         → Processor 增量融合 → drain → 交给 burn-cubecl
         ↓
CubeBackend 层：cubek-matmul + LocalTuner 查 autotune 缓存
         → CubeCL JIT miss → expand → cubecl-opt → NVRTC → PTX
         ↓
GPU 执行
```

**路径 B：GPU 推理栈** `Fusion<CubeBackend<WgpuRuntime>>` 或裸 `CubeBackend<WgpuRuntime>`

```
tensor.matmul(&other)
    ↓
Fusion<Wgpu>::float_matmul（如有 Fusion feature）
    → 入队 → drain → CubeBackend
或
CubeBackend<WgpuRuntime>::float_matmul（无 Fusion）
    → 直接通过 ComputeClient 提交 GPU kernel
    ↓
cubek-matmul → CubeCL JIT → WGSL/SPIR-V（无 NVRTC）
```

**路径 C：CPU 推理/训练栈** `Flex` 或 `Autodiff<Flex>`

```
tensor.matmul(&other)
    ↓
Flex::float_matmul → gemm crate → Rayon 并行 → 本机 CPU（无 JIT）
（若在 Autodiff<Flex> 下，同步记录 MatmulBackward 到 CPU 梯度图）
```

三条路径共享同一个 `Backend` trait 接口——用户代码写一次，换后端不改变调用逻辑。Fusion 和 Autodiff 各自决定是否参与，编译期消解为具体路径。

---

## 词汇说明表

### 类型栈

| 术语 | 简要说明 |
|------|----------|
| **Backend trait** | Burn 核心抽象：1 个 Backend + 8 个超 trait，每个拆分有精确动机 |
| **BackendTypes** | 四种独立关联类型区分张量种类，使各层可选择性包装 |
| **Backend decorator** | `Autodiff<B>`、`Fusion<B>` 等零大小包装：`PhantomData` + trait 委托，编译期单态化 |
| **类型栈** | 嵌套泛型的具体组合，如 `Autodiff<Fusion<CubeBackend<CudaRuntime>>>`（训练）、`Fusion<CubeBackend<WgpuRuntime>>`（推理）、`Flex`（CPU） |
| **FusionBackend** | 独立 trait（非 `Backend` 的子 trait）——后端实现它才能被 `Fusion<B>` 包装。要求 `BackendIr` + `cast_float` + `FullPrecisionBackend` |
| **Flex** | 纯 Rust CPU 后端，替代 `burn-ndarray`。不实现 `FusionBackend`；靠 Arc-COW + signed strides 实现零拷贝操作 |
| **Eager Mode** | `Backend` trait 的设计约束：所有 op 是无副作用的纯函数，无全局图上下文——Fusion 的 channel 架构是对此约束的补偿 |
| **FullPrecisionBackend** | `FusionBackend` 的关联类型——低精度后端的全精度对应物，用于梯度累积（如 f16 推理 → f32 梯度） |

### 融合流

| 术语 | 简要说明 |
|------|----------|
| **Fusion** | 运行时把连续操作合并为一个 kernel launch 的机制 |
| **MultiStream** | 多流管理：每个 `StreamId` 独立操作队列 + Processor |
| **增量融合** | Processor 喂 op 给 fuser，Open/Closed 状态机，融合决策与执行可流水线重叠 |
| **DeviceHandle / submit()** | 0.21.0 channel 架构：fire-and-forget 入队，worker 池并行 |
| **Drain** | `read_float` 等操作触发，排空流中所有 op，融合并提交执行 |
| **Block** | `StreamOptimizer` 中一组可融合操作的抽象：包含 op 序列、`OptimizationBuilder` 列表、ordering；通过 tensor ID 交集决定 accept/reject |
| **FuseBlockBuilder** | 构建融合块的 builder：跟踪 `ops`/`reads`/`writes`/`tensor_writes`，`tensor_writes()` 做数据流分析决定哪些中间结果不需写全局内存 |
| **FuseTrace** | 融合的最终产物：`Vec<FuseBlock>` + `FuseResources`（inputs/outputs/scalars），交给 `LaunchPlanExecutor` 执行 |

### 缩写

| 缩写 | 全称 |
|------|------|
| AOT | Ahead-of-Time compilation |
| JIT | Just-In-Time compilation |
| SSA | Static Single Assignment |
| PTX | Parallel Thread Execution（NVIDIA） |
| NVRTC | NVIDIA Runtime Compiler |

*Burn 底层机制系列 · 综合地图 · 导航见 [README](../../README.md)*

系列内相关：**[Autodiff 地图](autodiff/summary.md)**（选择性包装 + 梯度图 + Checkpointing） · **[ONNX 地图](onnx-summary.md)**（AOT 编译器） · **[跨项目架构](architecture.md)**（决策推迟主线）
