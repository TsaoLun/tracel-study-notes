# Burn：类型栈、融合流与全栈地图

## 读前须知

- **Burn 是什么**：Tracel 的 Rust 深度学习框架——用编译期单态化把正交能力（Autodiff / Fusion / 后端 / 路由）的自由组合压进类型系统。`Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 是编译期展开的具体类型，与 `device = "cuda:0"` 式的运行时字符串选择是不同路径。
- **本文覆盖**：编译期类型栈（Autodiff 只包浮点、Fusion 全包）、运行时融合流（v0.21.0 channel 重构）、框架开销——作为 Burn 底层机制系列的综合地图。ONNX AOT 见 [专文](onnx-summary.md)，GPU JIT 见 [CubeCL 篇](../cubecl/summary.md)。
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

> Burn 用 Rust trait 系统的编译期单态化，把深度学习框架的核心矛盾——正交能力如何自由组合——解决在类型层面。`Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 是编译期展开的具体类型。运行时，融合流把连续操作推迟合并，通过 worker channel 替代递归锁降低框架开销；GPU 代码由 CubeCL 在首次 launch 时 JIT 编译并 autotune 选最优实现。

---

## 一、问题是正交能力的自由组合

PyTorch 选后端：`device = torch.device("cuda:0")`——运行时 Dispatch Key 查表找 kernel。Burn 面对的问题不同：几组正交能力需要自由组合：

| 能力 | 训练需要 | 推理需要 |
|------|----------|----------|
| Autodiff | ✓ | ✗ |
| Fusion | ✓ | ✓ |
| 后端选择 | CUDA | 可能是 WebGPU |
| 多后端路由 | 不需要 | 可能需要 |

四个维度，Python 生态没法在框架层面组合（PyTorch 的 autograd 和 CUDA 后端在 C++ 里紧耦合）。Burn 的答案：每个能力是一个 trait，组合是泛型嵌套，编译器展开所有组合。

---

## 二、Backend trait：1 + 8 的拆分

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
    type FloatTensorPrimitive: TensorMetadata + 'static;
    type IntTensorPrimitive: TensorMetadata + 'static;
    type BoolTensorPrimitive: TensorMetadata + 'static;
    type QuantizedTensorPrimitive: TensorMetadata + QTensorPrimitive + 'static;
}
```

这种分类使各层可以**选择性包装**不同类型的张量。

---

## 三、Autodiff：只包装浮点张量

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

## 四、Fusion：四种张量全部包装

`Fusion<B>` 同样零大小。包装策略与 Autodiff 相反——四种张量全部换成 `FusionTensor`：

```rust
impl<B: FusionBackend> BackendTypes for Fusion<B> {
    type FloatTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type IntTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type BoolTensorPrimitive = FusionTensor<B::FusionRuntime>;
    type QuantizedTensorPrimitive = FusionTensor<B::FusionRuntime>;
}
```

整数运算也能融合——融合只关心"连续操作能否合并为一个 kernel launch"。

`FusionBackend::fusers(device)` 返回 Vec of fuser（Elementwise / Matmul / Reduce 等）。CubeCL 后端注册 fuser；纯 CPU Flex 返回空列表。

---

## 五、运行时：融合流与 channel 重构（v0.21.0）

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

## 六、DispatchDevice：泛型从用户代码退场

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

## 七、ONNX 入口：构建期 AOT

Burn 的 ONNX 支持是在 `build.rs` 里运行的 AOT 编译器——把 ONNX 翻译为可调试的 Rust 源码与 `model.bpk` 权重，运行时二进制不依赖 ONNX Runtime。生成的 `model.rs` 是普通 Burn 代码，穿过本文的类型栈和融合流，首次遇具体形状时触发 CubeCL JIT。

完整 IR 流水线、注意力融合、分区编译、测试体系——见 [Burn-ONNX 专文](onnx-summary.md)。

---

## 一根调用链（CUDA 训练栈）

`tensor.matmul(&other)` 在 `Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 上：

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

---

## 词汇说明表

### 类型栈

| 术语 | 简要说明 |
|------|----------|
| **Backend trait** | Burn 核心抽象：1 个 Backend + 8 个超 trait，每个拆分有精确动机 |
| **BackendTypes** | 四种独立关联类型区分张量种类，使各层可选择性包装 |
| **Backend decorator** | `Autodiff<B>`、`Fusion<B>` 等零大小包装：`PhantomData` + trait 委托，编译期单态化 |
| **类型栈** | `Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 形式的嵌套泛型 |
| **DispatchDevice** | `burn-dispatch` 的枚举 + `dispatch_device!` 宏，消解用户侧泛型传染 |

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
