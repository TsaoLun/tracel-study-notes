# Burn 不是后端选择器，是编译器：类型栈、融合流与 ONNX AOT

## 读前须知

- **Burn 是什么**：Tracel 开源的 Rust **深度学习框架**——它的核心不是"又一个 PyTorch 替代"，而是用 Rust 的编译期单态化，把 **正交能力（Autodiff / Fusion / 后端 / 路由）的自由组合** 压进类型系统里。一行 `Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 就是一个完整的深度学习栈。
- **解决什么问题**：PyTorch 的 autograd 引擎和 CUDA 后端在 C++ 里紧耦合——你没法把 autograd 装到 JAX 上，也没法把 XLA 融合装到纯 PyTorch eager 上。Burn 把每个能力变成一个 trait、组合变成泛型嵌套，编译器在编译期展开所有组合——零虚函数表、零运行时分支。
- **本文定位**：Burn 底层机制的**综合地图**——覆盖编译期类型栈、运行时融合流与 8.2× 框架开销、ONNX AOT 编译入口。深入 ONNX 编译器流水线见 [blog-burn-onnx-summary.md](blog-burn-onnx-summary.md)；CubeCL JIT 与 GPU 代码生成见 [blog-cubecl-summary.md](blog-cubecl-summary.md)。
- **术语**：*backend decorator*（零大小的 `PhantomData` + trait 委托包装）、*融合流*（操作先入队、延后 drain 才执行）、*JIT*（首次 launch 才编译 kernel）等，下文首次出现会括号简注；完整释义见文末 **[词汇说明表](#词汇说明表)**。
- **两种读法**：① **横向**——理解 Burn 全栈：编译期（本文前半）→ 构建期（[ONNX 篇](blog-burn-onnx-summary.md)）→ 运行期（本文后半）→ GPU 生成（[CubeCL 篇](blog-cubecl-summary.md)）；② **纵向**——只关心某一层时，跳读到对应小节，遇陌生词查文末词汇表。

---

## 核心结论（读正文前的 spoiler）

> Burn 用 **Rust trait 系统的编译期单态化**，把深度学习框架最核心的矛盾——**正交能力如何自由组合**——解决在类型层面。`Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 不是字符串配置，是编译期展开的具体类型。CPU 执行到 `tensor.matmul(&other)` 时，指令指针直接跳转到 CUDA 的矩阵乘法实现——中间不存在任何"判断我在哪个后端"的代码。
>
> 运行期，融合流把连续操作推迟合并，通过 **worker channel 替代递归锁** 拿回 **8.2×** 框架开销；CubeCL 在首次 launch 时 JIT 编译 GPU 代码并 autotune 选最优实现。
>
> 构建期，Burn ONNX 是一个真正的 **AOT 编译器**——把 ONNX 模型在 `build.rs` 里翻译为可调试的 Rust 源码，不依赖 ONNX Runtime 共享库。

---

## 先看一段用户代码

```rust
fn train<B: AutodiffBackend>(device: B::Device) {
    let model = MyModel::<B>::new(&device);
    let optimizer = AdamWConfig::new().init::<B>();
}
```

这里有且仅有一个泛型参数 `B`。但当你用 CUDA 训练时，Rust 编译器实际展开的类型是：

```
Autodiff<Fusion<CubeBackend<CudaRuntime>>>
```

三层泛型嵌套。每一层各自实现 `Backend` trait，把调用委托给内层。组合方式是**编译期确定的类型级联**，不是 `device = "cuda:0"` 那样的运行时字符串。

**这件事的后果比你想象的严重。**

因为这是类型，所以编译器知道你在训练时用了自动微分、算子融合、CUDA——它会在编译期把所有层展开为一个扁平的具体类型。没有虚函数表，没有运行时分支，没有字符串 hash 查找。

因为这是嵌套，所以任何组合都是合法的。`Autodiff<Fusion<Wgpu>>`（训练，Vulkan 后端）？合法。`Fusion<CubeBackend<CudaRuntime>>`（纯推理，CUDA）？合法。还有 `Router` 层，把不同操作分配到不同后端——大矩阵乘走 GPU，小激活走 CPU。**4 种能力包装 × 6 个后端 = 24 种组合，不需要 24 份代码。**

> **术语**：Burn 源码里把 `Autodiff<B>`、`Fusion<B>` 称为 *backend decorator*——本质是 `PhantomData` + trait 委托的**零大小包装**，不是运行时给张量对象再包一层。下文「层」均指类型栈里的一环。

---

## 一、问题比看起来更难

PyTorch 选后端：

```python
device = torch.device("cuda:0")
tensor = tensor.to(device)
```

背后是一个运行时 Dispatch Key 系统。`"cuda:0"` 被映射为 `DispatchKey::CUDA`，运行时查虚函数表找到 kernel 实现。灵活，但所有决策都在运行时——每个操作都要走一次查表。

Burn 面对的问题更复杂。不是"有几个后端可供选择"，而是"有几组**正交的能力**需要自由组合"：

| 能力 | 训练需要 | 推理需要 | 解释 |
|------|----------|----------|------|
| Autodiff | ✓ | ✗ | 推理时带着梯度图是纯浪费——但你不应该在推理代码里写 if/else |
| Fusion | ✓ | ✓ | 连续逐元素操作合并为一个 kernel，对训练和推理都有收益 |
| 后端选择 | CUDA | 可能是 WebGPU | 同一个模型，训练和部署跑在不同硬件上 |
| 多后端路由 | 可能不需要 | 可能需要 | 大算子放 GPU，小算子放 CPU，控制逻辑放主机 |

四个正交维度。Python 生态没法在框架层面组合它们——PyTorch 的 autograd 引擎和 CUDA 后端在 C++ 里紧耦合。

Burn 的选择：**如果每个能力是一个 trait，组合是泛型嵌套，编译器就会在编译期展开所有组合。**

---

## 二、地基：为什么 Backend trait 有 8 个超 trait

`Backend` trait 定义在 `crates/burn-backend/src/backend/base.rs:89`：

```rust
pub trait Backend:
    BackendTypes
    + FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    + ActivationOps<Self>
    + QTensorOps<Self>
    + TransactionOps<Self>
    + Clone + Default + Sized + Send + Sync + Debug + 'static
```

1 + 8。不是设计过度——每个拆分都有精确的动机。但现在先只看第一个：`BackendTypes`。

```rust
pub trait BackendTypes {
    type Device: DeviceOps;
    type FloatTensorPrimitive: TensorMetadata + 'static;
    type FloatElem: Element;
    type IntTensorPrimitive: TensorMetadata + 'static;
    type IntElem: Element;
    type BoolTensorPrimitive: TensorMetadata + 'static;
    type BoolElem: Element;
    type QuantizedTensorPrimitive: TensorMetadata + QTensorPrimitive + 'static;
}
```

**浮点、整数、布尔、量化——四种张量，四种独立的关联类型。** 这不是学究气的分类。这是整个类型包装能工作的前提。

原因马上揭晓。

---

## 三、第一层：Autodiff——只包装浮点张量

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}
```

两个 `PhantomData`。`Autodiff<B>` 在运行时**不占任何内存**——编译期标记：把 `Backend` 委托给 `B`，仅在浮点张量外跟踪梯度。

关键在于它对 `BackendTypes` 的实现（`crates/burn-autodiff/src/backend.rs:27-40`）：

```rust
impl<B: Backend, C: CheckpointStrategy> BackendTypes for Autodiff<B, C> {
    type FloatTensorPrimitive = AutodiffTensor<B>;  // ← 只替换这个
    type IntTensorPrimitive   = B::IntTensorPrimitive;    // 透传
    type BoolTensorPrimitive  = B::BoolTensorPrimitive;   // 透传
    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive; // 透传
}
```

**只有浮点张量被包装。整数、布尔、量化张量全部透传。**

这就是 `BackendTypes` 把四种张量分家的原因。若共用一个 `TensorPrimitive`，`Autodiff` 只能全包或全不包——要么给整数也建梯度图（无意义），要么绕开自动微分。

`AutodiffTensor<B>` 的内部结构：

```rust
pub struct AutodiffTensor<B: Backend> {
    primitive: B::FloatTensorPrimitive,  // 数据在 GPU 显存中
    node: NodeRef,                        // 计算图中的节点
    rc: NodeRefCount,                     // Arc 引用计数
}
```

数据在底层后端，梯度图在 CPU。`tensor.matmul(&other)` 时，`Autodiff::float_matmul` 做两件事：

1. 从 `lhs.primitive` 和 `rhs.primitive` 取出真实张量，在 GPU 上执行矩阵乘法
2. 记录 `MatmulBackward { lhs_id, rhs_id }` 到计算图

两者**同时发生**——数据在 GPU，图在 CPU，互不阻塞。

`name()` 暴露整个类型栈：

```rust
fn name(device: &Self::Device) -> String {
    format!("autodiff<{}>", B::name(device))
    // → "autodiff<fusion<cubecl<cuda>>>"
}
```

框架用它判断栈里是否含 Autodiff；`burn-train` 的 Learner 据此决定是否允许 `.backward()`。

`AutodiffBackend` 的 `inner()` / `from_inner()` 穿透包装——存权重时只要 `primitive`，不要梯度图。

---

## 四、第二层：Fusion——四种张量全部包装

`Fusion<B>` 同样零大小：

```rust
pub struct Fusion<B: FusionBackend> {
    _backend: PhantomData<B>,
}
```

包装策略与 Autodiff **相反**：

```rust
impl<B: FusionBackend> BackendTypes for Fusion<B> {
    type FloatTensorPrimitive = FusionTensor<B::FusionRuntime>;      // 全换
    type IntTensorPrimitive = FusionTensor<B::FusionRuntime>;        // 全换
    type BoolTensorPrimitive = FusionTensor<B::FusionRuntime>;       // 全换
    type QuantizedTensorPrimitive = FusionTensor<B::FusionRuntime>;  // 全换
}
```

**四种张量都换成 `FusionTensor`。** 整数运算也能融合——融合只关心「连续操作能否合并为一个 kernel launch」。

Autodiff 只包 float、Fusion 包全部——因为**职责不同**。独立关联类型让每一层精确控制自己要干预什么。

`FusionBackend` 的核心：

```rust
fn fusers(device: Self::FusionDevice)
    -> Vec<Box<dyn OperationFuser<Self::Optimization>>>;
```

Elementwise / Matmul / Reduce 等 fuser 由 CubeCL 后端注册；纯 CPU Flex 返回空列表。

---

## 五、运行时：融合流与 8.2× 框架开销

> 本节展开 [第四节](#四第二层fusion四种张量全部包装) Fusion 层的运行时行为——操作如何入队、如何融合、以及为什么 0.21.0 的重构拿回了 8.2× 性能。

### 融合反而更慢之后

2025 年底，Burn 团队 benchmark 融合引擎时得到一个反直觉结果。

融合的本意是把 `x.exp().log().sqrt()` 合成一次 kernel launch，少调度、少读写中间结果。逻辑上应该总是更快。但 16 线程高负载下，**`Fusion<CubeBackend>` 反而比裸 `CubeBackend` 慢**——733ms，而无融合路径更快。

根因不在 PTX 质量，在 **`DeviceHandle`**：0.20.1 用递归互斥锁串起 `FusionServer` 与 CubeCL 运行时——融合在锁里排队、JIT 在锁里编译执行，高并发下互相饿死。0.21.0 改为 worker 线程池上的 fire-and-forget **channel**：融合与执行流水线并行。

| 配置 | 0.20.1 fusion | 0.21.0 fusion | 加速比 |
|------|---------------|---------------|--------|
| 512 reps, 1 thread | 7.44 ms | 2.56 ms | 2.9× |
| 1024 reps, 8 threads | 261.20 ms | 35.36 ms | 7.4× |
| 1024 reps, 16 threads | 733.06 ms | 89.19 ms | **8.2×** |

低并发只有 2.9×——锁竞争尚不致命；16 线程时锁成为瓶颈，收益放大到 **8.2×**。**GPU 指令没变，变的是 CPU 不再在锁上等下游。**

### 融合流：推迟的是「算子怎么合并」

在 `Fusion<B>` 下，`tensor.matmul(&other)` 不会立刻触发 GPU matmul。它生成 `OperationIr::Matmul { lhs, rhs, out }`，进入当前流的队列。

`MultiStream`（`burn-fusion/src/stream/multi.rs:102`）：

```rust
pub struct MultiStream<R: FusionRuntime> {
    shared_sources: HashSet<TensorId>,
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
}
```

每个 `Stream` 有 `OperationQueue` + `Processor`（尝试融合）。只有**读**张量——`.to_data()` 或交给非融合路径——才 `drain_stream`：

```rust
// burn-fusion/src/server.rs
pub fn read_float<B>(&mut self, tensor: TensorIr, id: StreamId)
    -> B::FloatTensorPrimitive
{
    self.drain_stream(id);
    ...
}
```

**不读就不算。** 此外 Burn 还有 **增量融合**：`Processor` 把 op 喂给 Elementwise / Matmul / Reduce 等 fuser；fuser 返回 `Open` 或 `Closed`，关闭后写入 `ExecutionPlanStore`，新 fuser 继续吃下一段。同一条流上，前几段可能已执行，后几段仍在积累——**融合决策与执行流水线化**。channel 重构前，这条流水线被锁掐断；0.21.0 之后才和 8.2× 数据对齐。

### Channel 架构

`GlobalFusionClient` 经 `DeviceHandle` 与 `FusionServer` 通信：

```rust
pub struct GlobalFusionClient<R: FusionRuntime> {
    server: DeviceHandle<FusionServer<R>>,
    device: FusionDevice<R>,
}
```

注册操作是 fire-and-forget：

```rust
let _ = self.server.submit(move |server| {
    server.register(stream, repr, operation);
});
```

`submit()` 不阻塞——任务进 server 的 worker 队列，客户端继续入队。只有 `read_float()` 走 `submit_blocking()`，排空流并取回结果。

**0.20.1：** `DeviceHandle` 内是递归互斥锁，融合与 CubeCL 运行时在**同一把锁**里串行——融合流设计的流水线（边融合边执行）在实现上被压成单线程。

**0.21.0：** worker 池 + 任务队列；`DeviceServiceStage::Upstream` 让融合服务处在 CubeCL **上游**——一批 op 融合完，下游立刻 JIT/launch，上游同时处理下一批。

### 跨流共享：SSA-like 不变量

`FusionTensor` 可 `Send + Clone` 到另一线程的另一 `StreamId`。若 B 流消费 A 流上的张量，A 侧 op 可能还在队列里——GPU buffer 尚未分配。

`tag_shared_view` 的做法：先 **drain 源流**，再让目标流 id 指向同一 `Arc<GpuBuffer>`。

支撑条件：**融合 IR 每个输出用新 `TensorId`，不复用输入 id**——类似 SSA，handle 不被覆盖，跨流只需在 handle 层共享，无需分布式锁。`drop` 时在各自流排队 `OperationIr::Drop`，引用计数归零后回收。

### 与 CubeCL JIT 的边界

`drain` 之后 `burn-cubecl` 调用 CubeK / CubeCL。常见情况是该形状尚无 GPU 可执行体——触发 **JIT 编译**，再经 **autotune** 在候选实现里选最快的。

Burn 侧需要知道的三点：

1. **JIT**：首次 `(kernel, comptime, vectorization)` 组合才编译，结果磁盘缓存；后续直接 launch。
2. **Autotune**：同一 blueprint 下仍有多种 tile（`cubek-std` 里 13 种 `TileKind`）；首次遇 `(M,N,K,layout)` 在真实 buffer 上 benchmark。
3. **与 8.2× 的关系**：JIT/autotune 在 0.20.1 就已存在；**锁卡住的是 fusion 排空后能否及时进入这一层**，不是 JIT 本身变慢。

两层分工：

| 维度 | Fusion 流 | JIT + autotune |
|------|-----------|----------------|
| 推迟什么 | 连续 op 如何合并、何时 drain | 某次 launch 用哪份 GPU 代码、哪种 tile |
| 决策粒度 | 操作序列 | 单次 kernel 的实现 |
| 0.21 的变化 | **channel 消除锁竞争（8.2× 来源）** | 行为不变，但被锁拖累的上下游打通 |
| 只有一层时 | 能融合但可能选慢 kernel | 能选快 kernel 但多次 launch |

`#[cube]` 宏展开、`cubecl-opt` 的 SSA 管线、comptime 与 autotune 分工、13 种 TileKind 的 Blueprint 纪律——详见 [blog-cubecl-summary.md](blog-cubecl-summary.md)。

---

## 六、当泛型从用户代码中退场

代价是 `<B: Backend>` 在用户代码里传染。`burn-dispatch` 用 `DispatchDevice` 枚举 + `dispatch_device!` 静态 match 消解：

```rust
pub enum DispatchDevice {
    Cpu(CpuDevice), Cuda(CudaDevice), Rocm(AmdDevice),
    Metal(WgpuDevice), Vulkan(WgpuDevice), Wgpu(WgpuDevice),
    Flex(FlexDevice), LibTorch(LibTorchDevice), Remote(RemoteDevice),
    Autodiff(AutodiffDevice), // ...
}
```

热路径可内联到具体 `Backend` 实现，无虚表。用户写 `Tensor<Dispatch, 2>`，改后端配置不触发全工程级联重编译。

类型栈（`Autodiff<Fusion<…>>`）留在框架内部；对外是 `Tensor::zeros([128, 128], &device)`。

---

## 七、ONNX：不是加载模型，是编译模型

Burn 的 ONNX 支持不是"加载 ONNX 然后解释执行"——它是一个在 `build.rs` 里运行的 **AOT 编译器**，把 ONNX 模型翻译为可调试的 Rust 源码。

你在 PyTorch 里训练了一个 Transformer，导出为 ONNX。用 ONNX Runtime 加载它：运行时解析 protobuf，构建内部图表示，为每个节点查表找到 kernel 实现。你的二进制旁边必须带着 `libonnxruntime.so`（30MB+）。

用 Burn ONNX 加载它：在 `build.rs` 里调用 `ModelGen::new().input("model.onnx").out_dir("model/").run_from_script()`——**编译结束后**，`model.onnx` 不存在了。取而代之的是 `model.rs`（纯 Rust 源码，包含 struct 定义和 `forward()` 方法）和 `model.bpk`（权重二进制）。你在代码里写下：

```rust
let model: Model = Model::from_file("model.bpk", &device);
let output = model.forward(input);
```

没有图解释器。没有 protobuf。没有运行时查表。`model.forward()` 是你可以在调试器里逐行跟进去的 Rust 函数。

**6 阶段流水线**（protobuf 解析 → 类型推断 → 注意力融合等 8 轮简化 → Rust 代码生成）、**3 层测试**（790 快照 + 178 集成 + 1615 上游）、SDXL 级模型的分区编译——完整内容见 [blog-burn-onnx-summary.md](blog-burn-onnx-summary.md)。

生成的 `model.rs` 是普通 Burn 代码——它会穿过本文的 `Autodiff<Fusion<CubeBackend>>` 栈，在运行时走融合流与调度，并在首次遇到具体形状时触发 CubeCL 的 JIT 与 autotune。**从 PyTorch 导出的 ONNX 到 GPU 上的 PTX——整条链路都是 Rust，都可以追踪。**

---

## 八、为什么不直接用 PyTorch？

云端 8×A100 训 LLM，PyTorch 仍是默认答案。Burn 面向的是 PyTorch 难覆盖的场景：

- 浏览器（WebGPU）、嵌入式（no_std）、Rust 原生推理服务（无 Python FFI）
- 编译期发现图错误，而非运行时 crash
- 模型作为可 diff 的 Rust 源码管理（与 [ONNX AOT](blog-burn-onnx-summary.md) 导入衔接）

对这些场景，**编译期类型栈 + 运行时融合流 + 构建期 AOT** 不是过度设计——是在 Rust 里同时拿到「多后端 + 零运行时 dispatch 开销 + 无外部运行时依赖」的可行路径。

---

## 一根调用链（完整，CUDA 训练栈）

`tensor.matmul(&other)` 在 `Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 上：

```
用户调用 tensor.matmul(&other)
         ↓
dispatch_device! → Autodiff<Fusion<Cuda>>::float_matmul
         ↓
Autodiff 层：
  - 从 lhs.primitive / rhs.primitive 取出 FusionTensor
  - 调用 Fusion::float_matmul
  - 记录 MatmulBackward 到梯度图
         ↓
Fusion 层：
  - 生成 OperationIr::Matmul，入队
  - server.submit (fire-and-forget channel)
  - Processor 尝试与前后 op 融合
  - read / drain → 融合完成，交给 burn-cubecl
         ↓
CubeBackend<CudaRuntime> 层：
  - cubek-matmul + LocalTuner 查 autotune 缓存
  - CubeCL JIT miss → expand → cubecl-opt → NVRTC → PTX
         ↓
GPU 执行
```

0.20.1：16 线程整条路径 **733ms**。0.21.0：**89ms**。瓶颈在 channel 之前的锁，不在 matmul 的 PTX。

---

## 系列导航

| 文档 | 主题 | 适合 |
|------|------|------|
| **本文** | Burn 底层机制地图：类型栈 + 融合流 + ONNX 入口 | 理解 Burn 全栈 |
| [blog-burn-onnx-summary.md](blog-burn-onnx-summary.md) | ONNX→Rust AOT 编译器：6 阶段流水线、注意力融合、分区编译 | 深入 ONNX 导入 |
| [blog-cubecl-summary.md](blog-cubecl-summary.md) | CubeCL 编译器框架地图：`#[cube]`、SSA、autotune、CubeK | 理解 GPU 代码生成 |
| [blog-cubecl-plan.md](blog-cubecl-plan.md) | CubeCL 专题写作计划 + 入门引导 | 跟练 GPU kernel |
| [blog-cubecl-1.md](blog-cubecl-1.md) | CubeCL 专题 1：GELU 走通 launch | 跑第一个 kernel |

---

## 词汇说明表

> 正文首次出现的术语，在此可查完整释义。首次阅读可先看 **粗体** 词条，其余作查阅用。

### 类型栈与编译期

| 术语 | 简要说明 |
|------|----------|
| **Backend trait** | Burn 的核心抽象：1 个 `Backend` + 8 个超 trait（`BackendTypes`、`FloatTensorOps`、…），每个超 trait 拆分有精确动机。 |
| **BackendTypes** | 第一种超 trait：用四种独立关联类型（`FloatTensorPrimitive`、`IntTensorPrimitive`、`BoolTensorPrimitive`、`QuantizedTensorPrimitive`）区分张量种类，使各层可选择性包装。 |
| **Backend decorator** | `Autodiff<B>`、`Fusion<B>` 等零大小包装：`PhantomData` + trait 委托，编译期单态化，运行时无开销。 |
| **类型栈** | `Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 形式的嵌套泛型，编译器展开为一个扁平具体类型。 |
| **DispatchDevice** | `burn-dispatch` 的枚举：统一所有后端的 device 类型，用 `dispatch_device!` 宏静态 match，消除用户代码中的泛型传染。 |
| **单态化** | Rust 编译期为每个具体泛型组合生成一份专用代码，无虚函数表、无运行时分支。 |

### 运行时融合流

| 术语 | 简要说明 |
|------|----------|
| **Fusion** | Burn 运行时把连续逐元素/矩阵操作合并为一个 kernel launch 的机制；`Fusion<B>` 是类型栈中启用的 decorator。 |
| **融合流（Fusion stream）** | 操作先入 `OperationQueue`，由 `Processor` 尝试与前后 op 融合——只有 `drain`（读张量）时才真正执行。 |
| **MultiStream** | 多流管理：每个 `StreamId` 有独立的操作队列和 `Processor`；跨流共享通过 SSA-like 的 `TensorId` 不变性实现。 |
| **增量融合** | Processor 把 op 喂给 fuser；fuser 返回 Open/Closed，关闭后写入 ExecutionPlanStore——融合决策与执行可流水线重叠。 |
| **DeviceHandle / GlobalFusionClient** | 0.21.0 的 channel 架构：`submit()` fire-and-forget 入队，worker 池并行处理；替代 0.20.1 的递归互斥锁。 |
| **8.2×** | 0.21.0 channel 重构后 16 线程 benchmark 的加速比：733ms → 89ms。根因是锁消除了融合与执行的流水线并行。 |
| **Drain** | 排空融合流：`read_float` 等操作触发 `drain_stream`，将队列中所有 op 融合并提交执行。不读就不算。 |

### ONNX AOT 编译

| 术语 | 简要说明 |
|------|----------|
| **AOT 编译器** | Ahead-of-Time：在 `build.rs` 构建期把 ONNX 翻译为 Rust 源码，而非运行时解释 protobuf。 |
| **注意力融合** | `coalesce_attention.rs`（1368 行）：识别 ONNX 图里的 MatMul→Scale→Mask→Softmax→MatMul 模式，融合为单一 Attention 节点。 |
| **分区编译** | SDXL 级大图（上万个节点）切成 64–256 节点的子模块，用前缀和差分数组贪心选切点。 |
| **6 阶段流水线** | Protobuf → 类型推断 → 8 轮简化 → Rust 代码生成。详见 [ONNX 篇](blog-burn-onnx-summary.md)。 |

### CubeCL / GPU

| 术语 | 简要说明 |
|------|----------|
| **CubeCL** | Tracel 的多平台 GPU 计算框架：`#[cube]` 写 kernel，JIT 到 CUDA/HIP/WGPU/CPU。 |
| **CubeK / cubek** | 基于 CubeCL 的成品算子库（matmul、attention、convolution 等），与 cubecl 分仓。 |
| **JIT** | Just-In-Time：首次 launch 某组参数时才编译 GPU kernel；结果磁盘缓存。 |
| **Autotune** | 对多种已编译实现做 benchmark，按 `(shape, dtype, device…)` 缓存最快候选索引。 |
| **PTX** | NVIDIA 中间汇编；由 NVRTC 从 CUDA C++ 编译，驱动进一步 JIT 为 SASS。 |
| **Tile / TileKind** | 分块矩阵乘的数据布局与实现策略（Cmma tensor core、Register、PlaneVec 等 13 种）。 |
| **Blueprint** | CubeK 中描述 kernel 结构的 comptime 配置；严格控制参数数量以防 kernel explosion。 |

### 缩写速查

| 缩写 | 全称 / 含义 |
|------|-------------|
| AOT | Ahead-of-Time compilation |
| JIT | Just-In-Time compilation |
| SSA | Static Single Assignment |
| PTX | Parallel Thread Execution（NVIDIA） |
| NVRTC | NVIDIA Runtime Compiler |
| WMMA | Warp Matrix Multiply Accumulate |
| SIMD | Single Instruction Multiple Data |

*Burn 底层机制系列 · 综合地图 · [ONNX 篇](blog-burn-onnx-summary.md) · [CubeCL 篇](blog-cubecl-summary.md)*
