# Burn Fusion 专题 · 第一章：双客户端-服务器——从 `from_data` 到 GPU buffer

> **本章锚点**：`let tensor = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device)` 这一行代码。  
> 一次 tensor 创建穿过**两条**独立的 client-server 链路——Fusion 层和 CubeCL 层。Fusion 层把操作推迟入队；CubeCL 层立即在 GPU 上分配 buffer。两条链路通过 `DeviceServiceStage::Upstream` 排成流水线。

> **读者提示**：*Fusion* / *client-server* 等见 [Burn 地图词汇表](../summary.md#词汇说明表)。专题目录见 [index.md](index.md)。

---

## 本章在系列中的位置

| 文档 | 你得到什么 |
|------|------------|
| [index.md · 入门引导](index.md#入门引导burn-fusion-机制新人必读) | 主示例是什么、建议阅读顺序、跟跑方式 |
| [../summary.md §五](../summary.md#五运行时融合流与-channel-重构v0210) | 融合流宏观全貌（10 分钟） |
| **本章** | 看清两条 client-server 链路的结构、一次 tensor 分配穿过它们 |
| [第二章](index.md#章节目录)（待写） | OperationQueue：操作入队但不执行的语义 |

读完本章，你应该能解释：**`Fusion<CubeBackend<WgpuRuntime>>` 这种类型栈里，Fusion 的 server 和 CubeCL 的 server 各管什么**；**`from_data` 创建 tensor 时，Fusion 层登记了什么、CubeCL 层分配了什么**；**为什么 v0.21.0 要把 Fusion 放到 CubeCL 上游**。

---

## 一分钟跑通：用 `RUST_LOG` 看两层各自干活

在本仓库的 `src/burn-test/` 目录（需先在项目根 clone burn 仓库，见 [README](../../../README.md#仓库结构)）：

```bash
cd src/burn-test
RUST_LOG=burn_fusion=trace cargo run --release
```

你会在日志中看到 Fusion 层的内部决策：
- `[explorer]` — Explorer 探索融合机会
- `[stream]` — StreamOptimizer 注册/停止
- `[plan]` — Policy 决策（cache hit / exploration completed）

如果需要同时观察 CubeCL 层的 buffer 分配和 kernel launch，加 `cubecl_wgpu::runtime=trace`：

```bash
RUST_LOG=burn_fusion=trace,cubecl_wgpu::runtime=trace cargo run --release
```

**本章聚焦第一条链路**——从 `from_data` 到 Fusion 入队；第二条链路（CubeCL 内存分配）在本章后半讲。

---

## 主示例

全程用一个三操作融合示例：

```rust
let tensor_1 = Tensor::<2>::from_data(
    [[2., 3.], [4., 5.]], &device
);
let y = tensor_1.clone() * 2.0 + 1.0;  // ScalarMul + ScalarAdd
let z = y.tanh();                       // Tanh
println!("{}", z);                       // ← 触发 drain + 融合 + 执行
```

`Device::wgpu(..)` 默认启用 fusion，内部类型栈是 `Fusion<CubeBackend<WgpuRuntime>>`（等价于 `type Wgpu = Fusion<CubeBackend<WgpuRuntime>>`）。`Fusion` 是本章的主角——它把操作推迟到读张量时才执行。`CubeBackend<WgpuRuntime>` 是真正跟 GPU 对话的那层。

---

## 两条 client-server 链路

Burn 的类型栈里嵌套了两层，各自有独立的 client-server：

```
用户代码
    ↓
Fusion 层
    GlobalFusionClient ──→ FusionServer（worker channel）
    （推迟操作入队）            ├── MultiStream
                              ├── HandleContainer
                              └── Arc<FusionUtilities>
    ↓ 批量提交融合后的操作
CubeCL 层
    ComputeClient ──→ WgpuServer（独立线程）
    （立即分配/执行）          ├── SchedulerMultiStream
                              ├── MemoryManager
                              └── wgpu::Device
    ↓
GPU
```

两层分工明确：
- **Fusion 层**：操作先入队，不执行——等一批 op 凑齐了再融合成一个 kernel
- **CubeCL 层**：收到融合后的批量操作，立刻分配 GPU buffer、启动 kernel

v0.21.0 的关键变化：`FusionServer` 通过 `DeviceServiceStage::Upstream`（`burn/crates/burn-fusion/src/client.rs:36–37`）声明自己在 CubeCL 上游。这意味着 Fusion 处理完一批 op，下游 CubeCL 立即开始编译/执行，同时上游 Fusion 继续处理下一批——两条链路是流水线并行的。

---

## Fusion 层：`GlobalFusionClient` 与 `FusionServer`

### 客户端：`GlobalFusionClient`

`burn/crates/burn-fusion/src/client.rs:19–22`：

```rust
pub struct GlobalFusionClient<R: FusionRuntime> {
    server: DeviceHandle<FusionServer<R>>,
    device: FusionDevice<R>,
}
```

两个字段：
- `server: DeviceHandle<FusionServer<R>>` — 与 server 通信的 handle。`DeviceHandle`（来自 `burn-backend`，由 `cubecl-runtime` 提供）内部是 worker 线程池 + channel——**不是** `Arc<Mutex<...>>`。调用 `submit()` 时，闭包通过 channel 发给 worker 线程；调用 `submit_blocking()` 时，等待结果返回。
- `device: FusionDevice<R>` — 当前使用的设备标识。

`GlobalFusionClient` 的核心方法（同文件）：

| 方法 | 做什么 | 阻塞？ |
|------|--------|:---:|
| `register(stream, repr, operation)` | 把操作入队到指定流 | 否（`submit` fire-and-forget） |
| `sync(fn)` | 排空当前流，执行闭包，返回结果 | 是（`submit_blocking`） |
| `read_tensor_float(tensor, stream)` | 读张量数据：先 drain 流，再取回 float | 是（`submit_blocking`） |
| `register_tensor_handle(handle)` | 把已分配的 GPU buffer 注册到 Fusion 的 handle 表 | 否（`submit`） |

`register` 方法的实现透露出一个重要细节——它根据操作大小选择传输策略（`client.rs:121–136`）：

```rust
if size_of::<O>() < size_of::<UnfusedOp<R>>() {
    self.server.submit(move |server| {
        let operation = UnfusedOp::new(operation, stream);
        server.register(stream, repr, operation);
    });
} else {
    let operation = UnfusedOp::new(operation, stream);
    self.server.submit(move |server| {
        server.register(stream, repr, operation);
    });
}
```

小 op 类型直接传、到 server 侧再包装；大 op 类型在客户端包装好再传——减少 channel 中传输的字节数。这个优化与本章主题无关，但值得注意：**在 v0.21.0 的 channel 架构下，传输字节数直接影响性能**，所以框架做了这种微优化。

### 服务端：`FusionServer`

`burn/crates/burn-fusion/src/server.rs:18–22`：

```rust
pub struct FusionServer<R: FusionRuntime> {
    streams: MultiStream<R>,
    pub(crate) handles: HandleContainer<R::FusionHandle>,
    pub(crate) utilities: Arc<FusionUtilities>,
}
```

三个字段：
- `streams: MultiStream<R>` — 多流管理器。每个线程有自己的 `StreamId`，对应独立的 `OperationQueue` + `Processor`。这是二至五章的主角。
- `handles: HandleContainer<R::FusionHandle>` — tensor ID → GPU buffer handle 的映射表。当 Fusion 层需要"真正执行"一个操作时，从这里查找对应的 buffer。
- `utilities: Arc<FusionUtilities>` — 分布式通信的初始化状态（仅在 `distributed` feature 下使用）。

`FusionServer` 的关键方法：

```rust
pub fn register(&mut self, stream: StreamId, repr: OperationIr, operation: UnfusedOp<R>) {
    self.streams.register(stream, repr, operation, &mut self.handles)
}

pub fn drain_stream(&mut self, id: StreamId) {
    self.streams.drain(&mut self.handles, id)
}
```

- `register`：把操作交给 `MultiStream`，入队到对应流的 `OperationQueue`（第二章展开）
- `drain_stream`：排空指定流——触发 Processor 的 Policy/Explorer 融合决策，执行所有融合后的操作（第三、四章展开）

### `DeviceServiceStage::Upstream`：Fusion 在 CubeCL 上游

`client.rs:36–37`：

```rust
fn stage() -> DeviceServiceStage {
    DeviceServiceStage::Upstream
}
```

这个实现告诉 Burn 的设备服务框架：`FusionServer` 是上游服务。框架据此安排启动顺序和服务间通信——Fusion 先启动，CubeCL 后启动。融合完的一批 op 交给下游 CubeCL 执行，同时 Fusion 继续处理下一批。

---

## CubeCL 层：`ComputeClient` 与 `WgpuServer`

当 Fusion 层 drain 并决定"这批 op 该执行了"，它调用 CubeCL 的 `ComputeClient`。这是**第二条** client-server 链路。

### 客户端：`ComputeClient`

`ComputeClient`（在 `cubecl-runtime` 中定义）是 Fusion 层向 GPU 提交工作的入口。它封装了与具体 GPU 后端（WGPU/CUDA/CPU）的通信。

Fusion 层通过 `ComputeClient` 做两件事：
1. **分配内存**：tensor 创建时，Fusion 层登记一个 `NoOp`，但立即通过 CubeCL client 分配 GPU buffer
2. **启动 kernel**：drain 后，融合好的 `FuseTrace` 通过 CubeCL client 提交 kernel launch

### 服务端：`WgpuServer`

`cubecl/crates/cubecl-wgpu/src/compute/server.rs:60–73`：

```rust
pub struct WgpuServer<C: WgpuCompiler> {
    pub(crate) device: wgpu::Device,
    streams_pool: Vec<StreamId>,
    pipelines: HashMap<KernelId, (Arc<ComputePipeline>, CompilerInfo)>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    // ... compilation cache, utilities, etc.
}
```

关键字段：
- `device: wgpu::Device` — 对 GPU 的直接句柄。所有 buffer 创建、kernel 提交最终都通过它
- `scheduler: SchedulerMultiStream<...>` — 多流调度器，管理 GPU 上的并发执行
- `pipelines` — kernel 编译缓存：`KernelId` → 编译好的 `ComputePipeline`

Fusion 层和 CubeCL 层的通信不是嵌套锁——是 **worker channel + 流水线**。`submit()` 把任务放进 channel，worker 线程取出执行，客户端不等结果。只有读数据时走 `submit_blocking()`，客户端阻塞等结果。

---

## 一次 `from_data` 穿过两条链路

回到主示例的第一行：

```rust
let tensor_1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
```

这行代码涉及两个阶段。

### 阶段 1：Fusion 层登记 `NoOp`

`Tensor::from_data` 最终调用 `FusionBackend::float_from_data`（实现在 `burn/crates/burn-fusion/src/ops/tensor.rs`）。Fusion 层做两件事：

1. **通过 CubeCL 层分配 GPU buffer**（阶段 2）
2. **在 Fusion 流中登记一个 `NoOp`**——表示"这个 tensor 的数据已经在 GPU 上了，不需要额外计算"

`NoOp` 在 Fusion 的 `OperationQueue` 中记录 tensor ID、shape 和 `TensorStatus::NotInit`。它不产生实际计算，但让 Fusion 知道"这个 tensor 存在，后续操作可以引用它"。

### 阶段 2：CubeCL 层立即分配 GPU buffer

在 Fusion 登记 `NoOp` **之前**，实际的内存分配已经通过 CubeCL 链路完成了。调用链：

```
FusionBackend::float_from_data
  → client.register_tensor_handle(handle)    // 把 buffer 注册到 Fusion handle 表
      ↑
      这个 handle 来自 CubeCL 层的内存分配：
      ComputeClient::create_from_slice(data)
        → WgpuServer 收到分配请求
          → MemoryManager::reserve(size)
            → DynamicPool::try_reserve(size)
              → SlicedPool::try_reserve(size)    // 尝试从已有 page 切一片
                → 若没有合适空间：
                  → SlicedPool::alloc(storage, size)
                    → ExclusiveMemoryPool::alloc_page(storage, page_size)
                      → WgpuStorage::alloc(page_size)
                        → wgpu::Device::create_buffer(desc)
                          → GPU 显存分配
```

关键点：**Fusion 层推迟的是"操作怎么合并"，不推迟内存分配**。tensor 的数据在 `from_data` 时就已经在 GPU 上了。Fusion 层只是推迟"什么时候对这份数据执行计算"。

### 为什么分开：Fusion 推迟 vs CubeCL 立即

| 维度 | Fusion 层 | CubeCL 层 |
|------|-----------|------------|
| 操作 | 入队，不执行 | 收到后立即执行 |
| 内存 | 不管理 GPU 内存 | 管理 GPU 内存分配/释放 |
| 时机 | 读张量时才 drain | 立即（buffer 分配）或 drain 后（kernel launch） |
| 状态 | `OperationQueue` 中排队 | `MemoryManager` 中管理 buffer 生命周期 |

分开的理由：
- 融合需要"看到"连续多个操作才能决定哪些可以合并。如果每个操作都立即提交给 GPU，就没机会融合了。
- 但 GPU buffer 分配不需要等待——tensor 的数据就在那里，早分配晚分配不影响融合决策。立即分配还可以让 GPU 有更多时间准备。

---

## 跟读源码：追踪一条 `from_data`

以 WGPU 后端为例，从 `Tensor::from_data` 向下走：

### 1. `Tensor::from_data` → backend dispatch

`burn/crates/burn-tensor/src/tensor/api/base.rs:1895`：

```rust
pub fn from_data<T>(data: T, options: impl Into<TensorCreationOptions>) -> Self
where
    T: Into<TensorData>,
{
    let data = data.into();
    let opt = options.into();
    let dtype = opt.resolve_dtype::<K>();
    Self::new(K::from_data(data, &opt.device, dtype))
}
```

`K::from_data` 对于 `Float` tensor 派发到 `FloatTensorOps::float_from_data`。当 backend 是 `Fusion<CubeBackend<WgpuRuntime>>` 时，经过 `Autodiff`（如果有）→ `Fusion` 的 `float_from_data` 实现。

### 2. Fusion 的 `float_from_data`

`burn/crates/burn-fusion/src/ops/tensor.rs:179–180`：

```rust
async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
    tensor.into_data::<B>().await
}
```

`float_from_data` 的实现（在同文件中）做两件事：
1. 调用 `B::float_from_data(data, device, dtype)` 让底层 CubeCL 后端创建实际 tensor
2. 将结果包装为 `FusionTensor`，在 Fusion 流中登记

### 3. FusionTensor 的 client 和 stream

`burn/crates/burn-fusion/src/tensor.rs`（部分）：

```rust
pub struct FusionTensor<R: FusionRuntime> {
    pub(crate) id: TensorId,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) client: GlobalFusionClient<R>,
    pub(crate) stream: StreamId,
}
```

每个 `FusionTensor` 知道自己属于哪个 `stream`（即哪个线程创建了它）和哪个 `client`（与哪个 `FusionServer` 通信）。

### 4. CubeCL 层的 buffer 分配

在 `float_from_data` 调用链中，底层 CubeCL 后端（`CubeBackend<WgpuRuntime>`）分配 GPU buffer：

- `WgpuRuntime::client(device)` 获取 `ComputeClient`
- `client.create_from_slice(data)` 把 `[2., 3., 4., 5.]` 写入 GPU
  - 实际分配：`MemoryManager::reserve(size)` → `SlicedPool` → 最终 `wgpu::Device::create_buffer`

Fusion 层拿到 buffer handle 后，通过 `client.register_tensor_handle(handle)` 把 handle 和 `TensorId` 的映射注册到 `FusionServer` 的 `HandleContainer` 中。后续 drain 执行时，Fusion 层通过 `TensorId` 查找对应的 GPU buffer。

---

## 对比：有无 Fusion 的区别

内部后端类型从 `Fusion<CubeBackend<WgpuRuntime>>` 换为裸 `CubeBackend<WgpuRuntime>`（去掉 `Fusion<...>` 包装）的效果：

- **有 Fusion**：`tensor_1.clone() * 2.0 + 1.0` 入队三条 `OperationIr`（Clone, ScalarMul, ScalarAdd）；`.tanh()` 入队第四条。`println!` 时 drain → 四合一 `elemwise_fuse` kernel → **一次 GPU launch**
- **无 Fusion**：每条语句立即触发 GPU 操作，分别产生至少 4 次 kernel launch

（在 `src/burn-test/` 中，`Device::wgpu(...)` 默认包含 Fusion。要对比，可将 burn feature 中的 `fusion` 关闭，或在概念上把 Fusion 层替换为直通后端。）

---

## 常见误区

| 误区 | 事实 |
|------|------|
| Fusion 层也管 GPU 内存分配 | Fusion 只管理 handle 注册表；实际 GPU buffer 由 CubeCL 的 `MemoryManager` 分配 |
| `GlobalFusionClient` 用 `Mutex` 跟 server 通信 | v0.21.0 用 `DeviceHandle`（内部是 worker channel + 线程池），`submit()` fire-and-forget |
| `from_data` 时数据还在 CPU | 数据立即通过 CubeCL 分配并写入 GPU buffer |
| Fusion 的 `NoOp` 表示"什么都不做" | 它表示"tensor 数据已在 GPU 上，不需要计算，但后续操作要引用它" |
| 两条 client-server 是嵌套锁关系 | v0.21.0 用 `DeviceServiceStage::Upstream` 排成流水线，Fusion 和 CubeCL 可并行 |

---

## 小结

1. **两条 client-server 链路**：Fusion 层（`GlobalFusionClient` → `FusionServer`）推迟操作入队；CubeCL 层（`ComputeClient` → `WgpuServer`）管理 GPU 内存和 kernel 执行。
2. **`FusionServer`** 内部：`MultiStream`（多流管理）+ `HandleContainer`（tensor ID → buffer 映射）+ `Arc<FusionUtilities>`（分布式状态）。
3. **`from_data` 的两阶段**：先通过 CubeCL 层分配 GPU buffer（立即），再在 Fusion 流中登记 `NoOp`（入队）。
4. **v0.21.0 的 `DeviceServiceStage::Upstream`**：Fusion 在 CubeCL 上游，流水线并行——融合完一批就交给下游执行，上游同时处理下一批。
5. **Fusion 推迟的是操作怎么合并，不推迟内存分配**。

---

## 作业

1. 在 `src/burn-test/Cargo.toml` 的 burn features 中去掉 `wgpu` 的 fusion 子功能（或添加 `default-features = false` 后手动列出不含 fusion 的 features），用 `RUST_LOG=cubecl_wgpu::runtime=trace` 跑一次，对比有 Fusion 和无 Fusion 时 kernel launch 的次数差异。

2. 在 `from_data` 之后、`println!` 之前各加一条 `println!("tensor created")` 和 `println!("before drain")`，观察 Fusion 日志出现的时间点——验证操作入队和 drain 的时序。

3. （选做）阅读 `burn/crates/burn-fusion/src/client.rs:67–139` 中 `GlobalFusionClient::new` 和 `register` 的完整实现，写一段注释说明 `DeviceHandle::new` 做了什么、`register` 方法中大小比较优化的目的。

---

## 下章预告

**[第二章 · OperationQueue：惰性执行与"推迟了什么"](index.md#章节目录)**（待写）：`OperationQueue` 的五个字段（`global`/`relative`/`converter`/`operations`/`variables`）；三操作示例中 Clone → ScalarMul → ScalarAdd → Tanh 的入队过程；`println!` 通过 `Display → into_data → read_tensor_float → submit_blocking → drain_stream` 触发执行。

---

## 系列导航

| 篇 | 文档 |
|:---:|------|
| 地图 | [../summary.md](../summary.md) |
| 计划 | [index.md](index.md) |
| **专题 1** | **本文** |
| 专题 2–8 | 见 [计划表](index.md#章节目录) |

*Burn 底层机制 · Fusion 专题 · 第一章 · [系列索引](../../../README.md)*
