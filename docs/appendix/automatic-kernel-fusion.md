# Burn 中的自动 Kernel Fusion 是如何工作的？

## Burn 的 Tensor 抽象

1. **`Tensor` 的泛型参数**

   当前 `Tensor` 定义（`burn/crates/burn-tensor/src/tensor/api/base.rs:74`）：

   ```rust
   pub struct Tensor<const D: usize, K = Float>
   where
       K: Basic,
   {
       pub(crate) primitive: BridgeTensor,
       _kind: PhantomData<K>,
   }
   ```

   - `D`：`usize` —— 维度（编译期常量）
   - `K`：元素类型（默认为 `Float`，可以是 `Int`、`Bool` 等实现了 `Basic` trait 的类型）

   > **注意**：旧版 burn（2025 年中及之前）的 `Tensor` 定义为 `Tensor<B, D, K>`，其中 `B` 是 Backend 泛型参数。commit `dbf03c516` (#4717) 将 Backend 从编译期泛型中移除，改为通过 `Device` 在运行时确定。当前定义：`burn/crates/burn-tensor/src/tensor/api/base.rs:74`。内部 `primitive: BridgeTensor`（`burn/crates/burn-tensor/src/bridge/kind.rs:91`）持有运行时 dispatch 信息，根据 `Device` 将操作路由到对应的后端实现。

2. **后端组合性**

   虽然后端不再是 `Tensor` 的泛型参数，但后端类型本身仍然是可组合的。`Fusion` 包裹 `CubeBackend` 以获得 kernel fusion 能力：

   ```rust
   // burn/crates/burn-wgpu/src/lib.rs:34-75
   type WgpuInner<C> = burn_fusion::Fusion<CubeBackend<cubecl::wgpu::WgpuRuntime<C>>>;
   pub type Wgpu = WgpuInner<AutoCompiler>;
   ```

   `Wgpu` 类型别名展示了后端组合：`Fusion<CubeBackend<WgpuRuntime<AutoCompiler>>>`。运行时通过 `Device` 选择具体的后端实例。

3. **Client-Server 架构**

   Burn 的双层 client-server 抽象：

   **Fusion 层：**
   - `GlobalFusionClient` 通过 `DeviceHandle` 与 `FusionServer` 通信（`burn/crates/burn-fusion/src/client.rs:19`）
   - `FusionServer` 拥有 `MultiStream` 及操作队列（`burn/crates/burn-fusion/src/server.rs:18`）

   > **注意**：此代码已更新（`burn/crates/burn-fusion/src/client.rs:19`）。旧版使用 `MutexFusionClient` 以 `Arc<Mutex<>>` 包裹 `FusionServer`，现已改为 `GlobalFusionClient` + `DeviceHandle` + `DeviceService` 模式。`FusionClient` trait 已移除。

   **CubeCL 层：**
   - `ComputeClient` 与 `ComputeServer` 通信
   - 同理执行实际 GPU 操作

4. **高层 Tensor 分配流程**

   分配链路如下：

   ```rust
   reserve() → pool.alloc() → create_page() → WgpuStorage.alloc() → wgpu::Device::create_buffer()
   ```

   **注意要点：**
   - **操作注册**：tensor 创建在 fusion 队列中被记录为 `OperationIr::Init`（包含 handle 注册），但实际的 GPU 内存分配在 CubeCL 层立即发生。
   - **分层抽象**：Fusion 和 CubeCL 各自拥有独立的 client-server 模式
   - **惰性池化**：环形缓冲区复用优先于新分配
   - **关注点分离**：Fusion 处理操作批处理/fusion，CubeCL 处理实际 GPU 执行

**总结**

Burn 的架构依赖于可组合的后端类型系统、双层 client-server 抽象、运行时 `Device` dispatch 和自定义内存管理。

---

## 深入：创建一个 Tensor

1. **Burn 中 `Tensor` 的定义**

   `Tensor<const D: usize, K = Float>`（`burn/crates/burn-tensor/src/tensor/api/base.rs:74`）只有两个泛型参数：维度 `D` 和元素类型 `K`。**Backend 不再是编译期泛型参数**，而是通过运行时 `Device` 确定。

   后端类型本身仍然是可组合的：
   - `CubeBackend` 构建于 `cubecl` crate 之上——可面向多种 **运行时**（runtime）或 GPU 编程 API（Cuda、WebGPU、ROCm 等）
   - `Fusion<B: FusionBackend>` 包裹另一个后端 `B`，为其添加 kernel fusion 能力

   例如 `Wgpu` 类型别名（`burn/crates/burn-wgpu/src/lib.rs:75`）：

   ```rust
   pub type Wgpu = WgpuInner<AutoCompiler>;
   // 其中（当 features 包含 "fusion"）：
   // type WgpuInner<C> = burn_fusion::Fusion<CubeBackend<cubecl::wgpu::WgpuRuntime<C>>>;
   ```

   即 `Wgpu` = `Fusion<CubeBackend<WgpuRuntime<AutoCompiler>>>`，等价于"使用 wgpu 运行时、由 CubeBackend 执行、被 Fusion 包裹（启用 kernel fusion）的后端"。

2. **创建 `Tensor` 的 API：`from_data`**

   ```rust
   // burn-tensor/src/tensor/api/base.rs:1895
   // pub fn from_data<T>(data: T, options: impl Into<TensorCreationOptions>) -> Self
   // 其中 T: Into<TensorData>

   let device = Default::default();
   // 用明确的值创建 tensor —— Backend 不再作为泛型参数
   let tensor_1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
   ```

   > **注意**：旧版 `Tensor` 需要 Backend 作为泛型参数：`Tensor::<Backend, 2, Float>::from_data(...)`。当前版本（`burn/crates/burn-tensor/src/tensor/api/base.rs:1895`）中 Backend 通过 `Device` 在运行时确定，Tensor 不再接受 Backend 泛型参数。

   这一行代码的内部 dispatch 链路：

   ```
   Tensor::from_data()
       ↓
   BridgeTensor::float(Dispatch::float_from_data(data, device.as_dispatch()))
       ↓  (通过宏生成的 dispatch 路由到具体的后端实现)
   Fusion<B>::float_from_data(data, device)
       ↓  (burn-fusion/src/ops/tensor.rs:24)
   1. 调用内层后端 B::float_from_data(data, device) 执行实际 GPU 分配
   2. 创建 InitOperationIr 描述，通过 client.register() 将 handle 注册到 fusion 流
   3. 返回包裹在 FusionTensor 中的输出 tensor
   ```

   > **注意**：此 dispatch 链路已针对 burn 2026.05 源码更新。旧版通过编译期泛型 `B` 直接 dispatch 到 `B::float_from_data`。当前版本：`Tensor::from_data()`（`burn/crates/burn-tensor/src/tensor/api/base.rs:1895`）→ `BridgeTensor::float()`（`burn/crates/burn-tensor/src/bridge/kind.rs:91`）→ `Dispatch::float_from_data`（宏生成）→ `Fusion::float_from_data`（`burn/crates/burn-fusion/src/ops/tensor.rs:24`）。Backend 通过运行时 `Device` 确定，`Tensor` 结构体本身不携带后端类型信息。

3. **`Fusion` 后端的 client-server 抽象**（`burn/crates/burn-fusion/src/`）

   - **`GlobalFusionClient`**（`client.rs:19`）：具体的 fusion client，通过 `DeviceHandle` 与 `FusionServer` 通信。
   - 其核心工作是调用 `register()` 方法，将 tensor 操作（`OperationIr`）注册到 stream 的 `OperationQueue` 中，同时注册为 `DeviceService`。

   ```rust
   // burn/crates/burn-fusion/src/client.rs:19
   pub struct GlobalFusionClient<R: FusionRuntime> {
       server: DeviceHandle<FusionServer<R>>,
       device: FusionDevice<R>,
   }
   ```

   > **注意**：旧版 `MutexFusionClient` 使用 `Arc<Mutex<FusionServer<R>>>` 进行同步。当前版本（`burn/crates/burn-fusion/src/client.rs:19`）改为 `DeviceHandle` + `DeviceService` trait 模式（`burn/crates/burn-fusion/src/client.rs:24-38`），支持跨设备通信。`FusionClient` trait 已移除。

   - **`FusionServer`**（`server.rs:18`）：拥有 `MultiStream`、`HandleContainer` 和 `FusionUtilities`。

   ```rust
   // burn/crates/burn-fusion/src/server.rs:18
   pub struct FusionServer<R: FusionRuntime> {
       streams: MultiStream<R>,
       pub(crate) handles: HandleContainer<R::FusionHandle>,
       pub(crate) utilities: Arc<FusionUtilities>,
   }
   ```

   - **`MultiStream`**（`stream/multi.rs:102`）：包含 `shared_sources`（跨流共享追踪）、`streams: HashMap<StreamId, Stream<R>>`、`optimizations: ExecutionPlanStore` 和 `device`。

   - **`Stream`**（`stream/multi.rs:261`）：每个流包含 `OperationQueue<R>`、`Processor<R::Optimization>` 和 `cursor: u64`。

   > **注意**：以上类型均泛型于 `FusionRuntime`（后端的运行时类型，此处 `R = FusionCubeRuntime<WgpuRuntime<AutoCompiler>>`，定义于 `burn/crates/burn-cubecl/src/fusion.rs:156`，其 `FusionRuntime` trait 实现在 `burn/crates/burn-cubecl/src/fusion.rs:138`）。

4. **Tensor 的内存/缓冲区分配如何发生：**
   - 两层抽象：
     - **Fusion 抽象**：`Fusion::float_from_data` 先让内层后端分配 GPU 内存，再将 handle 注册到 fusion 流中（记录为 `OperationIr::Init`）
     - **CubeBackend 抽象**：实际的 GPU 内存分配由 `CubeBackend` 通过 CubeCL 运行时完成
   - CubeCL 本身也有 client-server 抽象（`ComputeClient` / `ComputeServer`）

### 完整分配流程：

```text
# Fusion Backend 调用被包裹的 CubeBackend，
# CubeBackend 再调用 WgpuRuntime 完成实际工作（即 Tensor 分配）。
# 以下是大致流程

Tensor::from_data()
    ↓
Fusion<CubeBackend ... >::float_from_data()
    ↓
CubeBackend::float_from_data() （直接导向 CubeCL 运行时）
    ↓
burn_cubecl::ops::base::from_data() （实例化 CubeCL 类型的代理）
    ↓
WgpuRuntime::client() （实例化 CubeCL compute client，此处为 Wgpu compute client）
    ↓
ComputeClient<WgpuServer, MutexComputeChannel>::create() （给定 buffer 等资源，存储并返回资源句柄）
    ↓
MutexComputeChannel::create() （若 compute client 使用 mutex 与 Compute Server 通信）
    ↓
WgpuServer::create() （假设 Wgpu 为后端）
    ↓
MemoryManager::reserve()
    ↓
MemoryManagement<WgpuStorage>::reserve()
    ↓
SlicedPool::alloc() （若使用 sliced pool）
    ↓
SlicedPool::create_page() （若需要新页）
    ↓
WgpuStorage::alloc()
    ↓
wgpu::Device::create_buffer() （实际 GPU 分配）
    ↓
StorageHandle created with Storage ID
    ↓
ID pushed to SlicedPool ring buffer
    ↓
ID returned
```

### GPU 内存分配详解（CubeCL 层）

上面的 ASCII 流程从 `Tensor::from_data()` 一路追踪到了 `wgpu::Device::create_buffer()`。前半段（Fusion → CubeBackend → ComputeClient）是层层委托；真正有实质逻辑的是后半段的 **CubeCL 内存管理系统**。它采用类似操作系统虚拟内存的 **Page / Slice 模型**来减少 GPU 分配开销。

#### 核心概念

**Page（页）**：一次 `wgpu::Device::create_buffer()` 调用创建的大块 GPU buffer（例如 32 MB、128 MB）。创建页很昂贵（涉及内核调用、驱动注册），因此应尽量避免频繁创建/销毁。

**Slice（切片）**：页内部的一个子区域。多个小 slice 共享同一个 page，通过 offset 定位，无需各自持有独立 buffer。

**三种 Pool 策略**：CubeCL 按用途将内存分为三类（`WgpuMemManager`，`cubecl/crates/cubecl-wgpu/src/compute/mem_manager.rs:19`）：

| Pool | 用途 | 策略 |
|------|------|------|
| `memory_pool` | 主 GPU 内存（STORAGE + COPY） | `MemoryManagement`（多层策略见下） |
| `memory_pool_staging` | CPU 可读的暂存内存（MAP_READ） | `ExclusiveMemoryPool`（每页一分配，环形复用） |
| `memory_uniforms` | Uniform buffer（UNIFORM + STORAGE） | `ExclusiveMemoryPool` |

之所以分开是因为 wgpu buffer 的 usage 标志在创建时锁定——不同用途的 buffer 不能混用。

#### MemoryManagement::reserve() 多层分配策略

`MemoryManagement`（`cubecl/crates/cubecl-runtime/src/memory_management/memory_manage.rs:451`）是主内存池的核心分配器，按优先级依次尝试：

**第 1 层：PersistentPool**

直接按精确大小索引预分配的 slice，命中后立即返回——跳过所有搜索，零开销。用于永不释放的分配（如模型参数）。`PersistentPool` 不存在"页"的概念，它维护的是平铺的 slice 列表。

**第 2 层：DynamicPool 复用**

`MemoryManagement` 维护多个 `DynamicPool`，每个负责一个大小范围的分配。调用 `pool.try_reserve(size)` 时：
- 遍历池中已有 page，调用 `MemoryPage::try_reserve()` 扫描空闲区域
- `MemoryPage::coalesce()` 合并相邻空闲 slice 以减少碎片
- 找到一个足够大的空闲 slice → 切出所需大小 → 返回，**零 GPU 分配开销**

**第 3 层：alloc 新 page**

前两层都未命中时，调用 `pool.alloc()`：
1. 调用 `WgpuStorage::alloc(page_size)` → `wgpu::Device::create_buffer(page_size)` —— 实际 GPU 分配
2. 将新 page 加入池中
3. 从新 page 中切出 slice 返回

`WgpuStorage`（`cubecl/crates/cubecl-wgpu/src/compute/storage.rs:117`）持有 `wgpu::Device` 引用和一个 `HashMap<StorageId, WgpuMemory>`，`StorageId` 是全局单调递增的原子 ID，供上层持有轻量句柄而不直接引用 `wgpu::Buffer`。

#### 补充说明

- 内存分配和数据传输通过 **DMA** 处理——在独立 GPU 上走 PCIe，在集成 GPU 或 Apple Silicon 上走专用片上互连。此过程不经过 GPU 的 compute pipeline，也不触发 kernel launch。数据传输通常由专用复制引擎通过 GPU 命令队列中的 `MemCopy` 命令处理。
- 对 `wgpu`，`queue.write_buffer()` 将数据暂存到 CPU 可访问的内存，内部记录复制命令并提交到 GPU，无需手动创建 `CommandEncoder`。
- 所有上述抽象都在 **CubeCL 层**，burn 的 `Fusion` 后端不参与内存分配——它的职责是操作排队、融合优化和执行调度。

---

## 观察三个操作及生成的 Kernel

通过追踪实际的 kernel fusion 来理解三个操作如何工作。

```rust
RUST_LOG=burn_fusion=trace cargo run --example custom-wgpu-kernel --release --features wgpu

let device = Default::default();
let tensor_1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device); // 记录为 Init/NoOp
let y = tensor_1.clone() * 2.0 + 1.0; // ScalarMul + ScalarAdd 操作被排队
let z = y.tanh();                     // Tanh 操作被排队
```

> **注意**：此代码已更新。旧版使用 `--example burn-test`，该示例在当前 `burn/examples/` 中已不存在。可参考 `custom-wgpu-kernel` 示例。`RUST_LOG` 目标路径（如 `burn_fusion::stream::store::base`）可能随源码重构而变更，当前 fusion 日志配置见 `burn/crates/burn-std/src/config/fusion.rs`。

日志输出显示了 fusion 演示期间生成了 4 个不同的 kernel，因为 Burn 框架的 fusion 系统在 `Wgpu` 后端上的工作方式：

1. **初始化 Kernel（2个）**：前两个 kernel 属于 WGPU 后端初始化：
   - 处理 buffer 建立和验证
   - 处理元数据和间接 buffer 操作

2. **融合操作 Kernel**：

   ```text
   [2025-06-30T07:48:56Z DEBUG wgpu_hal::metal::device] Naga generated shader for entry point 'elemwise_fuse' and stage Compute
   ```
   - 这是展示融合的关键 kernel。注意它将所有三个操作（乘以 2.0、加 1.0、tanh）合并到一个 kernel 中。在生成的代码中可以看到：

   ```cpp
   metal::float2 l_10_ = buffer_0_global[id];
   float _e66 = scalars_f32_.inner[0];
   metal::float2 l_13_ = l_10_ * _e66;           // 乘以 2.0
   float _e70 = scalars_f32_.inner[1];
   metal::float2 l_16_ = l_13_ + metal::float2(_e70);  // 加 1.0
   metal::float2 _e73 = safe_tanh(l_16_);        // tanh
   ```
   - 日志确认了此次融合：

   ```text
   [2025-06-30T07:48:56Z TRACE burn_fusion::stream::store::base] New execution plan 1 - Operations: 3 - Triggers 1
   ```

3. **Slice Kernel**：

   ```text
   [2025-06-30T07:48:56Z DEBUG wgpu_hal::metal::device] Naga generated shader for entry point 'slice_kernel' and stage Compute
   ```
   - 这个 kernel 处理打印结果时最终的数据提取和格式化。

4. 融合演示之所以有效，是因为 Burn 的 fusion 系统将操作排队而不立即执行。当你写成：

   ```rust
   let temp = tensor_1.clone() * 2.0;
   let y = temp + 1.0;
   let z = y.tanh();
   ```

这些操作被记录但不执行，直到你通过打印结果强制执行。此时，系统将所有三个操作融合到一个 kernel 中执行，比分别运行三个 kernel 更高效。

---

## Fusion 实际如何工作

```rust
let tensor_1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device); // 记录为 OperationIr::Init，分配已在 CubeCL 层完成
let y = tensor_1.clone() * 2.0 + 1.0; // ScalarMul + ScalarAdd 操作
let z = y.tanh();                     // Tanh 操作
```

在深入之前，先看一下生成快速、高性能 GPU kernel 需要什么前提。

---

## 理解 Fusion 优化中的读和写

在 fusion 优化的语境中，"reads" 和 "writes" 指代内存访问模式，用于跟踪操作如何与 tensor 交互。

### FuseBlockBuilder 中的 Reads 和 Writes

```rust
pub struct FuseBlockBuilder {
    pub settings: FuseSettings,
    locals: LocalVariablePool,
    pub ops: Vec<FuseOp>,
    reads: BTreeMap<TensorId, Vec<FuseOp>>,
    writes: BTreeMap<TensorId, Vec<FuseOp>>,
    outputs: RegisteredTensors,
    pub outputs_unhandled: Vec<FuseArg>,
    pub local_inputs: BTreeMap<TensorId, FuseArg>,
    pub shape_ref: Shape,
}
```

> **注意**：此代码已针对 burn 2026.05 源码更新（`burn/crates/burn-cubecl-fusion/src/engine/trace/block.rs:36`）。主要变更：
> - `reads` 值类型从 `(FusePrecision, LayoutInfo)` 改为 `Vec<FuseOp>`（支持同一 tensor 被多次读取）
> - `writes` 值类型从 `FusePrecision` 改为 `Vec<FuseOp>`（支持同一 tensor 被多次写入）
> - 移除了 `tensor_writes` 字段（改为在 `build()` 方法中计算，见 `block.rs:447`）
> - 新增 `outputs`、`outputs_unhandled`、`local_inputs`、`locals` 字段
> - `FuseBlockSettings` 已重命名为 `FuseSettings`

### Reads
`reads` 字段是一个映射，跟踪有哪些 tensor 被 block 中的操作读取：
- **Key**：`TensorId` —— 标识特定 tensor
- **Value**：`Vec<FuseOp>` —— 指定哪些操作读取了该 tensor

当一个操作需要读取 tensor 时，它被注册到此映射中。这帮助优化器理解哪些 tensor 需要从内存加载。

### Writes
`writes` 字段跟踪 block 中操作向哪些 tensor 写入：
- **Key**：`TensorId` —— 标识特定 tensor
- **Value**：`Vec<FuseOp>` —— 指定哪些写入操作与该 tensor 关联

当一个操作产生需要存储到 tensor 中的结果时，它被注册到此映射中。这帮助优化器理解哪些 tensor 需要写回内存。

### Tensor Writes
tensor writes 的逻辑（现在在 `tensor_writes()` 方法中计算）专门跟踪需要写入全局内存的 tensor（相对于可以保留在寄存器或共享内存中的临时结果）。

---

## 为什么跟踪 Reads 和 Writes 很重要

1. **内存访问优化**：通过知道哪些 tensor 被读和写，优化器可以：
   - 最小化全局内存访问
   - 将中间结果保留在更快的内存（寄存器或共享内存）中
   - 合并内存操作以获得更好的性能

2. **数据依赖分析**：跟踪读和写有助于识别：
   - 哪些操作相互依赖
   - 哪些操作可以并行执行
   - 哪些操作可以融合在一起

3. **内存分配**：有助于确定：
   - 中间结果需要多少内存
   - 何时可以复用内存
   - 何时需要分配内存

---

## 示例：如何跟踪 Reads 和 Writes

让我们看看在示例中如何跟踪读和写：

```rust
let tensor_1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
let y = tensor_1.clone() * 2.0 + 1.0;
let z = y.tanh();
```

### 1. 标量乘法：`tensor_1.clone() * 2.0`
- **Reads**：
  - `tensor_1` 从全局内存中读取
  - 标量 `2.0` 从常量中读取
- **Writes**：
  - 一个中间结果（设为 `temp1`）被写入

### 2. 标量加法：`temp1 + 1.0`
- **Reads**：
  - `temp1` 被读取（但这是中间结果，不来自全局内存）
  - 标量 `1.0` 从常量中读取
- **Writes**：
  - 另一个中间结果（设为 `y`）被写入

### 3. Tanh：`y.tanh()`
- **Reads**：
  - `y` 被读取（同样是中间结果）
- **Writes**：
  - 最终结果 `z` 被写入全局内存

### 优化
优化器意识到 `temp1` 和 `y` 是不需要写入全局内存的中间结果。它们可以保留在寄存器或共享内存中。只有 `z` 需要写入全局内存。

### `reads`（在 FuseBlockBuilder 中）：
- 跟踪**所有 tensor 读取**——包括来自全局内存和中间结果
- 映射 `TensorId → Vec<FuseOp>` 显示哪些操作读取了每个 tensor

### `writes`（在 FuseBlock build() 后）：
- **仅包含最终的全局内存写入**
- 映射 `TensorId → Vec<FuseOp>` 用于需要写入全局内存的 tensor

### `tensor_writes()` 方法（现为 `build()` 内的计算逻辑）：
- **分析数据流**，确定哪些中间结果确实需要写入全局内存
- **过滤掉**仅在内核内部使用的中间结果

### 示例分解：

```rust
let tensor_1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
let y = tensor_1.clone() * 2.0 + 1.0;  // temp1 = tensor_1 * 2.0, y = temp1 + 1.0
let z = y.tanh();                       // z = tanh(y)
```

### 注册期间（填充 `reads`）：
1. **`tensor_1 * 2.0`**：
   - `reads[tensor_1.id]` 获得 `FuseOp::Assign(Input → Local(0))`
   - 创建中间结果 `temp1`（Local(1)）

2. **`temp1 + 1.0`**：
   - `reads[temp1.id]` 获得读取 `temp1` 的操作
   - 创建中间结果 `y`（Local(2)）

3. **`y.tanh()`**：
   - `reads[y.id]` 获得读取 `y` 的操作
   - 创建最终结果 `z`（Local(3)）

### `build()` 期间（创建 `writes`）：
`tensor_writes()` 方法分析：
- `temp1`：仅被 `+ 1.0` 操作读取 → 不写入全局内存
- `y`：仅被 `tanh()` 操作读取 → 不写入全局内存
- `z`：最终结果，需要持久化 → 写入全局内存

所以 `writes` 只包含：

```rust
writes[z.id] = FuseOp::Assign(UnaryFuseArgs { input: local_3, out: Output(0) })
```

> **注意**：此代码已更新。旧版 `FuseOp::Assign(Local(3) → Output(0))` 已改为 `FuseOp::Assign(UnaryFuseArgs { input, out })` 结构（`burn/crates/burn-cubecl-fusion/src/engine/codegen/ir.rs`，`FuseOp` 和 `UnaryFuseArgs` 定义）。

### 核心洞见：
`tensor_writes()` 方法执行**数据流分析**：

```rust
// 所有从未被后续操作读取的输出 tensor 都应该被写入，
// 因为它们本质上是 shader 的"逻辑"输出。
for output in self.outputs.iter() {
    if let Some((tensor, _precision)) = output.as_normal_tensor() {
        if let Some((tensor, precision)) = resources.outputs.get(tensor.id) {
            if !matches!(tensor.status, TensorStatus::ReadWrite) {
                result.insert(*precision, tensor.clone());
            } else if resources.buffers.get(tensor.id).is_some()
                && !buffers.contains(&tensor.id)
            {
                result.insert(*precision, tensor.clone());
                buffers.push(tensor.id);
            }
        }
    }
}
```

> **注意**：此代码已针对 burn 2026.05 源码更新（`burn/crates/burn-cubecl-fusion/src/engine/trace/block.rs:447`）。`tensor_writes` 已从结构体字段改为 `build()` 方法内的计算逻辑，参数包括 `resources` 和 `buffers` 跟踪向量。

### 总结
1. **`reads`**：跟踪所有 tensor 读取（全局 + 中间）
2. **`ops`**：跟踪计算操作
3. **`tensor_writes()`**：分析哪些结果需要全局内存写入
4. **`writes`**：**仅**包含全局内存写入操作

通过仔细跟踪读和写，fusion 优化器可以最小化内存访问并最大化计算效率，从而更快地执行神经网络操作。

---

## Burn 中的高层 Fusion 架构（近似）：

（图片：fusion 架构流程图）

### 流程说明：

1. **操作队列（Operation Queue）**：
   - 所有 3 个操作被添加到队列中

2. **流优化器（Stream Optimizer）**：
   - 基于 tensor 依赖关系创建 block
   - `Block 1:` tensor 创建（不能与其他操作融合）
   - `Block 2:` `ScalarMul`、`ScalarAdd` 和 `Tanh`（全部可融合）

3. **Block 优化器（Blocks Optimizer）**：
   - 尝试合并 block（此处不可行）
   - 单独优化每个 block
   - 此简单示例中未发现空洞

4. **执行策略（Execution Strategy）**：
   - `Strategy 1`：单独执行 tensor 创建
   - `Strategy 2`：对逐元素操作执行融合 kernel

最终执行将：
1. 创建 `tensor_1`
2. 执行单个融合 kernel，计算 `z = tanh((tensor_1 * 2.0) + 1.0)`

这消除了存储中间 `y` tensor 的需要，减少了内存流量并提高了性能。

---

## 深入探讨

### Operation Queue：

所有操作被排队等待执行，但不会立即执行——**即执行是惰性的（lazy）**。Stream 中的 `OperationQueue` 实际上是**计算图**，包含等待执行的有序操作序列。

```rust
/// 一个不断增长的 [tensor 操作描述](OperationIr) 列表。
///
/// 每个队列与单个 StreamId 关联——它引用的每个 tensor
/// 都是该 stream 本地的（跨流共享由 MultiStream::tag_shared_view
/// 带外处理，在操作入队之前将源句柄别名到新的本地 tensor id 下）。
pub struct OperationQueue<R: FusionRuntime> {
    /// 操作描述列表。包含精确的 tensor ID 和形状，
    /// 使 kernel 能正确运行。
    pub(crate) global: Vec<OperationIr>,
    /// 操作描述列表。tensor ID 和形状是相对的，
    /// 因为我们不需要知道确切的值，但足以确定
    /// 哪些操作可以融合。
    pub(crate) relative: Vec<OperationIr>,
    pub(crate) converter: OperationConverter,
    pub(crate) operations: Vec<UnfusedOp<R>>,
    pub(crate) variables: HashMap<TensorId, TensorStatus>,
}
```

> **注意**：此代码已针对 burn 2026.05 源码更新（`burn/crates/burn-fusion/src/stream/queue/base.rs:13`）。主要变更：
> - `operations` 字段从 `Vec<Box<dyn Operation<R>>>` 改为 `Vec<UnfusedOp<R>>`
> - `variables` 字段值类型从 `(StreamId, TensorStatus)` 简化为 `TensorStatus`（StreamId 已在队列层面关联）

### 计算图的关键方面：

1. 同时存储**高层操作描述**（`OperationIr`）和**实际可执行操作**。
2. 维护两种表示：
   - **global**：精确操作，带有精确的 tensor ID 和形状
   - **relative**：帮助识别融合机会的表示
3. 通过 `variables` HashMap 跟踪 tensor 变量及其状态。
4. 操作通过 `add` 方法按顺序添加，逐步构建计算图。
5. 当执行被触发时，系统分析此队列以在执行操作之前识别融合机会。

### Stream 与 MultiStream

以上描述的 `OperationQueue` 和 `Processor` 都被封装在 `Stream` 内部：

```rust
// burn/crates/burn-fusion/src/stream/multi.rs:261
pub(crate) struct Stream<R: FusionRuntime> {
    pub(crate) queue: OperationQueue<R>,
    processor: Processor<R::Optimization>,
    pub(crate) cursor: u64,
}
```

**一条 `Stream` 就是一个独立的惰性操作流**。它的核心不变式是：**队列中每个 `OperationIr` 引用的 tensor 都被假定属于同一个 stream**。这保证了单 stream 内的操作可以简单串联——每个 `TensorId` 都可以从同一个 handle map 中解析，不需要跨流协调。

如果系统只有一条 `Stream`，那么 `FusionTensor` 只能在创建它的那个线程上使用。但 `FusionTensor` 实现了 `Send + Clone`——用户完全可以把一个 tensor 传送到另一个线程，然后在那个线程上操作它。接收方线程会提交引用了**其他 stream 的 tensor 的操作**。`MultiStream` 就是解决这个问题的：

```rust
// burn/crates/burn-fusion/src/stream/multi.rs:102
pub struct MultiStream<R: FusionRuntime> {
    shared_sources: HashSet<TensorId>,
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
}
```

**`MultiStream` 是多条并发流的协调器**，维护 `HashMap<StreamId, Stream<R>>`。它的核心职责不是融合优化本身（融合发生在单条 `Stream` 内部），而是保证跨流 tensor 共享的正确性。

**跨流共享策略（`tag_shared_view`）：**

绝不让其他 stream 的 tensor id 出现在本 stream 的队列里。当 `FusionTensor::clone` 或 `into_ir` 检测到 `self.stream != StreamId::current()` 时：

1. 分配一个新的本地 id（`dst`）
2. **物化 `src`**：若 `src` 的 handle 还没被注册（说明产出它的 op 仍 pending），先同步 drain 掉 `src_stream`，强制执行所有 pending op 以注册 handle
3. **别名 handle**：将 `src` 的 backend handle clone 到 `dst` 下（cubecl handle 是 `Arc` 语义，clone 只增加引用计数）- 返回的新 `FusionTensor` 携带 `(id=dst, stream=current)`，后续操作正常在当前 stream 上入队

`shared_sources` 缓存哪些 tensor id 已经做过 drain（避免重复），在对应的 `Drop` op 入队时清理。

**释放语义**：每个 alias 各自在自己所属的 stream 上 drop——refcount 归零后才释放底层 buffer。跨流释放无需额外协调。

**两层结构关系：**

```
FusionServer
  └── MultiStream                     ← 跨流协调
        ├── shared_sources            ← drain 去重
        ├── optimizations             ← 跨 stream 共享的执行计划缓存
        └── streams: HashMap<StreamId, Stream<R>>
              ├── Stream(id=A)        ← 单流，惰性队列
              │     ├── queue         ← OperationQueue
              │     ├── processor     ← Policy + Explorer（寻找融合机会）
              │     └── cursor        ← 已处理操作计数
              ├── Stream(id=B)
              │     └── ...
              └── ...
```

简单说：**融合优化在 `Stream` 内进行，`MultiStream` 只负责多流之间的正确执行顺序和 tensor 共享**。`ExecutionPlanStore` 是跨 stream 共享的，这意味着一条 stream 探索出的优化计划可以被另一条 stream 复用（前提是操作序列匹配）。

### Burn 的惰性执行系统如何工作？

```rust
let tensor_1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device); // 记录为 OperationIr::Init，分配已在 CubeCL 层完成
let y = tensor_1.clone() * 2.0 + 1.0; // ScalarMul + ScalarAdd 操作
let z = y.tanh();

println!("Final result: {}", z);
```

这里打印 `z` 调用的是 `Tensor` 的 `Display` 实现，但不直接触发 kernel fusion 和执行。相反，它尝试为显示而读取 tensor 数据，这间接导致待处理操作的执行。

确切的流程：

1. 当 `fmt` 在 tensor 上被调用时，调用 `display_fmt_impl`（`burn-tensor/src/tensor/api/base.rs:3244`），最终递归到 `display_fmt_recursive`（`base.rs:3147`）→ `display_fmt_inner`（`base.rs:3074`）

2. 在 `display_fmt_inner` 中，触发执行的关键行是：

   ```rust
   let data = burn_tensor::reader::try_read_sync(self.clone().slice(range).into_data_async());
   ```

   > **注意**：此代码已更新（`burn/crates/burn-tensor/src/lib.rs:49`）。旧版路径为 `burn_common::reader::try_read_sync`，当前为 `burn_tensor::try_read_sync`（重新导出自 `burn_std::reader::try_read_sync` → `cubecl_common::reader::try_read_sync`，`cubecl/crates/cubecl-common/src/reader.rs:16`）。

3. 此流程：
   - 首先创建 tensor 的一个切片
   - 然后调用 `into_data_async()` 返回一个 future
   - 然后 `try_read_sync` 尝试同步读取该 future

4. `into_data_async()` 调用触发 fusion 和执行管线：
   - 导致任何待处理操作被物化
   - fusion 系统分析操作图
   - 创建并执行融合 kernel 以计算实际的 tensor 值

5. 执行流程经过：
   - `burn-fusion/src/stream/multi.rs` —— 调用 `drain` 方法处理待处理操作
   - `burn-fusion/src/stream/execution/processor.rs` —— 处理器分析操作以进行融合
   - `burn-cubecl-fusion/src/engine/launch/base.rs` —— `FuseTraceLauncher::launch` 依次调用 `InputPlanner`、`OutputPlanner`、`VectorizationPlanner`，最后通过 `LaunchPlanExecutor::execute`（`engine/launch/executor.rs:46`）执行 kernel launch

### `drain` 方法做什么？

`MultiStream` 的 `drain` 方法负责执行特定 stream 中所有待处理的操作。它的作用如下：

```rust
/// 排空一个流
pub fn drain(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
    id.executes(|| {
        if let Some(stream) = self.streams.get_mut(&id) {
            let num_executed = stream.queue.global.len();
            stream.processor.process(
                Segment::new(&mut stream.queue, handles, id),
                &mut self.optimizations,
                ExecutionMode::Sync,
            );
            stream.cursor += num_executed as u64;
        }
    });
}
```

> **注意**：此代码已针对 burn 2026.05 源码更新（`burn/crates/burn-fusion/src/stream/multi.rs:244`）。旧版 `drain()` 包含 `shared_tensors` 和 `clear_shared_tensors` 等跨流共享逻辑，现已替换为更简洁的 `id.executes(|| ...)` 模式，跨流共享通过 `tag_shared_view`（`multi.rs:168`）在入队前解决。

1. 包装在 `id.executes()` 中以确保正确的流边界
2. 找到具有给定 ID 的 stream
3. 以 `Sync` 模式（立即执行）处理 stream 队列中的所有操作
4. 更新 stream 的 cursor 以跟踪执行进度

### 处理器如何识别可融合的段落？

如上述代码所示，排空 stream 会触发融合过程。请记住，每个 stream 由队列和处理器组成。

```rust
pub(crate) struct Stream<R: FusionRuntime> {
    pub(crate) queue: OperationQueue<R>,
    processor: Processor<R::Optimization>,
    pub(crate) cursor: u64,
}

/// 按照 [策略](Policy) 处理 [stream 段落](StreamSegment)。
pub(crate) struct Processor<O> {
    policy: Policy<O>,
    explorer: Explorer<O>,
}
```

工作流程：
1. 操作被排队在 `OperationQueue` 中
2. `Processor` 分析这些操作以寻找融合机会
3. `Processor` 使用 `StreamSegment` 抽象来访问队列中的操作

关键洞察是处理器不直接决定什么是可融合的。取而代之：
1. 处理器协调融合过程
2. 将实际的融合决策委托给运行时提供的优化构建器
3. 使用 `Policy` 决定何时探索、执行或推迟操作
4. 使用 `Explorer` 寻找优化机会

**注意**：处理器不直接决定什么是可融合的——它通过 `StreamSegment` 抽象工作：

```rust
#[derive(new)]
struct Segment<'a, R: FusionRuntime> {
    queue: &'a mut OperationQueue<R>,
    handles: &'a mut HandleContainer<R::FusionHandle>,
    id: StreamId,
}

pub fn process<Segment>(
    &mut self,
    mut segment: Segment,
    store: &mut ExecutionPlanStore<O>,
    mode: ExecutionMode,
) where
    Segment: StreamSegment<O>,
{
    // ...
    let action = self.policy.action(store, segment.operations(), mode);

    match action {
        Action::Explore => {
            self.explore(&mut segment, store, mode);

            if self.explorer.is_up_to_date() {
                break;
            }
        }
        Action::Defer => {
            match mode {
                ExecutionMode::Lazy => break,
                ExecutionMode::Sync => panic!("Can't defer while sync"),
            };
        }
        Action::Execute(id) => {
            if let ExecutionMode::Sync = mode {
                store.add_trigger(id, ExecutionTrigger::OnSync);
            }

            segment.execute(id, store);
            self.reset(store, segment.operations());
        }
    };
}
```

> **注意**：此代码已更新（`burn/crates/burn-fusion/src/stream/multi.rs:268`）。`Segment` 结构体新增了 `id: StreamId` 字段。

`Segment` 类型获取操作队列的独占访问权限，并通过 `operations()` 方法提供对操作的访问。然后处理器：
1. 使用 `Policy` 决定采取什么操作（**探索、执行、推迟**）
2. 使用 `Explorer` 寻找优化机会
3. 当优化被发现时，存储到 `ExecutionPlanStore` 中

```rust
/// 策略跟踪当前操作流的所有可能的执行计划（id）。
pub(crate) struct Policy<O> {
    /// 与当前流段落兼容的潜在执行计划列表
    candidates: Vec<OperationsValidator<ExecutionPlanId>>,
    /// 已找到的候选执行计划列表；可以继续搜索
    /// 以寻找更好的。
    availables: Vec<AvailableItem>,
    /// 应该执行的已找到的执行计划，以及计划中的操作数量。
    found: Option<(ExecutionPlanId, usize)>,
    /// 已分析的操作数量
    num_operations: usize,
    _item_type: PhantomData<O>,
}

/// 探索并创建新的优化。
pub struct Explorer<O> {
    optimizer: StreamOptimizer<O>,
    num_deferred: usize,
    num_explored: usize,
    is_still_optimizing: bool,
}
```

> **注意**：可融合性最终由运行时提供的 `OperationFuser` 实现决定（`burn/crates/burn-fusion/src/backend.rs:121`），各运行时通过 `FusionRuntime::fusers()` 注册自己的优化器（如 CubeCL 的注册在 `burn/crates/burn-cubecl/src/fusion.rs:144`）。

---

## 但 `Explorer` 到底如何找到优化机会？

如果查询 policy 在处理段落时返回 `Action::Explore`，我们进入上述 `match` 的探索分支。这调用 `Explorer`，其中包含 `StreamOptimizer`。

- `StreamOptimizer` 的首要工作是将 stream/段落中的所有操作注册（或添加）到 `block` 中。

### `StreamOptimizer` 如何注册操作：

此方法尝试在 `StreamOptimizer` 的现有 block 中注册一个操作。它的功能如下：

```rust
impl<O: NumOperations> Explorer<O> {

    // 探索提供的操作。
    pub(crate) fn explore(
        &mut self,
        operations: &[OperationIr],
        mode: ExecutionMode,
    ) -> ExplorationAction<O> {
        self.update(operations); // 这通过下方的 `register_inner` 完成 Block Op 注册

        // 仅在非 sync 模式下能继续探索。
        if let ExecutionMode::Lazy = mode {
            if self.is_still_optimizing {
                return ExplorationAction::Continue;
            }
        }

        let optimization = self.optimizer.optimize(operations); // 注册后，我们进行优化

        ExplorationAction::Completed(optimization)
    }
}

impl<O: NumOperations> StreamOptimizer<O> {

    fn register_inner(&mut self, operation: &OperationIr, force: bool) -> usize {
        let mut added_count = 0;
        for block in self.blocks.iter_mut() {
            match block.register(operation, self.length, force) {
                RegistrationResult::Accepted => {
                    added_count += 1;
                }
                RegistrationResult::NotPartOfTheGraph => {}
            }
        }
        added_count
    }
}
```

### 过程
1. 遍历 `StreamOptimizer` 中的所有现有 block
2. 对于每个 block，通过调用 `block.register()` 尝试注册操作
3. 传递：
   - 要注册的操作
   - 当前长度（在流中的位置）
   - 一个可以覆盖正常注册规则的 force 标志
4. 计数有多少 block 接受了该操作
5. 返回此计数

### 这里 "block" 是什么意思

`Block` 是 `StreamOptimizer` 对有潜力融合在一起的有序操作序列的抽象。每个 block：
1. 包含相关的操作（使用相同的 tensor）
2. 跟踪操作的顺序
3. 维护一组分析操作的优化构建器（现在称为 `OperationFuser`）
4. 可以确定操作是否可以融合

### 在 Block 中注册操作

关键部分是 block 如何决定是否接受一个操作：

```rust
pub fn register(
    &mut self,
    operation: &OperationIr,
    order: usize,
    force: bool,
) -> RegistrationResult {
    if self.ids.is_empty() {
        self.register_op(operation, order);
        return RegistrationResult::Accepted;
    }
    let mut contains = false;
    for node in operation.nodes() {
        contains = self.ids.contains(&node.id);

        if contains {
            break;
        }
    }

    if !contains && !force {
        return RegistrationResult::NotPartOfTheGraph;
    }

    self.register_op(operation, order);
    RegistrationResult::Accepted
}
```

Block 在以下情况下接受操作：
1. block 为空（始终接受第一个操作）
2. 操作使用的 tensor 已在 block 中
3. `force` 标志为 true（覆盖正常规则）

> **注意**：在 `Block<O>`（`burn/crates/burn-fusion/src/search/block.rs:10`）上调用 `register_op` 最终通过调用 `builder.fuse(operation)` 将操作注册到各个 `OperationFuser`（`burn/crates/burn-fusion/src/backend.rs:121`）中。旧版 trait 名为 `OptimizationBuilder`，方法名为 `register()`。操作流经多层抽象，最终结束于 `FuseBlockBuilder`（`burn/crates/burn-cubecl-fusion/src/engine/trace/block.rs:36`）的 ops 向量中。
>
> 在操作注册流程中，**所有 `FuseBlockBuilder` 字段同时被填充**：
> - **self.ops** 获得实际操作
> - **self.reads** 填充输入读取操作
> - **resources.inputs/outputs/scalars** 填充 tensor/标量元数据
> - **writes**（稍后计算）通过分析已填充的资源和操作确定

这就是为什么到 `Block::optimize()` 被调用时（内部调用 `OperationFuser::finish()` → `FuseBlockBuilder::build()`），生成最终融合 kernel 所需的所有信息已经可用！

### Block 操作注册流程

```
Block.register_op
    ↓
OperationFuser.fuse （trait 方法，旧版为 OptimizationBuilder.register）
    ↓
ElementWiseFuser.fuse （或其他具体实现如 ReduceFuser、MatmulFuser）
    ↓
TraceOperationFuser.fuse （旧版为 FuseOptimizationBuilder.register）
    ↓
  → register_numeric / register_binary_ops / register_scalar_ops / register_unary_ops ...
    ↓
TryTraceFuser.fuse （旧版为 TryFuseBuilder.register）
    ↓
TraceFuser.fuse_operation （旧版为 FuseTraceBuilder.register_operation）
    ↓
FuseBlockBuilder.ops.push
```

> **注意**：此流程已更新（trait 定义：`burn/crates/burn-fusion/src/backend.rs:121`）。旧版 trait `OptimizationBuilder` → `OperationFuser`，旧版方法 `register()` → `fuse()`。具体实现也一起重命名：`FuseOptimizationBuilder` → `TraceOperationFuser`（`burn/crates/burn-cubecl-fusion/src/engine/fuser.rs:33`），`TryFuseBuilder` → `TryTraceFuser`（`engine/fuser.rs:754`），`FuseTraceBuilder` → `TraceFuser`（`burn/crates/burn-cubecl-fusion/src/engine/trace/fuser.rs:21`，其中方法为 `fuse_operation` 而非 `register_operation`，`trace/fuser.rs:53`）。

### 更大的图景

此方法是 `StreamOptimizer` 中更大策略的一部分：
1. 当新操作到达时，首先尝试合并 block（如需要）
2. 然后使用 `register_inner` 尝试将操作注册到现有 block
3. 如果没有 block 接受它，为该操作创建一个新 block
4. 跟踪有多少 block，如果超过 `max_blocks` 可能停止优化

> **目标是将操作分组到可以一起优化的 block 中，同时维护操作之间正确的执行顺序和依赖关系。**

这种方法允许系统：
1. 在每个 block 内找到融合机会
2. 处理具有多个独立融合组的复杂流
3. 维护正确的执行语义

---

## Block Optimizer 和 Block 优化过程

**重申**：
- 每个 `Block<O>` 包含可能被融合的操作
- `Block<O>` 中的 `optimize()` 方法找到最佳优化策略：
- 每个 block 还包含一组 `OperationFuser` 实例（旧版为 `OptimizationBuilder`）分析操作
  - **示例**：对于逐元素操作，`ElementWiseFuser`（旧版为 `ElemwiseOptimizationBuilder`）识别可融合的模式

以下是从 `StreamOptimizer` 一直到产生 `FuseTrace` 的流程。

### 完整优化流程

```
Explorer.explore
    ↓
StreamOptimizer.optimize
    ↓
BlocksOptimizer.optimize
    ↓
Block.optimize
    ↓
find_best_optimization_index
    ↓
OperationFuser.finish （trait 方法，旧版为 OptimizationBuilder.build）
    ↓
TraceOperationFuser.finish （旧版为 FuseOptimizationBuilder.build）
    ↓
TryTraceFuser.finish （旧版为 TryFuseBuilder.build）
    ↓
TraceFuser.finish （旧版为 FuseTraceBuilder.build）
    ↓
FuseTrace 被创建
```

> **注意**：此流程已更新（trait 定义：`burn/crates/burn-fusion/src/backend.rs:121-137`）。Trait 和方法名变更：`OptimizationBuilder` → `OperationFuser`，`build()` → `finish()`。具体实现也一起重命名。

### 详细解释

#### 1. StreamOptimizer.optimize

```rust
pub fn optimize(&self, operations: &[OperationIr]) -> BlockOptimization<O> {
    let result = BlocksOptimizer::new(self.blocks.clone()).optimize();

    match result {
        BlocksOptimizerResult::Full(optimization) => optimization,
        BlocksOptimizerResult::WithHoles { strategies, ordering, holes } => {
            // 处理空洞情况...
        }
    }
}
```

`StreamOptimizer` 用其 block 创建一个 `BlocksOptimizer` 并调用 `optimize()`。

#### 2. BlocksOptimizer.optimize

```rust
// burn/crates/burn-fusion/src/search/optimization/blocks.rs:61
pub fn optimize(mut self) -> BlocksOptimizerResult<O> {
    self = self.merging_pass();

    let num_ops = self.num_ops;
    let blocks = core::mem::take(&mut self.blocks);

    let mut strategies: Vec<Box<ExecutionStrategy<O>>> = Vec::with_capacity(blocks.len());
    let mut ordering = Vec::new();
    let mut resolved = vec![false; num_ops];

    for block in blocks {
        let mut block_opt = block.optimize();        // 直接调用 Block::optimize()
        for pos in block_opt.ordering.iter() {
            resolved[*pos] = true;
        }
        ordering.append(&mut block_opt.ordering);
        strategies.push(Box::new(block_opt.strategy));
    }

    // 空洞检测：已解析的 block 之间未被覆盖的位置即为空洞
    let last_resolved_end = resolved.iter().rposition(|&r| r).map(|i| i + 1).unwrap_or(0);
    let holes: Vec<usize> = (0..last_resolved_end).filter(|i| !resolved[*i]).collect();

    if holes.is_empty() {
        let strategy = if strategies.len() > 1 {
            ExecutionStrategy::Composed(strategies)
        } else {
            *strategies.remove(0)
        };
        BlocksOptimizerResult::Full(BlockOptimization::new(strategy, ordering))
    } else {
        BlocksOptimizerResult::WithHoles { strategies, ordering, holes }
    }
}
```

`BlocksOptimizer`：
1. 尝试合并可以组合的 block
2. 处理每个 block 以创建优化策略
3. 将这些策略合并为最终的 `BlockOptimization`

#### 3. Block.optimize

```rust
pub fn optimize(mut self) -> BlockOptimization<O> {
    match find_best_optimization_index(&mut self.builders) {
        BestOptimization::Found { index, score } => {
            let opt = self.builders[index].finish();
            let opt_len = opt.len();
            if opt_len < self.operations.len() {
                self.ordering.drain(opt_len..);
            }

            let strategy = ExecutionStrategy::Optimization {
                ordering: Arc::new(self.ordering.clone()),
                opt,
                score,
            };
            BlockOptimization::new(strategy, self.ordering)
        }
        BestOptimization::NotFound => {
            let strategy = ExecutionStrategy::Operations {
                ordering: Arc::new(self.ordering.clone()),
            };
            BlockOptimization::new(strategy, self.ordering)
        }
    }
}
```

> **注意**：此代码已更新（`burn/crates/burn-fusion/src/search/block.rs:59-81` 和 `block.rs:240-260`）。旧版 `find_best_optimization_index` 返回 `Option<usize>`，现在返回 `BestOptimization` 枚举（`Found { index, score }` / `NotFound`）。`ExecutionStrategy::Optimization`（`burn/crates/burn-fusion/src/stream/store/base.rs:18`）新增了 `score: u64` 字段。构建器方法从 `build()` 改为 `finish()`。

`Block.optimize` 方法：
1. 使用 `find_best_optimization_index` 找到最佳优化构建器
2. 在该构建器上调用 `finish()`（旧版为 `build()`）以创建优化
3. 创建一个带有优化的 `ExecutionStrategy`
4. 返回带有策略和顺序的 `BlockOptimization`

#### 4. find_best_optimization_index

```rust
fn find_best_optimization_index<O>(
    optimizations: &mut [Box<dyn OperationFuser<O>>],
) -> BestOptimization {
    let mut best_index = BestOptimization::NotFound;
    let mut best_score = 0;

    for (i, optimization) in optimizations.iter().enumerate() {
        let properties = optimization.properties();

        // 分数为零比融合差。
        if properties.ready && properties.score > best_score {
            best_index = BestOptimization::Found {
                index: i,
                score: properties.score,
            };
            best_score = properties.score;
        }
    }

    best_index
}
```

> **注意**：此代码已更新（`burn/crates/burn-fusion/src/search/block.rs:240`）。函数签名从 `-> Option<usize>` 改为 `-> BestOptimization` 枚举。trait 从 `OptimizationBuilder` 改为 `OperationFuser`（`burn/crates/burn-fusion/src/backend.rs:121`）。

此函数：
- 检查所有优化构建器
- 找到准备好且分数最高的那个
- 返回其索引

#### 优化构建器的分数如何被填充

分数在注册过程中随着操作被添加到每个构建器而**增量**填充：

##### 1. 注册流程
当 `Block` 接收操作时调用：
- `Block.register()` → `Block.register_op()` → **`builder.fuse(operation)`** 对每个构建器（旧版为 `builder.register(operation)`）

##### 2. 注册期间的分数计算
每个优化构建器基于**它成功接受了多少操作**来计算分数：

```rust
fn properties(&self) -> FuserProperties {
    let ready = self.num_ops > 0;

    FuserProperties {
        ready,
        score: self.num_ops as u64,  // 分数 = 接受的操作数
    }
}
```

> **注意**：`OptimizationProperties` 已重命名为 `FuserProperties`（`burn/crates/burn-fusion/src/backend.rs:101`）。

##### 3. 注册期间的分数更新
当 `TraceOperationFuser.fuse()`（旧版为 `FuseOptimizationBuilder.register()`）被调用时：

```rust
fn fuse(&mut self, operation: &OperationIr) {
    // ... 操作类型检查 ...

    if !self.register_numeric::<i32>(ops) {
        self.status = FuserStatus::Closed;  // 构建器拒绝未来的操作
        return;
    }

    self.status = FuserStatus::Open;
    self.num_ops += 1;  // 这会增加分数！
}
```

> **注意**：`OptimizationStatus` 已重命名为 `FuserStatus`（`burn/crates/burn-fusion/src/backend.rs:93`），方法从 `register()` 改为 `fuse()`（`backend.rs:122`）。

##### 4. 不同构建器的不同评分策略

**TraceOperationFuser**（旧版为 FuseOptimizationBuilder）：分数 = `num_ops`（可融合的操作数）

**ReduceFuser**（旧版为 ReduceBuilder）：若有 reduce 操作，分数 = `base_score + 1`

```rust
fn properties(&self) -> burn_fusion::FuserProperties {
    let mut properties = self.builder.properties();

    if self.reduce.is_some() {
        properties.ready = true;
        properties.score += 1;  // 有 reduce 加一分
    } else {
        properties.ready = false;
    };

    properties
}
```

**MatmulFuser**（旧版为 MatmulBuilder）：分数 = `base_score + 1`（matmul 操作加分）

```rust
fn properties(&self) -> burn_fusion::FuserProperties {
    let mut properties = self.builder.properties();
    properties.score += 1;  // matmul 加分
    properties
}
```

##### 5. 最佳构建器选择
最终，`find_best_optimization_index` 选择分数最高的构建器。

##### 总结
分数在操作注册期间**增量填充**：
1. **每个操作**被注册到 block 中的所有构建器
2. **每个构建器**决定是否能处理该操作
3. **若接受**：`num_ops++` 增加分数
4. **若拒绝**：构建器状态变为 `Closed` 并停止接受操作
5. **专门构建器**（Matmul、Reduce）对其特定操作获得加分
6. 当 `Block.optimize()` 被调用时，**最佳构建器**基于最高分数被选中

#### 5. OperationFuser.finish（旧版为 OptimizationBuilder.build）

这是一个 trait 方法，由各种构建器实现。例如 `ElementWiseFuser.finish()`（旧版为 `ElementWiseBuilder.build()`）：

```rust
fn finish(&mut self) -> CubeOptimization<R> {
    let client = R::client(&self.device);
    let trace = self.builder.finish();
    let elementwise =
        ElemwiseOptimization::<R>::new(trace, client, self.device.clone(), self.len());

    CubeOptimization::ElementWise(elementwise)
}
```

构建器在其内部构建器（通常为 `TraceOperationFuser`，旧版为 `FuseOptimizationBuilder`）上调用 `finish()`。

#### 6-8. 路径中的后续构建步骤

构建链路转发经过 `TraceOperationFuser.finish()` → `TryTraceFuser.finish()` → `TraceFuser.finish()`（旧版为 `FuseOptimizationBuilder.build()` → `TryFuseBuilder.build()` → `FuseTraceBuilder.build()`），在此创建实际的 `FuseTrace`：

```rust
// TraceFuser::finish（旧版为 FuseTraceBuilder::build）— 在这里实际创建 FuseTrace
pub fn finish(&self, shape_ref: Vec<usize>) -> FuseTrace {
    let mut resources = self.resources.clone();
    let mut outputs = RegisteredTensors::default();
    let mut blocks = Vec::new();

    let mut register_block =
        |block: &FuseBlockBuilder, shape_ref: &Vec<usize>, offset: usize| {
            let (block, block_tensor_writes) =
                block.build(&self.resources, shape_ref.clone(), offset);
            blocks.push(block);

            let num_outputs = block_tensor_writes.len();
            for (ir, precision) in block_tensor_writes.into_iter() {
                outputs.insert(precision, ir);
            }

            num_outputs
        };

    let mut offset = 0;

    for (block, shape_ref) in self.blocks_previous.iter() {
        offset += register_block(block, shape_ref, offset);
    }
    register_block(&self.block_current, &shape_ref, offset);

    resources.outputs = outputs;

    FuseTrace { blocks, resources }
}
```

> **注意**：`OperationFuser` trait 上的 `build()` 方法已重命名为 `finish()`（`burn/crates/burn-fusion/src/backend.rs:125`）。注意 `FuseBlockBuilder::build()`（`burn/crates/burn-cubecl-fusion/src/engine/trace/block.rs:398`）仍保留 `build` 名称，这是不同的方法。

#### 9. FuseBlockBuilder.build（此方法保留 `build` 名称）

```rust
pub fn build(
    &self,
    resources: &FuseResources,
    outputs: &mut RegisteredTensors,
    buffers: &mut Vec<TensorId>,
) -> FuseBlock {
    let ops = self.ops.clone();
    let reads = self.reads.clone();
    let tensor_writes = self.tensor_writes(resources, buffers);

    let mut writes = self.writes.clone();

    for (tensor, precision) in tensor_writes
        .iter()
        .filter_map(|entry| entry.as_normal_tensor())
    {
        if let Some(local) = self.locals.get_any_precision(tensor.id) {
            let out_index = outputs.insert(*precision, tensor.clone());

            let ops = match writes.get_mut(&tensor.id) {
                Some(ops) => ops,
                None => {
                    writes.insert(tensor.id, Vec::new());
                    writes.get_mut(&tensor.id).unwrap()
                }
            };

            ops.push(FuseOp::Assign(UnaryFuseArgs {
                input: local,
                out: FuseArg::Output(out_index, *precision, LayoutInfo::Unknown),
            }));
        }
    }

    FuseBlock {
        settings: self.settings,
        ops,
        shape_ref: self.shape_ref.clone(),
        reads,
        writes,
    }
}
```

> **注意**：此代码已针对 burn 2026.05 源码更新（`burn/crates/burn-cubecl-fusion/src/engine/trace/block.rs:398`）。方法签名变更：旧版接收 `(resources, shape_ref, offset)` 并返回 `(FuseBlock, RegisteredTensors)`，现在接收 `(resources, outputs, buffers)` 并直接返回 `FuseBlock`。tensor writes 现在在方法内计算而非作为单独返回值。

#### FuseTrace 如何产生

总结产生 `FuseTrace` 的过程：
1. **Block 收集**：`StreamOptimizer` 收集操作的 block
2. **Block 优化**：每个 block 使用最佳可用构建器被优化
3. **构建器选择**：最佳构建器基于分数被选出
4. **Trace 构建**：选中的构建器通过以下方式构建 trace：
   - 处理每个 block 以创建 `FuseBlock`
   - 收集所有 block 及其资源
   - 用 block 和资源创建 `FuseTrace`

关键是，到 `finish()`（旧版为 `build()`）被调用时，所有操作已经与构建器注册。`finish()` 方法不注册新操作——它处理已注册的操作以创建优化的执行计划。

最终产生的 `FuseTrace` 包含：
1. 一个 `FuseBlock` 列表，每个都有其操作、读和写
2. 资源，包括输入、输出和标量
3. 高效执行融合操作所需的一切

此 trace 可由运行时执行以高效地进行融合操作。

---

## CubeCL Kernel 如何生成

（本节省略，仍在写作中）

```rust
$ cargo expand --manifest-path crates/burn-cubecl-fusion/Cargo.toml --lib elemwise::optimization
```

据我所知，融合 kernel 不是在运行时动态生成的——它们是**静态定义和预编译的**。它们像"通用模板"一样可以处理任意序列的操作。

例如，下方的 `elemwise_fuse` kernel 似乎可以处理任何逐元素操作序列：

```rust
#[cube(launch_unchecked, address_type = "dynamic")]
fn elemwise_fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: &FuseBlockConfig,
) {
    let values = Registry::<Arg, Line<f32>>::new();
    let args = comptime![Sequence::<Arg>::new()];
    let pos = ABSOLUTE_POS;

    let mut locals = init_locals(inputs, outputs, config);
    let length = ref_len(inputs, outputs, &locals, config);

    if pos < length {
        fuse_on_write::<f32>(inputs, outputs, &mut locals, pos, values, args, config)
    }
}
```

> **注意**：此代码已更新（`burn/crates/burn-cubecl-fusion/src/optim/elemwise/optimization.rs:122`）。`#[cube(launch_unchecked)]` 现在需要 `address_type = "dynamic"` 参数。

据我理解，流程大致如下：
1. 一个 `FuseTrace` 被记录，包含如下操作序列：

```rust
ops: [
    Assign(input -> local_0),
    Mul(local_0, scalar_0 -> local_1),
    Add(local_1, scalar_1 -> local_2),
    Tanh(local_2 -> local_3),
]
```

2. 在 kernel launch 时，此序列作为 `FuseBlockConfig` 的一部分传入。

3. 实际 kernel 使用一个通用的 fuse 函数，看起来像：

```rust
#[cube]
fn fuse(..., #[comptime] config: &FuseBlockConfig) {
    #[unroll]
    for index in 0..config.ops.len() {
        let op = comptime! { config.ops.index(index).clone() };
        match op {
            FuseOp::Mul(op) => mul(...),
            FuseOp::Add(op) => add(...),
            FuseOp::Tanh(op) => tanh(...),
            // 等等。
        }
    }
}
```

所以这里没有运行时代码生成——只有通过 `#[comptime]` 和 `#[unroll]` 的编译期特化（或者说 JIT 编译特化）。

---

## 杂项：

### 处理不可融合的操作

当操作无法融合时：
1. 处理器仍然通过相同的流处理它们
2. `Explorer` 将无法找到优化
3. 操作将使用 `ExecutionStrategy::Operations` 策略单独执行

```rust
pub(crate) enum ExecutionStrategy<O> {
    /// 找到了优化，因此应执行优化。
    Optimization {
        opt: O,
        ordering: Arc<Vec<usize>>,
        score: u64,
    },
    /// 未找到优化，每个操作应单独执行。
    Operations { ordering: Arc<Vec<usize>> },
    /// 多个执行策略的组合。
    Composed(Vec<Box<Self>>),
}
```

> **注意**：此代码已更新（`burn/crates/burn-fusion/src/stream/store/base.rs:18`）。`Optimization` variant 新增了 `score: u64` 字段。

### 跨流 Fusion

跨多个 stream 的操作不会直接融合。系统被设计为：
1. 保持 stream 独立以便并发执行
2. 处理 stream 之间共享的 tensor
3. 在必要时同步 stream

`MultiStream` 类型通过 `resolve_streams` 和 `merge_streams_timelines` 等方法管理这些交互，确保在 stream 共享 tensor 时有正确的顺序，但不尝试跨不同 stream 融合操作。

当一个 tensor 在 stream 之间共享时，系统通过在需要时排空依赖的 stream 来确保正确的执行顺序，但融合仅在单独的 stream 内部发生。

### 搜索算法
- 正确的 tile 大小是多少？
- 展开此循环是否更好？

---

> **翻译说明**：本文翻译自 nihalpasham 的 GitHub Gist ["How does automatic kernel fusion work in burn"](https://gist.github.com/nihalpasham/fc128f074e20d880bfd97198c2ac784b)（2025.07）。所有代码块均已针对 burn 源码 2026.05 版本更新（commit `45267d0ae`）。主要变更包括：
> - `Tensor<B, D, K>` → `Tensor<const D: usize, K = Float>`（Backend 泛型参数已移除，改为运行时 `Device` dispatch）
> - `MutexFusionClient` → `GlobalFusionClient`（`DeviceHandle` + `DeviceService` 模式）
> - `WgpuRuntime` 泛型参数从三元素类型改为单编译器类型
> - `OptimizationBuilder` → `OperationFuser`、`FuseOptimizationBuilder` → `TraceOperationFuser` 等 trait/实现重命名
> - `FuseBlockBuilder` 和 `OperationQueue` 字段调整
> - `cube(launch_unchecked)` → `cube(launch_unchecked, address_type = "dynamic")`
> - `build()` 方法 → `finish()` 方法（`OperationFuser` trait）
