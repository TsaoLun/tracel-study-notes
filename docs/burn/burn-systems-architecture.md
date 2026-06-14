# Burn 技术栈全景：从 Rust 代码到 GPU 执行的全链路

本文用一行代码贯穿全程：

```rust
let x = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
let z = (x.clone() * 2.0 + 1.0).tanh();
z.backward();
```

这四行触发了一套完整的系统链路：操作排队与融合、自动调参、Rust 宏到 GPU 二进制生成、以及自动微分。本文展开每一步的系统设计——为什么这么设计，和主流方案的区别，以及设计的限制。

先看总体架构：

```
Autodiff<Fusion<CubeBackend<WgpuRuntime<AutoCompiler>>>>
  ↑ 装饰器模式——每层包裹下一层，各层独立

  Autodiff    ← 梯度跟踪、图构建、反向传播
  Fusion      ← 操作排队、融合优化、执行调度
  CubeBackend ← CubeCL 运行时 + 融合 fuser 注册
  WgpuRuntime ← wgpu 设备抽象、内存管理、kernel 编译与启动
```

回到那行代码。它的 `Tensor` 定义在 `burn/crates/burn-tensor/src/tensor/api/base.rs:74`：

```rust
pub struct Tensor<const D: usize, K = Float>
where K: Basic {
    pub(crate) primitive: BridgeTensor,
    _kind: PhantomData<K>,
}
```

`Tensor` 只有两个泛型参数——维度 `D` 和元素类型 `K`。**Backend 不是编译期泛型**。旧版 burn（2025 年中及之前）定义为 `Tensor<B, D, K>`，commit `dbf03c516` 将 Backend 移除，改为通过运行时 `Device` 确定。内部的 `BridgeTensor`（`burn/crates/burn-tensor/src/bridge/kind.rs:91`）根据 `Device` 将操作路由到对应的后端实现。Backend 类型本身仍是可组合的——`Wgpu` 别名展开为 `Fusion<CubeBackend<WgpuRuntime<AutoCompiler>>>`——但这种组合性体现在 `Device` 类型层面，而非 `Tensor` 的泛型参数。

---

## 第一部分：Kernel Fusion —— 为什么三个操作需要一个 Kernel

### 问题：Kernel Launch 有可测量的开销

在 NVIDIA A100 上，一次 CUDA kernel launch 的软件开销约 5-10 μs（隐式同步、grid/block 参数计算、驱动调度）<sup>*</sup>。一个 memory-bound 的 element-wise op（如 `tensor_a + tensor_b`）在 1024×1024 f32（约 4MB）数据上的执行时间约 3-5 μs。这意味着：

```
launch overhead ≈ 2× compute time
```

三个连续的 element-wise op（`* 2.0`、`+ 1.0`、`tanh`）分别执行：

```
总耗时 ≈ 3 × (kernel launch) + 3 × (compute)
       ≈ 3 × 7μs + 3 × 4μs  ≈  33μs
```

融合为一个 kernel：

```
总耗时 ≈ 1 × (kernel launch) + 1 × (12μs compute)  ≈  19μs
```

**消除 launch 开销可带来 30-50% 的端到端延迟改善。** 融合同样减少了全局内存读写——中间结果不再写回 GPU DRAM 再读出来，而是停留在寄存器或共享内存中。

| 指标 | 无融合 | 有融合 | 改善 |
|------|--------|--------|------|
| Kernel launch 次数 | 3 | 1 | -67% |
| 全局内存读写次数 | 6（每 op 读+写） | 2（读输入+写输出） | -67% |
| 估算总延迟 | ~33 μs | ~19 μs | -42% |

> <sup>*</sup> 该数字来自 CUDA 社区常见估算，实际取决于驱动版本、CUDA 版本和系统负载。可复验：`nvprof --print-gpu-trace` 或 CUPTI API。

### 融合的代价

融合不是无代价的。把十个 op 融合成一个 kernel：
- 寄存器压力增大，可能降低 occupancy（同时活跃的 warp 数）
- 某些 op 需要不同的 tile 大小才高效，融合后被迫折中
- 编译/探索融合方案本身需要时间

融合是否有净收益取决于具体的硬件、数据规模和操作组合。一个融合方案在 A100 上可能是净收益，在 RTX 3060 上可能因为寄存器预算不同而劣于分别执行。

### 三种融合范式

| 范式 | 代表 | 触发方式 | 优化成本 | 缓存 |
|------|------|---------|---------|------|
| 静态融合 | XLA | 编译期基于 HLO 图规则合并 | 零（规则固化） | 无需缓存 |
| JIT 编译融合 | Triton | `@triton.jit` 运行时编译 | 高（首次 autotune 秒级） | 编译缓存 |
| 惰性队列融合 | Burn | 操作排队，同步点触发 | 首次探索 ms 级，缓存命中 ~0 | `ExecutionPlanStore` |

Burn 选择的路线：**操作不立即执行，而是排队。在需要结果时（同步点），对队列中的操作序列做融合优化，生成并缓存优化方案。** 这和 PyTorch 的 eager 模式不同（每个 op 立即 launch kernel），也不同于完全的静态编译。核心差异在于 **overhead 发生时机**：静态融合在编译期、JIT 在运行期首次执行、Burn 在同步点时探索但后续命中缓存。

### 惰性执行：操作如何排队

回到代码。`(x * 2.0 + 1.0).tanh()` 触发了三次 `OperationIr` 的生成和入队。Burn 的 `Fusion` 后端包装了内层后端（`CubeBackend`），为每个 tensor 操作生成 IR 描述并推入当前 stream 的 `OperationQueue`：

```rust
// burn/crates/burn-fusion/src/stream/queue/base.rs:13
pub struct OperationQueue<R: FusionRuntime> {
    pub(crate) global: Vec<OperationIr>,    // 精确 tensor ID 和 shape
    pub(crate) relative: Vec<OperationIr>,  // 相对 tensor ID，用于模式匹配
    pub(crate) converter: OperationConverter,
    pub(crate) operations: Vec<UnfusedOp<R>>,
    pub(crate) variables: HashMap<TensorId, TensorStatus>,
}
```

`global` 包含精确的 tensor ID 和 shape——执行时需要。`relative` 将 tensor ID 替换为局部的相对编号——用于发现"这个操作序列和之前见到的某个序列是否结构相同"。

我们的三个操作入队后，队列状态为：
```
global:   [Init(tensor_1), MulScalar(tensor_1, 2.0), AddScalar(temp, 1.0), Tanh(y)]
relative: [Init(#0),       MulScalar(#0, #s0),        AddScalar(#1, #s1),      Tanh(#2)]
```

`relative` 中的 `#0`、`#1` 等是 `OperationConverter` 分配的局部编号——消除了具体 tensor id 的差异，使得不同执行中的相同操作模式（如 `x*2.0` 和 `y*2.0`）映射到同一个 relative 操作描述。

### 触发点：是什么启动了执行

操作入队后不执行。三种情况触发 drain（`MultiStream::drain()`）：

1. **读取 tensor 数据**：`println!("{}", z)` → `Tensor::fmt()` → `display_fmt_recursive` → `display_fmt_inner` → `into_data_async()` → `FusionServer::float_data()` → `drain_stream()`（`burn/crates/burn-fusion/src/server.rs:102`）
2. **跨 stream 共享**：tensor clone 到另一线程，源 stream 在共享前 drain
3. **显式 sync**：`client.sync(|| ...)`（`burn/crates/burn-fusion/src/client.rs:142`）

在我们的示例中，`z.backward()` 需要前向结果来计算梯度，所以 backward 调用间接触发了前向的 drain。

### Stream 与 MultiStream：并发安全的操作隔离

一条 `Stream`（`stream/multi.rs:261`）是一个独立的惰性操作流：包含 `OperationQueue`、`Processor` 和 `cursor`。它的核心不变式是：**队列中每个操作引用的 tensor 都被假定属于同一个 stream**。这保证了单 stream 内操作可以简单串联执行——每个 tensor id 可以从同一个 `HandleContainer` 中解析。

但 `FusionTensor` 是 `Send + Clone` 的——用户可以把 tensor 传送到另一个线程。接收方提交的操作引用了**其他 stream 的 tensor**。`MultiStream`（`stream/multi.rs:102`）就是解决这个问题的：它维护 `HashMap<StreamId, Stream<R>>`，管理多条并发流。

跨流共享使用 **alias 而非 merge**。当 `FusionTensor::clone` 检测到 `self.stream != StreamId::current()`：

1. **物化源 tensor**：如果源还没被注册（产出它的 op 仍 pending），先同步 drain 源 stream（`multi.rs:180`）
2. **创建 alias**：在目标 stream 分配新的本地 id，将源 tensor 的 backend handle clone 过去（`multi.rs:185`）

clone handle 是 `Arc` 语义——两个 id 指向同一块 GPU 内存，refcount 共享。释放时各自 drop，最后一方释放底层 buffer。

**融合只在单 stream 内进行**。跨流操作在共享前 drain（`ExecutionMode::Sync`），保证数据一致性，但不跨流融合。

### 融合引擎：OperationFuser 的竞争探索

drain 触发时，`Processor`（`stream/execution/processor.rs:11`）对队列中的连续操作做融合。核心机制是 **`OperationFuser` 的竞争**。

Fusion 后端通过 `FusionRuntime::fusers()` 注册具体的优化器（`burn/crates/burn-cubecl/src/fusion.rs:144`）：

```rust
fn fusers(device) -> Vec<Box<dyn OperationFuser<Self::Optimization>>> {
    vec![
        Box::new(ElementWiseFuser::new(device)),
        Box::new(MatmulFuser::new(device)),
        Box::new(ReduceFuser::new(device, ReduceSettings::Always)),
        Box::new(ReduceBroadcastedFuser::new(device)),
    ]
}
```

`OperationFuser` trait（`burn/crates/burn-fusion/src/backend.rs:121`）定义：

```rust
pub trait OperationFuser<O>: Send {
    fn fuse(&mut self, operation: &OperationIr);  // 接受/拒绝一个操作
    fn finish(&mut self) -> O;                     // 产生最终优化方案
    fn reset(&mut self);
    fn status(&self) -> FuserStatus;               // Open/Closed
    fn properties(&self) -> FuserProperties;       // score + ready
}
```

每个 fuser 在收到 IR 操作时：
- **accept**：操作可融合 → `num_ops++`（分数增加）
- **reject**：操作无法融合 → `status = FuserStatus::Closed`（不再接受后续操作）

Block 被优化时，`find_best_optimization_index()`（`search/block.rs:240`）选择**分数最高的 fuser** 的方案执行。对于我们的示例，三个操作都是 element-wise——`ElementWiseFuser` 全盘接受，score=3，没有竞争者。

`fusers()` 返回 `Vec<Box<dyn OperationFuser<O>>>`。使用 trait object 的原因是：四种 fuser 是不同具体类型，需要共存于同质容器；每个 Block 通过 `clone_dyn()` 获得所有 fuser 的独立可变副本；burn-fusion 框架层不知道 cubecl 层有哪些 fuser 实现——`Box<dyn>` 使新增 fuser 不需改动框架代码。

一个 Block 持有全部 fuser 的副本。操作注册时遍历所有 fuser：`ElementWiseFuser` 对 matmul 返回 `Closed`，`MatmulFuser` 对 element-wise 返回 `Closed`——**这是竞标，不是分工**。最终只有一个胜出。

Block 的划分由 `Block::register()` 中一行判断决定（`search/block.rs:140-148`）：

```rust
for node in operation.nodes() {             // 操作引用的所有 tensor
    if self.ids.contains(&node.id) { break; } // tensor 是否已在 block 中
}
// 无交集 → NotPartOfTheGraph；有交集 → Accepted
```

规则：操作涉及的任意 tensor ID 已出现在 block 中，操作即划入该 block。无交集则 `on_new_block()` 创建新 block。这是连通分量的增量构建。

这个规则对 tensor 依赖优先于 fuser 边界。当 `*2.0` 和 `Matmul(z)` 通过中间 tensor 连接时，两者同属一个 Block，但一个偏向 ElementWiseFuser、一个偏向 MatmulFuser——最终只有一个 fuser 胜出，另一个 op 失去最优处理。`BlocksOptimizer::merging_pass()` 有合并拒绝机制（`Block::merge()`，`block.rs:100-118`）避免破坏已 ready 的融合，但它只在 block 间无 tensor 交集时有效——同 block 内的融合冲突当前没有切分机制。

### 探索与缓存

找到的融合方案被缓存在 `ExecutionPlanStore` 中——跨 stream 共享。下次相同操作序列被 drain 时，`Policy` 直接命中缓存，跳过探索。

缓存匹配使用 **trigger 机制**：一个 plan 在"前 N 个 op 匹配 + 后续 op 出现在 trigger 列表中"时被激活。与精确序列比对不同，trigger 机制允许同一个融合方案（如 `*2.0 + 1.0 + tanh`）在前两个 op 匹配后立即执行，无论第三个 op 是否是 `tanh`（只要在 trigger 集合中）。

### 从融合方案到 GPU Launch

`FuseTrace` 是融合引擎的输出：包含操作序列的 `Vec<FuseBlock>` 和资源描述 `FuseResources`（输入/输出 buffer、标量常量、shape 信息）。将它变成 GPU 可执行代码的是 `FuseTraceLauncher::launch()`（`burn-cubecl-fusion/src/engine/launch/base.rs:36`）。

`LaunchPlan::new(&blocks)` 创建一个空的执行计划。随后三个 planner 顺序填充：

1. **`InputPlanner`**（`engine/launch/input.rs:15`）——遍历 blocks 的 `reads`，确定哪些 tensor 需要绑定为 GPU buffer 输入，怎么读（全局内存 / 标量常量 / swap_dims 转置读取）。这一步填充 `plan.inputs`。

2. **`OutputPlanner`**（`engine/launch/output.rs:28`）——遍历 blocks 的 `writes`，为每个输出 tensor 分配 GPU buffer（或复用已有 buffer），记录写入方式和偏移。这一步填充 `plan.outputs`。

3. **`VectorizationPlanner`**（`engine/launch/vectorization/planner.rs:31`）——根据输入/输出的对齐和操作语义，为每个 buffer 选择向量化宽度（f32 对齐到 16 字节时用 vec4，否则 vec2 或标量）。这一步修改 `plan` 中已有条目的向量化标记。

三者共享同一个 `&mut LaunchPlan`——每一步在前一步的基础上累积信息，最终产生一个完整的 buffer binding 描述。最后由 `LaunchPlanExecutor::execute()`（`engine/launch/executor.rs:46`）将 plan 转化为 wgpu 的 `BindGroup` + `ComputePipeline` + `dispatch_workgroups` 调用。

### GPU 内存管理：Page / Slice 模型

在融合执行之前，tensor 数据必须存在于 GPU 内存中。Burn/CubeCL 的内存管理采用类似操作系统的 **Page / Slice** 模型，目的是减少昂贵的 `wgpu::Device::create_buffer()` 调用。

**Page（页）**：一次 `create_buffer()` 创建的大块 GPU buffer（如 32MB、128MB）。创建页涉及内核调用和驱动注册。

**Slice（切片）**：页内部的子区域。多个小 slice 共享同一个 page，通过 offset 定位。

**三层分配策略**（`cubecl/crates/cubecl-runtime/src/memory_management/memory_manage.rs:451`）：

**第 1 层——PersistentPool**：按精确大小通过 HashMap 索引的预分配 slice。命中后 HashMap 查找 O(1)，用于永不释放的分配（如模型权重）。同一大小桶内如有多个空闲 slice 则线性扫描；在典型场景（每种权重只分配一次）中桶内通常只有一个。

**第 2 层——DynamicPool 复用**：内存池维护多个 page，每个 page 内有可变大小的 slice。调用 `MemoryPage::try_reserve()` 扫描空闲区域，`MemoryPage::coalesce()` 合并相邻空闲 slice 减少碎片。找到一个足够大的空闲 slice → 切出所需大小 → 返回——**零 GPU 分配开销**。

**第 3 层——创建新页**：所有现有页都无法满足时，调用 `WgpuStorage::alloc()` → `wgpu::Device::create_buffer()`。最昂贵的路径，但被前两层大幅降低了调用频率。

CubeCL 维护三个独立的内存池（`WgpuMemManager`，`cubecl/crates/cubecl-wgpu/src/compute/mem_manager.rs:19`）：

| 池 | 用途 | Buffer Usage |
|----|------|-------------|
| `memory_pool` | 主 GPU 内存 | `STORAGE \| COPY` |
| `memory_pool_staging` | CPU 可读暂存 | `MAP_READ \| COPY_DST` |
| `memory_uniforms` | Uniform Buffer | `UNIFORM \| STORAGE \| COPY_DST` |

分离的原因是 wgpu buffer 的 usage 标志在创建时确定且不可变——不同用途的 buffer 不能混用。

`memory_pool_staging` 和 `memory_uniforms` 使用 `ExclusiveMemoryPool`（`cubecl/crates/cubecl-runtime/src/memory_management/memory_pool/exclusive_pool.rs:17`）：每个页只有一个分配。已释放的页不立即销毁，而是留在页表中。`cleanup()` 方法遍历所有页：连续被标记 free 达到 5 次（`ALLOC_AFTER_FREE`）的页才调用 `dealloc()`。这个双向量轮转的滞留池模式使频繁复用的页保留，偶尔使用的页被淘汰。

### 限制

- **单 stream 融合**：跨线程操作不融合
- **单 producer 限制**：操作要求在同一个 stream 上；跨 stream 的读先 drain，破坏融合机会
- **首次探索开销**：新的操作序列首次执行需要探索（~1-5ms）
- **动态 shape 缓存命中**：shape 变化时 relative IR 匹配可能失败

---

## 第二部分：Autotune —— 同一个 Kernel，不同的最优参数

### 问题

矩阵乘法 `A × B`。当 A = [1, 4096]，B = [4096, 4096]——这是 matvec。当 A = [4096, 4096]，B = [4096, 4096]——这是 gemm。同一个操作，最优的 tile size、workgroup 大小、向量化宽度完全不同。

向量化也是如此。融合 kernel `[Assign, Mul(2.0), Add(1.0), Tanh]` 在 [2048, 64] shape 下利于 vec4 coalesced 读取，在 [64, 2048] shape 下可能只能用 vec2 甚至标量。

**不存在一套参数在所有场景下最优**。

### CubeCL 的路：策略枚举 vs Triton 的参数网格

Triton 的 `@triton.autotune` 让用户定义参数网格——`BLOCK_SIZE_M = [16, 32, 64, 128]`、`num_warps = [2, 4, 8]`——对每个组合生成 kernel、做 exhaustive search。候选数爆炸：5 × 4 × 3 × 3 = 180。

CubeCL 走的是完全不同的路线。Kernel 作者**枚举一组具体的实现策略**，每个策略是一个闭包，包含固定的参数。这是基于一个关键的观察：GPU kernel 参数不是正交的。double buffering 只在配合 tile gemm 时才有效。TMA 只在 H100 上有效。向量化宽度受 shared memory 大小和寄存器预算的联合限制。正交网格产生大量无效组合；枚举只包含验证过的组合。

对于 matmul（`burn/crates/burn-cubecl/src/kernel/matmul/tune/base.rs`），注册了 30 个候选：

| 类别 | 候选数 | 内容 |
|------|--------|------|
| Naive fallback | 1 | 保障可用性，始终第一位 |
| GEMV 变体 | 4 | DoubleVecMat、SimpleVecMat、Gemm、GemvUnitPerpendicular |
| Unit 变体 | 4 | SimpleUnit/DoubleUnit × Max/Min tile size |
| Gemm no-stage | 1 | 无多级流水线的简单 gemm |
| 加速变体 | 14 | Cmma/Mma × multi_rows × double_buffering × partition_k 组合 |
| TMA 变体 | 6 | SimpleTma/SpecializedTma × Cmma/Mma，H100 特性 |

对 reduce（`burn/crates/burn-cubecl/src/kernel/reduce/tune.rs`）：6 个候选（Unit/Plane/Cube × 两种向量化模式）。对 sum：7 个候选（链式 + 6 种向量化宽度）。

### 搜索策略：优先级分组 + 批量提前终止

对 30 个候选逐一 benchmark（warmup 3 + sample 10）需要 390 次 kernel launch。在 A100 上约 8-15ms。对延迟敏感的推理场景不可接受。

CubeCL 用**优先级分组（`TuneGroup`） + 批量提前终止**：

```rust
// cubecl/crates/cubecl-runtime/src/tune/base.rs:62
let accelerated_group = TuneGroup::new("accelerated", |key: &MatmulKey| -> i8 {
    match key.kind {
        MatmulKind::Large | MatmulKind::General => 3,  // 最高优先
        _ => 1,
    }
});
```

优先级分组的工作机制：

1. **分级优先**：高优先级组先执行。rank=3 的候选在 rank=1 的候选之前 benchmark
2. **组内子优先级**：同一个组内每个候选有自己的优先级。负值跳过
3. **提前终止**：某一批中有至少一个候选成功 → 后续分组不执行

完整的 autotune walkthrough，以 matmul [1024, 4096, 4096] f32 为例：

```
输入: MatmulAutotuneKey { shape=[1024, 4096, 4096], dtype=F32 }
  ↓ TunePlan::new(key) → 计算所有候选的优先级

第 1 批: accelerated 组（组优先级 3）
  候选: cmma_multi_rows(3), cmma_double_buffer(3), mma_multi_rows(2), ...
  ↓ warmup×3 + sample×10 对每个候选
  cmma_multi_rows → 45μs, score=43.2 ✓
  cmma_double_buffer → 52μs, score=49.1 ✗
  至少 1 个成功 ✓ → 提前终止！
  ↓ 后续 unit 组（组优先级 2）、gemv 组、naive 全部跳过

结果: cmma_multi_rows @ ~45μs → 缓存到 TuneCache
下次 [1024, 4096, 4096] f32 matmul → 缓存命中 → 零 overhead
```

评分函数（`cubecl/crates/cubecl-common/src/benchmark.rs:124`）：

```rust
const ALPHA: f64 = 0.8;
let base = min × ALPHA + median × (1.0 - ALPHA);
let cv = 1.0 + sqrt(variance) / (1.0 + mean);
let score = (base × cv) as u64;  // 越低越好
```

混合了最小值（80%）和中位数（20%），乘以变异系数惩罚。不稳定的 kernel（方差大）即使最快也被惩罚——在 GPU 计算中，方差大通常意味着容易触发 throttle 或 cache miss。测量使用 GPU 时间戳（`cubecl/crates/cubecl-wgpu/src/compute/timings.rs`），Metal 上受限于最多 28 个活跃查询集，用 round-robin 溢出。

### 缓存密钥：Anchor 量化

如果每种 shape 都单独 autotune，缓存永远无法命中——shape 空间是连续的。CubeCL 用 **anchor 量化** 将 shape 空间映射到离散桶。

`anchor(value, base)` 函数（`cubecl/crates/cubecl-runtime/src/tune/util.rs:16`）将值向上取整 (ceil) 到最近的 `base^n`——不是除以，是舍入到离满足 `value ≤ base^n` 的最小的 n。

```rust
// util.rs:28
let exp = (value as f64).log(base).ceil();
let power = base.powf(exp).ceil() as usize;
```

| AutotuneLevel | factor | base 缩放 | 效果 |
|---------------|--------|----------|------|
| Minimal | 1.25 | base × 1.25 | 粗桶——更少缓存条目，更快命中 |
| Balanced | 1.0 | base × 1.0 | 默认 |
| Extensive | 0.75 | base × 0.75 | 细桶——更精确 |
| Full | — | 直接返回 | 每种 shape 独立 key |

shape [1024, 2048] 在 Balanced 级别可能和 [1024, 2050] 映射到同一个桶——2050 向上取整到 2048 的下一个 2^n。融合 autotune 的 key 还包含 `num_out_buffers` 和 `num_ops`，同样做 anchor 量化（`burn/crates/burn-cubecl-fusion/src/optim/matmul/tune.rs:22`）：

```rust
pub struct FusedMatmulAutotuneKey {
    matmul_key: MatmulAutotuneKey,
    #[autotune(anchor)] num_out_buffers: usize,
    #[autotune(anchor)] num_ops: usize,
}
```

### 缓存架构

`TuneCache<K>` 同时维护内存缓存和持久化缓存（`tune/tune_cache.rs:83`）：

```
TuneCache<K>
  ├── in_memory: HashMap<K, CacheEntry>
  │     ├── Done { checksum: Match, fastest_index: usize }
  │     └── Pending   (其他线程正在对该 key 做 autotune)
  └── persistent: Cache<(key, checksum), (fastest_index, results)>
        └── {root}/autotune/{version}/{device_id}/{tuner_name}.json.log
```

checksum 是所有 tunable 名称拼接后的 MD5（`tune/operation.rs:96`）。如果 kernel 作者增加或删除候选策略，checksum 不匹配，全部持久化缓存自动失效——保证不会使用过时的"最快第 3 个"索引。持久化缓存在进程间共享，首次 autotune 后所有后续进程直接命中磁盘。

### 容错

- **第一个候选始终是 fallback**（注释约定，如 `base.rs:164`）：naive kernel 在任何 shape/dtype 下一定可运行
- **配置级 fallback**：`ConvStrategy::default()` 在 `cfg(feature = "autotune")` 为假时返回 `Direct`——autotune 可作为编译期 feature 关闭
- **fusion fallback**：如果 autotune 选中的 candidate 实际执行失败（benchmark 时通过但运行时不同），`tune_fused` 在原始 context 上调用 `opt.execute_fallback(ctx)` 回退到默认执行路径
- **`autotune-checks` feature**：开启后运行所有候选并验证输出一致性，用于开发阶段

### 与 Triton Autotuner 对比

| 维度 | CubeCL | Triton |
|------|--------|--------|
| 搜索空间定义 | 枚举手写策略闭包 | 参数网格 |
| 候选数 | 6--35 | 数十到数百 |
| 搜索算法 | 优先级分组 + 批量早停 | exhaustive 或启发式/遗传 |
| 无效组合 | 零（手写验证） | 可能存在（编译失败或 OOM） |
| 缓存 key | anchor 量化的 shape/type/fusion combo | 输入 shape 和 dtype |
| key 精度控制 | `#[autotune(anchor)]` + AutotuneLevel 四级 | 无（形状直接作为 key） |
| 缓存失效 | checksum（tunable 名称变化 → 自动失效） | 手动清理或版本变更 |
| 首次延迟 | 低（候选少 + 早停） | 可能很高（全网格搜索） |
| 测量 | GPU 时间戳，3 warmup + 10 sample | CUDA events，可配置 |
| 评分 | min×0.8+median×0.2，CV 惩罚 | 中位数/均值 |
| 硬件 | CUDA+Metal+Vulkan+WGPU+WASM | 仅 CUDA（ROCm 实验性） |
| Fusion 感知 | 原生：fork context 隔离 benchmark | 无内置融合 autotune |
| 策略表达力 | 低（需手写每个候选） | 高（改参数即可生成新候选） |

最根本的差异：**CubeCL 信任 kernel 作者枚举高质量策略**——用人力换搜索范围缩小。**Triton 信任编译器对任何参数生成正确代码**——用搜索换人力投入减少。

### 限制

- **覆盖面由作者决定**：没有合适策略的 shape 只能选次优
- **冷启动代价**：首次执行的 benchmark 开销（390 次 launch ≈ 8-15ms），缓存命中后消失
- **anchor 误差**：精确 shape 的最优策略和锚定 shape 的最优策略可能不同
- **单候选场景**：只有一个候选时 benchmark 跳过——但如果那个候选不是最优，autotune 无法改善

---

## 第三部分：JIT 编译管线 —— 从 Rust 宏到 GPU 二进制

### 问题：如何让 Rust 函数在 GPU 上运行

autotune 选中了 `ElemwiseOptimization`，它的 `FuseTrace` 携带了操作序列 `[Assign, Mul(2.0), Add(1.0), Tanh]`。现在需要执行这个 kernel——但它仍然是一个 Rust 函数。CubeCL 的 JIT 管线将其转换为 GPU 可执行代码。

### 第一步：`#[cube]` 过程宏 —— 从 Rust 到 IR

```rust
#[cube(launch_unchecked, address_type = "dynamic")]
fn elemwise_fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: &FuseBlockConfig,
) { ... }
```

`#[cube]` 属性宏（`cubecl/crates/cubecl-macros/src/lib.rs:56`）展开时做三件事：

1. **保留原函数**（作 AST 变换：移除辅助函数、替换 `define!` 宏）
2. **生成 `expand()` 函数**——原函数的"IR 化"版本，每次调用 `scope.register()` 生成一条 `cubecl_ir::Instruction`
3. **生成 `launch()` / `launch_unchecked()` 包装**——连接 `KernelBuilder` → `KernelLauncher` → `ComputeClient`

Rust 表达式被一对一映射为 IR 操作：

```
Rust:  out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] * scalar + scalar
  ↓ 宏解析
IR:    Index(lhs, AbsolutePos)  →  var#1
       Mul(var#1, Constant(2.0)) →  var#2
       Add(var#2, Constant(1.0)) →  var#3
       Store(var#3, out@AbsolutePos)
```

`ABSOLUTE_POS` 成为 `VariableKind::Builtin(Builtin::AbsolutePos)`。`#[comptime]` 参数被标记为 `is_const: true`——在宏展开阶段保持 Rust 值而非转变为 IR 变量。`comptime!` 宏（`lib.rs:190`）直接将内容作为 Rust 代码传入，绕过 IR 生成。这些 comptime 值的哈希进入 `KernelId::info`——不同操作序列产生不同的 `KernelId`，触发不同的编译或缓存命中。

`#[unroll]` 标记的 for 循环在宏代码生成阶段展开（`generate/expression.rs:259`），不是 IR 层面的优化 pass。循环边界必须是 comptime 常量，宏在 Rust 的 `for` 循环中为每次迭代调用一次 body 闭包——循环体被物理复制 N 次。这对融合 kernel 至关重要：`FuseBlockConfig.ops` 的操作序列在编译期展开为直线代码。

### 第二步：IR 设计 —— 嵌套 Scope 树而非 CFG

CubeCL 的 IR 使用**嵌套 Scope 树**（`cubecl/crates/cubecl-ir/src/scope.rs:36`）：

```rust
pub struct Scope {
    pub instructions: RefCell<Vec<Instruction>>,
    pub return_value: Option<Variable>,
    pub locals: RefCell<Vec<Variable>>,
    pub const_arrays: RefCell<Vec<(Variable, Vec<Variable>)>>,
    pub global_state: GlobalState,
}
```

每个 `Branch`（`if`、`loop`、`for`）携带自己的子 `Scope`。树形表示**不需要 phi 节点**——每个分支有自己的 scope，合并值通过显式变量处理。代价是复杂控制流（loop 内多 break 路径并合并值）的生成代码不够紧凑。收益是**编译器简单**——每个后端编译器递归遍历 Scope 树即可生成嵌套着色器代码，无需处理 CFG 的支配树和 SSA 重建。

变量系统使用 `Versioned { id, version }` 变体（`variable.rs:122`）实现 SSA 式值版本控制：每次赋值产生新版本，旧版本视为不可变。`LocalConst` 用于从未被重绑定的变量。`Constant` 是 comptime 值在 IR 中的体现。

IR 操作全集（`operation.rs:29`）覆盖了 GPU 计算原语：

- `Arithmetic`（Add/Mul/Fma/Sin/Cos/Exp/Log/Tanh/Sqrt...）
- `Branch`（If/IfElse/Loop/RangeLoop/Switch/Return/Break）
- `Synchronization`（sync_cube/sync_plane）
- `Plane`（subgroup broadcast/shuffle/sum/min/max）
- `CoopMma`（协同矩阵乘累加——Tensor Core 抽象）
- `Tma`（张量内存加速器——H100 特性）
- `Memory`（Load/Store/Index——支持全局/共享/局部地址空间）

CubeCL 的 IR 覆盖范围比 WGSL 更丰富（包含 tensor core 和 subgroup 原语），比 SPIR-V 更倾向 GPU 语义而非通用计算。

### 第三步：IR 优化 —— 多次 pass 循环收敛

编译目标代码之前，CubeCL 对 Scope 树运行优化 pass（`cubecl/crates/cubecl-core/src/post_processing/mod.rs:27`），在 `loop` 中反复执行直到无更多变化：

1. **`ConstOperandSimplify`**（`constant_prop.rs:24`）：半常量化简——`Add(0, x)` → `x`、`Mul(x, 1)` → `x`、`Mul(x, 0)` → `0`、`Div(x, 1)` → `x`，以及布尔短路（`true || x` → `true`）。在融合 kernel 中，`x * 1.0` 被直接移除。

2. **`ConstEval`**（`constant_prop.rs:131`）：两个操作数都是常量时，在编译器的 Rust 代码中用 `num_traits::Float` 求值。`Add(Constant(1.0), Constant(2.0))` → `Constant(3.0)`。支持三角函数、指数、对数——计算时不生成任何 GPU 指令。

3. **`InlineAssignments`**（`expression_merge.rs:13`）：建立替换表。当看到 `Copy(input)` 且输入/输出类型匹配时，记录 `{out → input}`。后续所有使用 `out` 处替换为 `input`。`x = y; z = x + 1` → `z = y + 1`。

4. **死代码消除**：前几步产生的不再被引用的变量被移除。

循环收敛是关键——常量折叠可能打开内联机会，内联又可能打开新的常量折叠。

WGSL 编译器在代码生成前还运行后端特定的 pass（`compiler/wgsl/compiler.rs:123`）：

- `CheckedIoVisitor`——为矢量化访问插入边界检查
- `DisaggregateVisitor`——将胖指针（Tensor 参数包含 data + shape + stride）拆分为基本分量
- `UnrollVisitor`——**向量拆解**（vector unrolling），将宽向量（如 vec16）分解为标量/窄向量操作。这是向量层面的拆分，不是循环展开

### 第四步：多平台代码生成

CubeCL 在 `WgpuServer` 初始化时根据 wgpu adapter 选择后端编译器（`cubecl/crates/cubecl-wgpu/src/compiler/base.rs:35`）：

```rust
pub enum AutoCompiler {
    Wgsl(WgslCompiler),
    #[cfg(feature = "spirv")] SpirV(SpirvCompiler),
    #[cfg(feature = "msl")]  Msl(MslCompiler),
}
```

#### WGSL 编译器

`WgslCompiler`（`compiler/wgsl/compiler.rs`）将 IR 操作一对一翻译为 WGSL 代码。WGSL 原生不支持的数学函数被注入为扩展函数（`extension.rs`），在 `ComputeShader` 格式化时追加在 `fn main` 之后。关键示例如 `powf` 的扩展（`extension.rs:241`）：

```wgsl
fn powf_primitive_f32(lhs: f32, rhs: f32) -> f32 {
    if rhs == 0.0 { return 1.0; }           // 指数 0
    let even = rhs % 2.0 == 0.0;
    if even { return pow(abs(lhs), rhs); }   // 偶指数：取绝对值
    return -pow(-lhs, rhs);                  // 奇指数：取负绝对值
}
```

`isNan` 和 `isInf` 通过 IEEE 754 位操作实现：`bitcast<u32>(x)` 取位，掩码取指数字段，分别比对全 1（NaN）或全 1 + 分数全 0（Inf）。WGSL 没有原生的 NaN/Inf 检测——这些扩展使 CubeCL 的 IR 语义完整映射到 WGSL。

#### SPIR-V 编译器

`cubecl-spirv` crate 实现完整的 `Compiler` trait 后端（`compiler.rs:144`），将 CubeCL IR 翻译为 SPIR-V 二进制。它运行与 WGSL 相同的优化 pass（`CheckedIoVisitor`、`DisaggregateVisitor`、`UnrollVisitor`），外加 SPIR-V 专用的变换（`ErfTransform`、`BitwiseTransform`、`HypotTransform`）。

SPIR-V 并非"绕过编译"——它是完整的编译后端。区别在于 wgpu 的提交方式：编译后的二进制通过 `create_shader_module_passthrough` 直接传给 Vulkan 驱动（`backend/base.rs:89`），跳过 wgpu 的内部 WGSL 编译和验证。编译后的二进制还缓存在磁盘上，key 为 `(properties_hash, kernel_id.stable_hash())`——驱动更新通过 `properties_hash` 自动使缓存失效。

MSL 路径（Metal）同理，`cubecl-cpp` crate 将 IR 编译为 Metal Shading Language 源代码，通过 passthrough 提交。

### 第五步：Pipeline 创建与 Dispatch

编译完成的最终阶段在 `WgpuServer::pipeline()`（`compute/server.rs:165`）：

1. **生成 `KernelId`** = (type_id, address_type, cube_dim, mode, info)，检查 `self.pipelines: HashMap<KernelId, (ComputePipeline, CompilerInfo)>`
2. 命中 → 跳过编译；未命中 → 编译 → 创建 `ShaderModule` → `ComputePipeline`
3. **GPU dispatch** 在 `WgpuStream::register_pipeline`（`compute/stream.rs:587`）：
   - 从 `WgpuResource` 构建 `BindGroupEntry` 列表
   - 打开 `ComputePass`，设置 `Pipeline` 和 `BindGroup`
   - `pass.dispatch_workgroups(x, y, z)` 或 `dispatch_workgroups_indirect()`

### 与 Triton JIT 对比

| 维度 | CubeCL | Triton |
|------|--------|--------|
| 编译时机 | Rust crate 编译期（proc macro） | Python 运行时（`@triton.jit`） |
| 特化机制 | `#[comptime]` 泛型参数 + Rust monomorphization | Python AST → Triton IR → 编译 |
| IR 结构 | 嵌套 Scope 树 | CFG + 基本块 |
| 优化 | 多次 pass 循环收敛 | Triton IR → Triton GPU IR passes |
| 后端 | WGSL/SPIR-V/MSL → wgpu | LLVM IR → PTX |
| 缓存 key | KernelId (type + dim + comptime hash) | (kernel_fn, input_signature) |
| 首次延迟 | 低（comptime 值不同的重编译） | 高（完整 JIT + autotune 秒级） |
| 循环展开 | 宏层面（代码生成阶段） | IR 优化 pass |
| IR 覆盖 | GPU 原语（tensor core, subgroup, TMA） | 通用 + Triton 方言 |

### 限制

- **IR 是树不是图**：无 phi 节点，复杂控制流优化不充分
- **SPIR-V/MSL 需额外 crate**：非 WGSL 路径依赖 `cubecl-spirv` / `cubecl-cpp`
- **编译缓存维度**：每种 comptime 组合产生独立缓存条目
- **无跨 kernel 优化**：per-kernel 编译，无法共享常量池或做跨 kernel 内联

---

## 第四部分：Autodiff —— 训练的引擎

### 架构：装饰器模式

Burn 的 autodiff 是编译期的后端装饰器（`burn/crates/burn-autodiff/src/backend.rs:22`）：

```rust
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}
```

`B` 是任意 `Backend`。`Autodiff<B>` 自身也实现 `Backend`——这意味着可以写出 `Autodiff<Fusion<CubeBackend<WgpuRuntime>>>` 这样的嵌套。与 PyTorch 的根本差异：**Burn 将"是否需要 autodiff"编译期为类型差异**。推理场景直接用 `CubeBackend`，autodiff 代码完全排除在二进制之外。PyTorch 将其作为运行时属性（`tensor.requires_grad_()`）。

### 图构建：Tape-based Walkthrough

`z.backward()` 调用前，前向操作必须已执行。前向执行时，每个 op 同时注册其反向步骤到 `AutodiffServer.steps: HashMap<NodeId, StepBoxed>`。整个流程是 tape-based 的，和 PyTorch 一致。

回到我们的示例。对 `z = (x * 2.0 + 1.0).tanh()` 进行 trace：

```
前向图构建:

x ──→ Mul(x, 2.0), order=1  ──→ temp
                                  ↓
s=1.0 ──→ Add(temp, 1.0), order=2 ──→ y
                                        ↓
         Tanh(y), order=3 ──→ z
```

每个 op 通过统一的模式注册反向步骤（以 `Mul` 为例，`ops/tensor.rs:319`）：

1. 定义单元结构体 `struct Mul;`
2. 实现 `Backward<B, 2>` trait（`N=2` 表示两个父输入）：
   ```rust
   impl<B: Backend> Backward<B, 2> for Mul {
       type State = (Shape, Shape);  // 前向保存的状态：两个输入的 shape

       fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _c) {
           let grad = grads.consume::<B>(&ops.node);      // 1. 消费输出梯度
           binary::<B, _, _>(ops.parents, ops.node, grads,
               |grad| broadcast_shape(grad, &shape_lhs),  // 2. 流向 lhs
               |grad| broadcast_shape(grad, &shape_rhs),  // 3. 流向 rhs
           );
       }
   }
   ```
3. 通过类型状态构建器 `OpsPrep` 注册到图中

**`OpsPrep` 类型状态构建器**（`ops/base.rs:25`）使用 Rust 的 PhantomData 模式在编译期强制正确的注册顺序：

```
Init → .memory_bound() → MemoryBound → .retro_forward() → MemoryBoundRetroForward
  → .parents() → ComputePropertyDone → .stateful() → Tracked/UnTracked → .finish()
```

每个状态只能调用该状态上定义的特定方法——在 `Init` 上调用 `.retro_forward()` 是编译错误。这个模式消除了运行时状态错误。

`binary()` 辅助函数（`ops/backward.rs:50`）处理标准流程：
1. `grads.consume::<B>(&node)`——对 `GradInBackward` 节点调用 `remove()`（消费后移除），对 `Grad` 节点调用 `get()`（clone 保留）
2. `duplicate()`——为每个需要梯度的父节点 clone 梯度；`Requirement::None` 的父节点产生 `None`，跳过
3. 应用 `func_lhs`/`func_rhs` 变换（broadcast、transpose、negate 等）
4. `grads.register::<B>(node_id, grad)`——如有已有梯度则 `float_add` 累加

### 检查点：ComputeBound vs MemoryBound

每个前向 op 被实现者分类为（`graph/node.rs:23`）：

```rust
pub enum ComputingProperty {
    ComputeBound,                                      // 保留前向输出
    MemoryBound { retro_forward: Arc<dyn RetroForward> }, // 丢弃，反向时重算
    Ambiguous,
}
```

**`ComputeBound`** op：`matmul`、`conv2d`、`embedding`、`gather`、`scatter`、`pooling`、`ctc_loss`——前向输出在反向传播时需要且计算成本高，保留在内存中。

**`MemoryBound`** op：`Add`、`Mul`、`Neg`、`Exp`、`Tanh`、`Sigmoid`、`sqrt`、`abs`、`reshape`、`select`、`slice`、`permute`——前向输出重算成本极低，通过 `RetroForward` 闭包重算：

```rust
// checkpoint/retro_forward.rs:17
pub trait RetroForward: Debug + Send + 'static {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId);
}

// 通过宏生成具体实现
retro_binary!(RetroAdd, |lhs, rhs| B::float_add(lhs, rhs));
retro_unary!(RetroNeg,  |input| B::float_neg(input));
```

在我们的示例中，`Mul` 和 `Add` 是 MemoryBound——`temp` 和 `y` 被丢弃。反向传播时：

```
TanhBackward 需要 y → checkpointer.retrieve_node_output(y)
  → y 是 MemoryBound → 重算 y = temp + 1.0
    → temp 也是 MemoryBound → 再重算 temp = x * 2.0
      → x 是 ComputeBound（叶子 tensor, require_grad）→ 直接读取
  → 重算得到 y → TanhBackward 计算 ∂z/∂y
```

### 检查点策略

两个预置策略（`checkpoint/strategy.rs`）：

- **`NoCheckpointing`**（默认）：`compute_property()` 永远返回 `ComputeBound`，忽略 op 的标记。适合 GPU 内存充足的场景。
- **`BalancedCheckpointing`**：`compute_property()` 将 `RetroForward` 包装进 `MemoryBound`，尊重 op 分类。用内存换时间。

与 PyTorch 的 `torch.utils.checkpoint.checkpoint()` 不同——Burn 的检查点是**op 级标记 + 策略选择**，不需要用户手动标记分段。代价是粒度固定在 op 级，无法做 segment 级或 layer 级的自定义检查点。

### 反向执行：BFS 逆序

`z.backward()` → `AutodiffServer::backward()`（`runtime/server.rs:62`）：

1. **构建 tape**：`BreadthFirstSearch::traverse()` 从 z 做 BFS，`build_tape()` 的闭包按 `step.depth()`（即 `order` 字段）将 step 分组
2. **逆序执行**：`tape.into_iter().rev()`——depth 最大（叶子节点）先执行
3. **梯度累积**：`Gradients::register()` 检测已有梯度 → `float_add` 累加
4. **图销毁**：每个 step 在 `step()` 调用中消费；`GraphMemoryManagement::free_unavailable_nodes()` 在 backward 完成后清理

完整 trace：

```
前向:
  x ──→ (*2.0, order=1) ──→ temp
                              temp ──→ (+1.0, order=2) ──→ y
                                                              y ──→ (tanh, order=3) ──→ z

反向 (BFS 从 z，按 order 逆序):
  order=3: TanhBackward  ← dz
            y ← RetroForward 重算: tanh(y) 的反向需要 y 来计算 1-tanh²(y)
            → grad[y] = dz × (1 - tanh²(y))

  order=2: AddBackward  ← grad[y]
            → grad[temp] = grad[y]  （加法反向：梯度直接透传）

  order=1: MulBackward  ← grad[temp]
            → grad[x] = grad[temp] × 2.0  注册到 Gradients
            → grad[scalar] = x × grad[temp]  （标量不需要梯度，Requirement::None → 跳过）
```

### 分布式梯度同步

分布式训练时，每个设备独立计算局部梯度。Burn 通过 `on_register` 钩子在梯度注册时内联触发 `all_reduce`（`grads.rs:142`）：

```rust
// burn-autodiff/src/distributed.rs:45
fn on_register(&mut self, id: &NodeId, container: &mut TensorContainer) {
    if let Some(params) = self.sharded_parameters_map.get(id) {
        *self.n_required_map.get_mut(id) -= 1;
        if *self.n_required == 0 {              // 所有路径梯度到位
            B::submit_gradient_sync(tensor, params);  // 触发 all_reduce
        }
    }
}
```

`n_required_map` 以引用计数跟踪每个参数还有多少路梯度待注册。计数归零时才提交同步——无需对整个梯度图做第二遍遍历。梯度同步不是作为独立操作插入图中，而是在 `Gradients` 容器上钩子化——在梯度注册完成后立即可用于同步。

### 内存管理：图生命周期

`GraphMemoryManagement`（`runtime/memory_management.rs`）通过 `Arc<Node>` 的强引用计数追踪节点存活状态。其三层算法：

1. `unavailable_propagation()`——标记父节点被消费/删除的节点为 Unavailable
2. `useful_propagation()`——标记 `strong_count > 1`（仍有 tensor 引用）的节点为 Useful
3. `identify_leaves_and_deletables()`——既不是 Useful 也不是 Unavailable 的节点可删除

如果用户持有任何图中 tensor 的引用（`strong_count > 1`），`maybe_useful()` 返回 true——整个子图保留。这不是 bug，是正确性要求。图总是在反向传播被消费后销毁——不支持高阶梯度的根本原因。

### 与 PyTorch Autograd 对比

| 维度 | Burn | PyTorch |
|------|------|---------|
| 架构 | 装饰器 `Autodiff<B, C>`，编译期参数化 | 内置于 tensor 类型，C++ 运行时 |
| 梯度跟踪 | 编译期类型区分 | 运行时 `requires_grad_()` |
| 图构建 | Tape-based，eager | Tape-based，eager |
| 图存储 | `HashMap<NodeId, StepBoxed>` 扁平 | `grad_fn` 指针链 DAG |
| 遍历 | BFS 分层 + 逆序 | 拓扑排序 |
| 高阶梯度 | 不支持 | `create_graph=True` |
| 检查点 | op 级标记 + 策略参数化 | 手动 `checkpoint()` context |
| 分布式 | `on_register` 钩子 + 引用计数 | `DistributedDataParallel` |
| 推理模式 | 编译期排除 autodiff crate | 运行时 `torch.no_grad()` |
| 图生命周期 | 反向传播中销毁 | `retain_graph=True/False` |
| 线程模型 | `GraphMutexClient`，per-device 锁 | GIL + C++ 线程锁 |
| 反向融合 | 不支持（绕开 fusion 层） | 不支持 |

### 限制

- **无高阶梯度**：图在反向传播中消费。需要 Hessian 的场景无法实现
- **无反向融合**：反向 element-wise 链独立执行，没有经过 fusion 优化
- **检查点粒度固定于 op 级**：无 segment/layer 级的手动设置
- **图扁平化**：`HashMap` 不保留层次结构，BFS 重建拓扑有开销
- **类型状态 builder 冗长**：每个 op 实现约需 50 行 boilerplate

---

## 全景回顾：一个 Training Step 的全链路

回到开篇的代码，汇总每一步在四个系统中的经历：

```rust
// ─── 前向：Fusion + Autodiff 记录 ───
let x = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
// → Tensor::from_data()
//   → BridgeTensor::float(Dispatch::float_from_data(data, device))
//   → Fusion::float_from_data()
//     1. B::float_from_data() → CubeCL 层分配 GPU 内存（Page/Slice）
//     2. client.register(OperationIr::Init(desc), NoOp) → 入队
// → autodiff: x 是叶子节点，require_grad，Requirement::Grad

let z = (x.clone() * 2.0 + 1.0).tanh();
// → 3 个 OperationIr 入队: MulScalar, AddScalar, Tanh
// → autodiff 图同步构建:
//   x ─MulBackward─ temp ─AddBackward─ y ─TanhBackward─ z
//   （MemoryBound: temp, y 通过 RetroForward 可重算）

// ─── backward 触发 drain → Fusion 执行 → 反向传播 ───
z.backward();
// 1. 前向必须已执行（反向需要中间结果）→ drain 触发
//    MultiStream::drain(ExecutionMode::Sync)
//
// 2. Processor::process → Policy.action → Explore
//    Explorer::explore → StreamOptimizer::optimize
//    → BlocksOptimizer::optimize → Block::optimize
//    → find_best_optimization_index → ElementWiseFuser 胜出
//    → FuseTrace { ops: [Assign, Mul, Add, Tanh] }
//    → ExecutionPlanStore 缓存方案
//
// 3. FuseTraceLauncher::launch
//    → InputPlanner  → buffer binding layout
//    → OutputPlanner → output allocation
//    → VectorizationPlanner → vec4 for f32
//    → LaunchPlanExecutor → Runner::run → wgpu dispatch
//
// 4. [可选] Autotune: 向量化宽度候选
//    候选数小（1-2个），warmup×3 + sample×10
//    → 缓存 key = (elemwise_fuse TypeId, [64,1,1], Unchecked, config_hash)
//
// 5. Pipeline 编译/缓存命中:
//    KernelId → HashMap 缓存 → 命中则跳过编译
//    否则: WGSL/SPIR-V/MSL 编译 → ShaderModule → ComputePipeline
//
// 6. GPU dispatch:
//    BindGroup → ComputePass → dispatch_workgroups
//    生成的 Metal 着色器:
//      metal::float2 * scalar → + scalar → safe_tanh
//
// 7. 反向传播开始:
//    BFS 从 z: order=3 TanhBackward → order=2 AddBackward → order=1 MulBackward
//    每步: grads.consume → 计算 → grads.register
//    中间结果通过 RetroForward 重算
//
// 8. 图清理:
//    GraphMemoryManagement::free_unavailable_nodes

let grad_x = x.grad();
// → Gradients 容器中查询 x.node.id
// → ∂z/∂x = (1 - tanh²(2x+1)) × 2
```

### 四系统交互图

```
                     z.backward()
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Autodiff │    │  Fusion  │    │ Autotune │
  │ 图构建   │    │ 操作排队 │    │ 参数选择 │
  │ BFS 逆序 │    │ Block 优化│    │ 优先级剪枝│
  │ 梯度累积 │    │ 方案缓存 │    │ anchor 量化│
  │ 检查点重算│    │ Stream隔离│    │ 评分+缓存 │
  └──────────┘    └──────────┘    └──────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                ┌──────────────────────┐
                │    JIT 编译管线       │
                │ #[cube] → IR → 优化   │
                │ → WGSL/SPIR-V/MSL    │
                │ → Pipeline → Dispatch │
                └──────────────────────┘
                         │
                         ▼
                    ┌──────────┐
                    │   GPU    │
                    └──────────┘
```

### 各系统核心设计选择总结

| 系统 | 核心选择 | 替代方案 | 取舍 |
|------|---------|---------|------|
| Fusion | 惰性队列 + 同步点触发 | XLA 静态编译 / PyTorch eager | 探索开销换无静态图需求 |
| Autotune | 策略枚举 | Triton 参数网格 | 覆盖范围换候选最少化 + 无效组合为零 |
| JIT | comptime 泛型模板 | Triton 运行时 JIT / Candle AOT | 编译期开销换运行时一致性 + 融合特化 |
| Autodiff | 装饰器模式 + 类型状态构建 | PyTorch 内置 autograd | 推理编译期排除 crate + 图销毁保证正确性 |
| Memory | Page/Slice 三层分配 | 每次创建 wgpu buffer | 少量空闲内存换大幅减少 GPU 分配调用 |

### 关键源码入口总览

| 系统 | 入口文件 |
|------|---------|
| Fusion 引擎 | `burn/crates/burn-fusion/src/stream/multi.rs` |
| Fusion 执行 | `burn/crates/burn-cubecl-fusion/src/engine/launch/base.rs` |
| Autotune 框架 | `cubecl/crates/cubecl-runtime/src/tune/` |
| Matmul 候选注册 | `burn/crates/burn-cubecl/src/kernel/matmul/tune/base.rs` |
| Autotune 评分 | `cubecl/crates/cubecl-common/src/benchmark.rs` |
| `#[cube]` 宏 | `cubecl/crates/cubecl-macros/src/lib.rs` |
| IR 定义 | `cubecl/crates/cubecl-ir/src/scope.rs`、`operation.rs` |
| IR 优化 | `cubecl/crates/cubecl-core/src/post_processing/mod.rs` |
| WGSL 编译器 | `cubecl/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs` |
| WGSL 扩展 | `cubecl/crates/cubecl-wgpu/src/compiler/wgsl/extension.rs` |
| SPIR-V 编译器 | `cubecl/crates/cubecl-spirv/src/compiler.rs` |
| Pipeline 缓存 | `cubecl/crates/cubecl-wgpu/src/compute/server.rs` |
| GPU 内存管理 | `cubecl/crates/cubecl-runtime/src/memory_management/memory_manage.rs` |
| Autodiff 后端 | `burn/crates/burn-autodiff/src/backend.rs` |
| 图构建与执行 | `burn/crates/burn-autodiff/src/runtime/server.rs` |
| 检查点策略 | `burn/crates/burn-autodiff/src/checkpoint/strategy.rs` |
| 类型状态 Builder | `burn/crates/burn-autodiff/src/ops/base.rs` |
| 分布式同步 | `burn/crates/burn-autodiff/src/distributed.rs` |

---

## 继续阅读

| ← 上一篇 | 下一篇 → |
|-----------|----------|
| [architecture.md](../architecture.md) — 跨项目设计哲学 | 任选一篇深入：[Fusion](kernel-fusion-system-design.md) · [Autotune](../cubecl/autotune-system-design.md) · [JIT](../cubecl/jit-compilation-pipeline.md) · [Autodiff](autodiff-system-design.md) |

动手：[src/burn-test/](../../src/burn-test/) — `RUST_LOG=burn_fusion=trace` 观察融合日志
