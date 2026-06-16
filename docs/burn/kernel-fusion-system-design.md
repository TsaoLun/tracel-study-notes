# Burn Kernel Fusion 的系统设计

> Burn 不立即执行操作——而是排队、在同步点触发融合优化、缓存方案。与 XLA 的静态融合和 Triton 的 JIT 融合不同，它选择了惰性队列 + 探索缓存的中间路线。本文展开这个选择的设计权衡。

## 为什么需要 Kernel Fusion

一个 ML 框架的推理/训练是一系列 GPU kernel 的执行。每个 kernel 完成一小段计算：矩阵乘法、逐元素加法、激活函数等。**kernel launch 有可测量的时间开销**。

### 一组估算数字

在 NVIDIA A100 上，一次 CUDA kernel launch 的软件开销约 5-10 μs（隐式同步、grid/block 参数计算、驱动调度）<sup>*</sup>。一个典型的 memory-bound element-wise op（比如 `tensor_a + tensor_b`）在 1024×1024 f32（约 4MB）数据上执行时间约 3-5 μs。这意味着：

> <sup>*</sup> 该数字来自 CUDA 社区常见估算，实际开销取决于驱动版本、CUDA 版本和系统负载。复验：`nvprof --print-gpu-trace` 或 CUPTI API。

```
launch overhead ≈ 2× compute time
```

三个连续的 element-wise op（`* 2.0`、`+ 1.0`、`tanh`）若分别执行：

```
总耗时 ≈ 3 × (kernel launch) + 3 × (compute)
       ≈ 3 × 7μs + 3 × 4μs
       ≈ 33μs
```

而如果把三个操作融合成一个 kernel：

```
总耗时 ≈ 1 × (kernel launch) + 1 × (12μs compute)
       ≈ 19μs
```

**在 element-wise 密集的计算图上，消除 launch 开销可以带来 30-50% 的端到端延迟改善。** 更重要的是，融合同样减少了全局内存的读写次数——中间结果不再需要写回 GPU DRAM 再读出来，而是停留在寄存器或共享内存中。

### 融合的代价

融合也有代价。把十个 op 融合成一个 kernel 意味着：
- 寄存器压力增大——可能降低 occupancy（同时活跃的 warp 数）
- 某些 op 需要不同的 tile 大小才高效，融合后被迫折中
- 编译/探索融合方案本身需要时间

融合是否有净收益取决于**具体的硬件、数据规模和操作组合**。一个融合方案在 A100 上可能是净收益，在 RTX 3060 上可能因为寄存器预算不同而劣于分别执行。

举个具体例子。NVIDIA A100 每个 SM 有 65536 个 32-bit 寄存器，每个 thread 最多用 255 个。假设一个 matmul kernel 已经用了 160 个寄存器/线程（tensor core 需要大量寄存器做双缓冲），occupancy 刚好维持在 50%（每个 SM 可同时跑 1024 个线程）。如果再融合一个 GELU（~20 个额外寄存器做指数计算），寄存器压力推到 180/线程，occupancy 掉到 37.5%——可用 warp 数从 32 掉到 24。matmul 本身是 compute-bound，受 occupancy 影响大；多出的 25% 寄存器压力可能让有效吞吐下降 15-20%，而融合省掉的一次 launch（~5μs）在 compute-bound 的毫秒级 kernel 上几乎看不出来。**这个融合是净负收益。**

反过来，对于 memory-bound 的 element-wise 操作链（`*2.0 + 1.0 + tanh`），寄存器压力本身很低（~30 个/线程），融合后也不到 64，occupancy 不变。但省掉的两次 launch（~10μs）在总延迟（~33μs）中是显著比例。**净收益明确。**

融合引擎的挑战在于：它需要在编译期（没有实际跑 benchmark 之前）判断一个融合方案属于前一种还是后一种情况。

---

## 融合的三种范式

### 静态融合（XLA）

XLA 在编译期基于计算图做融合。HLO 图上的 op 按规则合并——比如 `dot + bias_add + relu` 可以融合成 `cublasLt` 的一个 fused epilogue。优点是决策成本为零（规则固化），缺点是需要静态图，且规则覆盖范围有限。

### JIT 编译融合（Triton / TVM）

Triton 在运行时为 Python 函数生成优化的 GPU kernel。用户用 Python 描述计算，编译器做 tiling、向量化、autotuning。优点是灵活且优化空间大，缺点是编译开销高（首次运行时 autotune 可能耗时秒级），且需要用户显式标注可融合的计算。（详见 [CubeCL JIT 编译管线](../cubecl/jit-compilation-pipeline.md) 和 [Autotune 系统设计](../cubecl/autotune-system-design.md) 与 Triton 的对比。）

### 惰性队列融合（Burn）

Burn 选择了一条中间路线：**操作不立即执行，而是排队。在需要结果时（同步点），对队列中的操作序列做融合优化，生成并缓存优化方案。** 这和 PyTorch 的 eager 模式不同（eager 模式下每个 op 立即 launch kernel），也不同于完全的静态编译。

融合方案有缓存——相同的操作序列第二次遇到时直接命中缓存，不重新探索。探索开销只在首次出现时发生。

三种范式的核心差异在于 **overhead 发生时机**：静态融合在编译期、JIT 在运行期第一次执行、Burn 在同步点时探索但后续命中缓存。

---

## Burn 的融合引擎：核心设计决策

### 决策 1：Stream-local 优化，而非全局图编译

Burn 没有全局计算图的概念。每个 tensor 操作被记录为 `OperationIr`（一种 IR，包含操作类型、输入/输出 tensor id 和形状），推入当前 stream 的 `OperationQueue`。优化范围限定在单条 stream 的连续操作序列内。

这个设计的核心好处是**不需要静态图**。PyTorch 式的动态控制流（if/for 中动态创建 tensor）可以正常工作——每个分支自然形成自己的操作序列。代价是优化范围受限于单条 stream 内的连续操作，跨分支的融合不会发生。

### 决策 2：惰性执行 + 同步点触发

操作入队后不执行。以下操作会触发同步（drain）：

- **读取 tensor 数据**：`println!("{}", tensor)` → `into_data_async()` → `drain_stream()`
- **跨 stream 共享**：tensor 被 clone 到另一个线程，源 stream 在共享前 drain
- **显式 sync**：`client.sync(|| ...)`

关键点：**显示 tensor 值才触发执行，构建计算图不触发执行**。这和使用 tracing（`torch.jit.trace`）或 JIT 编译（`torch.compile`）有本质不同——不需要提前声明"这一段需要优化"，也不需要 warmup 迭代。

> ▶ **动手**：`cd src/burn-test && RUST_LOG=burn_fusion=trace cargo run --release`
> 设置 `RUST_LOG=burn_fusion=trace` 后运行，观察 `[stream]`（操作入队）、`[plan]`（Policy 决策）、`[explorer]`（探索融合机会）日志行。对照本节描述的 drain → process → explore 流程。

### 决策 3：探索 + 缓存的二级命中模型

当 stream 被 drain 时，`Processor` 对队列中的操作序列做两件事：

1. **查询缓存**（`Policy`）：当前操作序列是否已经有已知的优化方案？
2. **探索新方案**（`Explorer`）：如果没有，尝试将操作分组为 Block，对每个 Block 找最佳融合策略

探索的结果（`ExecutionPlan`）被存入 `ExecutionPlanStore`——一个跨 stream 共享的缓存。下次相同或相似的操作序列被 drain 时，Policy 直接命中缓存，探索开销为 0。

缓存匹配使用 **trigger 机制**：一个 plan 在"前 N 个 op 匹配 + 后续 op 出现在 trigger 列表中"时被激活。与精确序列比对不同，trigger 机制允许同一个融合方案应用于不同的后续上下文——比如 `*2.0 + 1.0 + tanh` 的融合方案在 `*2.0 + 1.0` 匹配后，无论后面是否跟着 `tanh` 还是其他已注册的 trigger op，都可以执行。

### 决策 4：OperationFuser 的竞争制

一个 `Block` 包含多个 `OperationFuser` 实现——`ElementWiseFuser`（逐元素融合）、`MatmulFuser`（矩阵乘法融合）、`ReduceFuser`（规约融合）等。

每个 fuser 在收到操作时：
- **accept**：操作可以被融合 → `num_ops++`（分数增加）
- **reject**：操作无法被融合 → `status = Closed`（此 fuser 关闭，不再接受后续操作）

当 Block 被优化时，选择**分数最高的 fuser** 的方案执行。

这个竞争机制优雅地解决了"没有全局计算图"的问题——不需要提前决定"这个序列应该用哪种优化器"。所有优化器同时尝试消化操作序列，最强的那个胜出。专用的优化器（如 MatmulFuser）通过加分策略获得优先级。

**但这引入了一个微妙的问题**：如果操作序列的前半段是 element-wise（可以被 ElementWiseFuser 消化），后半段是 matmul（可以被 MatmulFuser 消化），Block 的 merging pass 可能无法正确处理这种边界。这是一个实实在在的限制。

> ▶ **跟练**：[fusion/1-client-server.md](fusion/1-client-server.md) — 理解 `Tensor::from_data` 如何穿过 Fusion client-server 到达 GPU buffer 的完整链路。与本文描述的 fusion 排队和执行互补。

---

## 内存管理的系统设计：Page / Slice 模型

Burn/CubeCL 的 GPU 内存管理是这篇文章的第二个重点。它采用了类似操作系统虚拟内存的 **Page / Slice** 模型。

### 为什么需要自定义分配器

`wgpu::Device::create_buffer()` 是一个昂贵的调用——需要内核交互和驱动注册。如果每个 tensor 分配都创建一个新的 wgpu buffer，不仅慢，还会产生大量碎片化的 GPU 内存。

### 三层分配策略

**层 1：PersistentPool（永久池）**

按精确大小通过 HashMap 索引的预分配 slice。用于永不释放的分配（如模型权重）。HashMap 查找是 O(1)，同一大小桶内如有多个空闲 slice 则线性扫描；在典型场景（每种大小的权重参数只分配一次）中桶内通常只有一个 slice。这是最快的路径。

**层 2：DynamicPool 复用（SlicedPool）**

维护多个"页"（Page，一块固定大小的 wgpu buffer，如 32MB/128MB）。每个页内部有多个可变大小的"切片"（Slice）。调用 `try_reserve(size)` 时：
1. 遍历已有页，调用 `MemoryPage::coalesce()` 合并相邻空闲切片
2. 找到一个足够大的空闲区域 → 切出所需大小 → 返回

此路径**不创建新的 wgpu buffer**，只是 offset 运算，开销接近零。

**层 3：创建新页**

所有现有页都无法满足分配时，调用 `WgpuStorage::alloc()` → `wgpu::Device::create_buffer()` 创建新页。这是最昂贵的路径，但出现频率被前两层大幅降低。

### 分离的内存域

CubeCL 维护三个独立的内存池（`WgpuMemManager`）：

| 池 | 用途 | Buffer Usage |
|----|------|-------------|
| `memory_pool` | 主 GPU 内存 | STORAGE \| COPY |
| `memory_pool_staging` | CPU 可读暂存 | MAP_READ \| COPY_DST |
| `memory_uniforms` | Uniform Buffer | UNIFORM \| STORAGE \| COPY_DST |

原因很简单：wgpu buffer 的 usage 标志在创建时确定且不可变。一个 `STORAGE` buffer 不能用作 uniform buffer。三种 buffer 必须分别管理，不能混合分配。

### 环形复用（ExclusiveMemoryPool）

Staging pool 和 uniform pool 使用 `ExclusiveMemoryPool`——每个页只有一个分配（不切片）。已释放的页不会立即销毁，而是留在页表中。`cleanup()` 方法每次遍历所有页：连续被标记 free 达到 5 次（`ALLOC_AFTER_FREE`）的页才调用 `dealloc()`。这个双向量轮转（two-vector swap）的滞留池模式使频繁复用的页保留在池中，偶尔使用的页被淘汰。（`cubecl/crates/cubecl-runtime/src/memory_management/memory_pool/exclusive_pool.rs:50-227`）

这个设计和 Linux kernel 的 slab allocator 思路一致：**在分配频率和内存占用的权衡中，用少量空闲内存换取大幅减少的 GPU 分配调用**。

---

## Stream 与 MultiStream：并发安全的操作隔离

### 问题

`FusionTensor` 是 `Send + Clone` 的。用户可以：
```rust
let t1 = Tensor::<2>::from_data(data, &device);  // stream A
let t2 = t1.clone();  // clone 到 stream B（另一个线程）
let t3 = t2 * 2.0;    // 在 stream B 上操作
```

`t2 * 2.0` 的操作会被推入 stream B 的队列。但 `t2` 的数据在 stream A 上——如果 stream A 的 pending 操作还没执行，`t2` 实际上还不存在于 GPU 内存中。

### 解决方案：alias 而非 merge

`MultiStream` 的策略是**不允许跨 stream 的 tensor id 出现在其他 stream 的队列中**。当检测到 `self.stream != StreamId::current()` 时：

1. **物化源 tensor**：如果源 tensor 还没被注册（产出它的 op 仍 pending），先同步 drain 源 stream
2. **创建 alias**：在目标 stream 分配新的本地 id，将源 tensor 的 backend handle clone 过去

clone backend handle 是 `Arc` 语义——两个 id 指向同一块 GPU 内存，refcount 共享。释放时各自 drop，最后一方释放底层 buffer。

### 为什么这个设计重要

它让**每个 stream 的操作模式保持简单**：单 stream 内的操作只需关心本地 tensor id，不需要跨流协调。融合也只在单 stream 内进行——跨流融合没有意义因为操作依赖不同的执行上下文。

对比 PyTorch：`tensor.to(device="cuda:1")` 触发的是**实际数据复制**，因为不同 GPU 有不同的内存空间。Burn 的跨 stream 共享是**同设备内的 alias**，零拷贝——这是因为它所有的 stream 运行在同一个 GPU 上，只是逻辑上隔离。

---

## 代码生成：Compile-time Specialization

### 融合不是运行时生成代码

Burn/CubeCL 的融合 kernel 是**预编译的模板**。核心是 `#[cube]` 宏和 `#[comptime]` 系统：

```rust
#[cube(launch_unchecked, address_type = "dynamic")]
fn elemwise_fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: &FuseBlockConfig,
) {
    // ...
    if pos < length {
        fuse_on_write::<f32>(inputs, outputs, &mut locals, pos, values, args, config)
    }
}
```

`config: &FuseBlockConfig` 携带了操作序列：`[Assign, Mul(scalar), Add(scalar), Tanh]`。kernel 内部通过 `#[unroll]` 在编译期展开这段序列——对于不同的操作列表，编译器生成不同的 specialize 实例。

### 这和 Triton 的区别

Triton 在 Python 层面描述计算，`@triton.jit` 装饰器触发 JIT 编译（Python → Triton IR → Triton GPU IR → LLVM IR → PTX）。每次修改 kernel 逻辑都需要重新编译。Burn 的方案是 **comptime 泛型**——用 Rust 的编译期能力将操作序列编码为泛型参数，这样不同的操作序列对应不同的 monomorphized 函数实例。GPU 代码生成的细节见 [CubeCL JIT 编译管线](../cubecl/jit-compilation-pipeline.md)。

代价是生成的 GPU 二进制数量随融合组合爆炸，但 CubeCL 的缓存机制缓解了这一问题。

---

## 限制与 trade-off 总结

1. **单流融合**：不跨 stream 融合。如果计算横跨多个线程，各自独立优化。
2. **单 producer 限制**：融合要求操作的输入 tensor 在同一 stream 上。跨 stream 的读操作会先 drain，破坏融合机会。
3. **首次探索开销**：新的操作序列首次执行时需要探索，延迟较高。适合重复执行相同模式的场景（训练循环的每个 iteration 执行相同的 op 序列）。
4. **缓存依赖操作序列的精确匹配**：`OperationIr` 的 relative 表示有助于模式匹配，但 tensor shape 变化会导致缓存 miss。动态 shape 场景下命中率下降。
5. **没有跨 op 的自动 tiling**：每个融合 kernel 的向量化策略由 `VectorizationPlanner` 决定，但不像 Triton 那样做全局的 tile size autotuning。

---

## 与其他系统对比

| 维度 | Burn | PyTorch Inductor | XLA | Triton |
|------|------|-----------------|-----|--------|
| 触发方式 | 惰性排队 + 同步点 | `torch.compile` 调用 | 静态图编译 | `@triton.jit` 装饰 |
| 图需求 | 不需要 | 需要 tracing | 需要静态图 | 需要显式标注 |
| 优化成本 | 首次探索 (~ms) | 编译 (~s) | 编译 (~s) | JIT+autotune (~s) |
| 缓存 | 操作序列缓存 | FX graph cache | 无 | 编译缓存 |
| 动态 shape | 天然支持 | 通过 guard 和重编译 | 有限 | 支持 |
| kernel 生成 | comptime 模板 | Triton IR → PTX | HLO → GPU IR | Python → PTX |

Burn 的路线适合**动态计算图 + 高重复性的计算模式**（比如 LLM 推理的 decode 阶段，每次都执行相同的逐 token 计算）。不适合**计算图频繁变化**的场景（首次探索开销重复发生）或**跨操作的大粒度优化**（单 stream 的连续 op 范围有限）。

---

## 关键源码入口

- 融合引擎：`burn/crates/burn-fusion/src/`
- CubeCL 融合实现：`burn/crates/burn-cubecl-fusion/src/`
- 内存管理：`cubecl/crates/cubecl-runtime/src/memory_management/`
- CubeCL 运行时：`cubecl/crates/cubecl-wgpu/src/compute/`

---

← [全景篇](burn-systems-architecture.md) | → 下一篇：[JIT 编译管线](../cubecl/jit-compilation-pipeline.md)
