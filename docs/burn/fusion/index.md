> **归档**：旧架构的章节计划。当前阅读路径见 [README](../../../README.md)，系统设计分析见 [Fusion 系统设计](../kernel-fusion-system-design.md)。已完成的第一章保留为可选延伸阅读。

# Burn Fusion 专题写作计划（已归档）
> **读计划前**：若你尚未读过 Burn 的类型栈概览，可先扫一眼 [架构分析](../../architecture.md)（5 分钟），再回到这里。

---

## 入门引导（Burn Fusion 机制新人必读）

### 你不需要先读过 CubeCL 专题

本专题假设你会 Rust，并了解 Burn 的最基本使用（`Tensor::from_data`、`matmul`）。不要求读过 CubeCL 编译器专题（`docs/cubecl/`）。遇到 CubeCL 概念时只点到其与 Fusion 的边界，不展开其编译器内部。

### 本专题的「主示例」是什么？

本专题用一个可在 Burn 仓库跑通的融合示例（独立可运行版本见 `src/burn-test/`；机制草稿参考 [nihalpasham 的 Gist](https://gist.github.com/nihalpasham/fc128f074e20d880bfd97198c2ac784b)）：

```rust
let tensor_1 = Tensor::<2>::from_data(
    [[2., 3.], [4., 5.]], &device
);
let y = tensor_1.clone() * 2.0 + 1.0;  // ScalarMul + ScalarAdd
let z = y.tanh();                       // Tanh
println!("{}", z);                       // ← 触发 drain + 融合 + 执行
```

三行操作生成 **一个** fused kernel（含 `*2.0 + 1.0 + tanh`），通过 `RUST_LOG=burn_fusion=trace` 可见完整融合日志。

**跟跑方式**：在本仓库 `src/burn-test/` 目录（需先在项目根 clone burn 仓库）：

```bash
cd src/burn-test
RUST_LOG=burn_fusion=trace cargo run --release
```

如需同时看 CubeCL 层的日志：

```bash
RUST_LOG=burn_fusion=trace,cubecl_wgpu::runtime=trace cargo run --release
```

| 名字 | 是什么 | 在哪里 |
|------|--------|--------|
| **`Wgpu`** | 内部类型栈：`Fusion<CubeBackend<WgpuRuntime>>`（用户通过 `Device::wgpu(..)` 获取，默认启用 fusion） | `burn/` · `crates/burn-wgpu/src/lib.rs` |
| **`GlobalFusionClient`** | Fusion 层的客户端：`DeviceHandle<FusionServer>` + `FusionDevice`，通过 `submit`/`submit_blocking` 与 server 通信 | `burn/` · `crates/burn-fusion/src/client.rs` |
| **`MultiStream`** | 管理多个操作流：`HashMap<StreamId, Stream>` | `burn/` · `crates/burn-fusion/src/stream/multi.rs` |
| **`OperationQueue`** | 单流的操作队列：`global`/`relative`/`operations`/`variables` | `burn/` · `crates/burn-fusion/src/stream/queue/base.rs` |
| **`Processor`** | 执行引擎：Policy（Explore/Defer/Execute）+ Explorer | `burn/` · `crates/burn-fusion/src/stream/execution/processor.rs` |
| **`Block<O>`** | `StreamOptimizer` 中一组可融合操作的抽象 | `burn/` · `crates/burn-fusion/src/search/block.rs` |
| **`FuseBlockBuilder`** | 构建融合块的 builder：`ops`/`reads`/`writes`/`tensor_writes` | `burn/` · `crates/burn-cubecl-fusion/src/engine/trace/block.rs` |
| **`elemwise_fuse`** | CubeCL kernel：用 `#[comptime]` + `#[unroll]` 展开任意 op 序列 | `burn/` · `crates/burn-cubecl-fusion/src/optim/elemwise/` |

> **路径约定**：下文 `crates/…` 路径均相对各仓库根（Burn 机制在 `burn/`，CubeCL runtime 在 `cubecl/`）。机制基准版本见 [README · 源码版本](../../../README.md#源码版本)（Burn **v0.21.0**）。

### 建议阅读顺序

1. 先读 [Fusion 系统设计](../kernel-fusion-system-design.md)——建立 MultiStream / drain / channel 的心智模型。
2. **第一章**：跟跑融合示例，确认 `RUST_LOG=burn_fusion=trace` 能看到融合日志。
3. 后续章按顺序读——每章依赖前一章的机制理解。

### 三份材料如何分工

| 材料 | 角色 |
|------|------|
| [../summary.md](../summary.md) | **地图**：类型栈 + 融合流全景 + Autodiff + 框架开销 |
| [CubeCL 专题](../../cubecl/index.md)（plan + [1](../../cubecl/1-gelu-launch.md)–[8] 章） | CubeCL 编译器：`#[cube]` → expand → SSA → JIT → autotune |
| **本专题** | **对照源码走 Fusion 运行时路径**（double client-server → queue → block → FuseTrace → kernel launch） |
| [Gist 草稿](https://gist.github.com/nihalpasham/fc128f074e20d880bfd97198c2ac784b) | **外部参考**：英文长文 + 调用链笔记；部分路径已随 v0.21.0 迁移（如 `shared/trace/` → `engine/trace/`），以本计划锚点为准 |

---

## 定位与读者

**目标读者**：读过 [Burn 地图](../summary.md) 或了解 Burn 基本使用；想知道"操作入队后到底发生了什么"的 Rust 开发者。
**不覆盖**：Autodiff 梯度图细节、Backend trait 实现（`burn-cubecl`）、CubeCL JIT 编译器内部（见 [CubeCL 专题](../../cubecl/index.md)）。

---

## 写作约定

1. **每章开头**：用 2–3 句话说明「本章锚点是什么、读完能解释什么」，不假设读者已懂源码路径。
2. **每章一个主示例**：始终用 `tensor.clone() * 2.0 + 1.0; tanh()` 三操作融合示例贯穿。
3. **源码路径写全**：`burn/` · `crates/burn-fusion/src/stream/multi.rs`（符号 `MultiStream::drain`），行号作近似参考。
4. **正文先跟练、再钉源码**；从具体调用链进入，再展开机制。
5. **章末**：小结 + 作业 + 下章预告。
6. **术语**以 [Fusion 系统设计](../kernel-fusion-system-design.md) 为准；各章只引入本章最少新词。

---

## 章节目录

| 章 | 文件 | 标题 | 读完能解释 | 核心源码锚点 |
|:---:|------|------|------------|--------------|
| 1 | [1-client-server.md](1-client-server.md) | 双客户端-服务器：从 `from_data` 到 GPU buffer | Fusion 和 CubeCL 各自有独立的 client-server；一次 tensor 分配穿过两条链路 | `burn/` · `crates/burn-fusion/src/client.rs`（`GlobalFusionClient`）、`crates/burn-fusion/src/server.rs`（`FusionServer`）；`cubecl/` · `crates/cubecl-wgpu/src/compute/server.rs`（`WgpuServer`）、`crates/cubecl-runtime/src/memory_management/memory_pool/`（`MemoryManager`、`SlicedPool`） |
| 2 | 2-operation-queue.md | OperationQueue：惰性执行与"推迟了什么" | `OperationQueue` 的五个字段（`global`/`relative`/`converter`/`operations`/`variables`）；操作入队但不执行意味着什么；什么触发 drain | `burn/` · `crates/burn-fusion/src/stream/queue/base.rs`（`OperationQueue`）、`crates/burn-fusion/src/op.rs`（`OperationIr`）、`crates/burn-fusion/src/stream/context.rs`（`StreamId`） |
| 3 | 3-drain-processor.md | Drain 与 Processor：Policy 状态机 | `MultiStream::drain` 全场；Policy 的 Explore/Defer/Execute 三态决策；`ExecutionMode::Sync` vs `Lazy` 的区别 | `burn/` · `crates/burn-fusion/src/stream/multi.rs`（`drain`）、`crates/burn-fusion/src/stream/execution/processor.rs`（`Processor::process`）、`crates/burn-fusion/src/stream/execution/policy.rs`（`Policy::action`） |
| 4 | 4-block-scoring.md | 增量融合：Block 注册与 Builder 评分 | `StreamOptimizer` 如何把 op 注册到 Block；Block 的 accept/reject 规则（tensor ID 交集）；Builder 评分如何选出最优 fuser | `burn/` · `crates/burn-fusion/src/stream/execution/explorer.rs`（`Explorer::explore`）、`crates/burn-fusion/src/search/optimization/stream.rs`（`StreamOptimizer`）、`crates/burn-fusion/src/search/block.rs`（`Block::register`、`Block::optimize`）、`crates/burn-fusion/src/search/optimization/blocks.rs`（`BlocksOptimizer`） |
| 5 | 5-fuse-block-builder.md | FuseBlockBuilder：reads、writes 与数据流分析 | `reads`（所有被读 tensor，含中间结果）vs `tensor_writes`（只有最终输出写全局内存）；`tensor_writes()` 的数据流分析决定哪些中间结果不需要写回 | `burn/` · `crates/burn-cubecl-fusion/src/engine/trace/block.rs`（`FuseBlockBuilder`、`tensor_writes`、`build`）、`crates/burn-cubecl-fusion/src/engine/trace/fuser.rs`（`TraceFuser`） |
| 6 | 6-fuse-trace-launch.md | 从 FuseTrace 到 kernel launch：input/output/vectorization 规划 | `InputPlanner`/`OutputPlanner`/`VectorizationPlanner` 如何把 FuseTrace 翻译为 launch plan；`LaunchPlanExecutor` 如何处理实际 kernel 启动 | `burn/` · `crates/burn-cubecl-fusion/src/engine/launch/plan.rs`、`engine/launch/input.rs`、`engine/launch/output.rs`、`engine/launch/executor.rs` |
| 7 | 7-elemwise-fuse.md | `elemwise_fuse` kernel：`#[comptime]` 展开任意 op 序列 | `elemwise_fuse` 通过 `#[comptime]` config + `#[unroll]` 在 JIT 编译期特化 op 序列，每种组合对应不同 JIT 产物 | `burn/` · `crates/burn-cubecl-fusion/src/optim/elemwise/optimization.rs`、`crates/burn-cubecl-fusion/src/engine/codegen/kernel.rs`（`elemwise_fuse` 的 `#[cube]` 定义） |
| 8 | 8-cross-stream-channel.md | 跨流共享、不可融合处理与 v0.21.0 channel 重构 | 跨流 tensor 共享（`tag_shared_view`、`resolve_streams`）；不可融合时退化为 `ExecutionStrategy::Operations`；v0.21.0 用 worker channel 替代递归锁（`submit`/`submit_blocking`、`DeviceServiceStage::Upstream`） | `burn/` · `crates/burn-fusion/src/client.rs`（`submit`、`submit_blocking`）、`crates/burn-fusion/src/stream/multi.rs`（`resolve_streams`）、`crates/burn-fusion/src/stream/memory_checks.rs`、`crates/burn-fusion/src/stream/store/base.rs`（`ExecutionStrategy`） |

> **与 [Fusion 系统设计](../kernel-fusion-system-design.md) 的关系**：系统设计文给出融合流的宏观全貌与设计权衡；本专题的 8 章逐机制追踪源码。读完地图知道"有 Policy/Explorer 这两个东西"；读完本专题能解释"Policy 为什么有三种 Action、Explorer 的 register_inner 为什么检查 tensor ID 交集"。

---

## 各章要点（写作 checklist）

### 第一章（已写）：双客户端-服务器

- [x] 章首说明"一次 `Tensor::from_data()` 穿过两条 client-server 链路"
- [x] Fusion 层 client-server：`GlobalFusionClient`（`server: DeviceHandle<FusionServer<R>>` + `device: FusionDevice<R>`）← 通过 worker channel 通信（`submit`/`submit_blocking`）
- [x] `FusionServer` 内部：`MultiStream` + `HandleContainer` + `Arc<FusionUtilities>`
- [x] CubeCL 层 client-server：`ComputeClient` / `WgpuServer`
- [x] 内存分配链：`MemoryManager::reserve()` → `SlicedPool::alloc()` → `ExclusiveMemoryPool::alloc_page()` → `WgpuStorage::alloc()` → `wgpu::Device::create_buffer()`
- [x] 对比两层：Fusion 层推迟操作、CubeCL 层立即分配
- [x] tensor 创建在 Fusion 队列中登记为 `NoOp`；实际 buffer 分配走 CubeCL 链
- [x] 作业：修改示例，用 `RUST_LOG` 观察不同后端（wgpu/cpu）的分配日志

### 第二章（待写）：OperationQueue 与惰性执行

- [ ] `OperationQueue` 五个字段的作用与关系：
  - `global: Vec<OperationIr>` — 精确 tensor ID 和 shape
  - `relative: Vec<OperationIr>` — 相对表示，帮助识别融合机会
  - `converter: OperationConverter` — 全局→相对表示的转换器
  - `operations: Vec<UnfusedOp<R>>` — 实际可执行操作（`UnfusedOp` 是具体包装，非 trait object）
  - `variables: HashMap<TensorId, TensorStatus>` — tensor 状态追踪
- [ ] 三操作示例的入队过程：`clone` → `ScalarMul` → `ScalarAdd` → `Tanh`
- [ ] 为什么 `println!("{}", z)` 才触发执行：`Display` → `display_fmt_impl` → `into_data()` → `FusionTensor::into_data()` → `client.read_tensor_float()` → `submit_blocking` → `server.drain_stream()`
- [ ] 对比：`Fusion<B>` 下操作被推迟 vs 裸 `CubeBackend` 下操作立即执行
- [ ] 作业：在 `println!` 前后各加一条 `println!` 观察时序；验证不移除 `println!` 就不会执行

### 第三章（待写）：Drain 与 Processor

- [ ] `MultiStream::drain()` 流程：
  1. `id.executes(|| ...)` — 在 stream ID 上下文中执行
  2. 取出目标 stream，调用 `processor.process(segment, store, ExecutionMode::Sync)` — 以 Sync 模式处理所有剩余操作
  3. 更新 `stream.cursor += num_executed` 追踪进度
  4. （tensor 清理在 `register()` 中通过 Drop op 触发 `shared_sources.remove()`，非 drain 直接负责）
- [ ] `Processor::process()` 主循环：`Policy::action()` 返回 Explore / Defer / Execute
- [ ] `Segment`（`StreamSegment`）抽象：给 Processor 提供对 queue 的独占访问
- [ ] `Action::Explore` → `Explorer::explore()` 寻找融合机会
- [ ] `Action::Execute(id)` → `segment.execute()` 执行已找到的融合计划
- [ ] `Action::Defer` → 仅在 Lazy 模式下合法
- [ ] 作业：用 `RUST_LOG=burn_fusion=trace` 观察 Policy 决策序列

### 第四章（待写）：增量融合——Block 注册与 Builder 评分

- [ ] `Explorer::explore()` 的两个阶段：`update()`（注册 op 到 block）→ `optimize()`（选出最优策略）
- [ ] `StreamOptimizer::register_inner()`：遍历所有 block，尝试注册 op
- [ ] `Block::register()` 的 accept/reject 规则：
  - 空 block 永远 accept
  - op 的 tensor ID 与 block 已有 tensor ID 有交集 → accept
  - 无交集且非 force → `NotPartOfTheGraph`
- [ ] 注册链路：`Block::register_op()` → 遍历 `builders`，调用 `builder.fuse(operation)` → 各 `OperationFuser` 实现（如 `TraceFuser`）内部将 op 推入 `FuseBlockBuilder.ops`
- [ ] `Block::optimize()`：`find_best_optimization_index` 遍历所有 builder，选 `properties().ready && score > best_score` 的最高分
- [ ] Builder 评分机制：`FuserProperties { score: u64, ready: bool }`；各 `OperationFuser` 实现自行定义 score 计算方式；框架只做最大分比较
- [ ] `BlocksOptimizer` 的 merging_pass：尝试合并相邻 block
- [ ] 作业：改示例加一个 matmul 操作，观察融合日志中 builder 选择变化

### 第五章（待写）：FuseBlockBuilder——reads、writes 与数据流分析

- [ ] `FuseBlockBuilder` 核心字段（`engine/trace/block.rs:36–49`）：
  - `ops: Vec<FuseOp>` — 操作序列
  - `reads: BTreeMap<TensorId, Vec<FuseOp>>` — 所有被读 tensor（含 input/swap_dims/reshaped 的 Assign op）
  - `writes: BTreeMap<TensorId, Vec<FuseOp>>` — 所有被写 tensor（multi-block 间传值 + 最终输出）
  - `outputs: RegisteredTensors` — 本 block 声明的输出 tensor
  - `locals: LocalVariablePool` — 局部变量分配池（按 `(FuseType, TensorId)` 映射到 `FuseArg::BlockLocal`）
  - `tensor_writes()` **是方法**（`pub fn tensor_writes(&self, ...) -> RegisteredTensors`），**不是字段**——做数据流分析，遍历 `outputs` 判断哪些需要写全局内存
- [ ] 中间结果可留在寄存器/共享内存；仅最终输出需写全局内存
- [ ] `tensor_writes()` 数据流分析算法（遍历 `self.outputs`）：
  - 检查 `resources.outputs` 中对应 tensor 的 `TensorStatus`：若不是 `ReadWrite` → 标记为全局内存写
  - 若是 `ReadWrite` 且已在 `resources.buffers` 中且尚未写入 → 标记为全局内存写（并记录到 `buffers: &mut Vec<TensorId>` 避免重复写）
  - 仅被本 block 内部消费的中间结果不写入
- [ ] `FuseBlockBuilder::build()` → `FuseBlock`（含 `settings`/`ops`/`shape_ref`/`reads`/`writes`）
- [ ] `TraceFuser`（`engine/trace/fuser.rs`）：管理 `FuseBlockBuilder` 的生命周期，调用 `tensor_writes()` 推算各 block 的 buffer 写入次数用于性能估计
- [ ] 作业：用三操作示例，手绘 reads/writes/tensor_writes 的 BTreeMap 内容

### 第六章（待写）：从 FuseTrace 到 kernel launch

- [ ] `FuseTrace` 的结构：`blocks: Vec<FuseBlock>` + `resources: FuseResources`
- [ ] `FuseResources`：`inputs`/`outputs`/`scalars`/`dropped`
- [ ] Launch 规划三剑客：
  - `InputPlanner`：确定哪些 tensor 作为 kernel 输入
  - `OutputPlanner`：确定哪些 tensor 是 kernel 输出
  - `VectorizationPlanner`：确定向量化策略
- [ ] `LaunchPlanExecutor::execute()`：实际启动 kernel
- [ ] 用 `RUST_LOG=cubecl_wgpu::runtime=trace` 观察后端生成的 shader（WGPU: 经过 naga 翻译；CUDA: 经过 NVRTC → PTX）
- [ ] 作业：对比 fused kernel 的 shader 代码和三操作分别执行的 shader，数内存读写次数

### 第七章（待写）：`elemwise_fuse` kernel

- [ ] `elemwise_fuse` 的 `#[cube(launch_unchecked)]` 签名
- [ ] `FuseBlockConfig` 作为 `#[comptime]` 参数：携带 op 序列
- [ ] `#[unroll]` 循环展开 ops：`for index in 0..config.ops.len()`
- [ ] `comptime!{ config.ops.index(index) }` 在 JIT 编译时解析为具体 op
- [ ] comptime 特化：每种 op 组合生成不同的 JIT 产物（编译期/特化，非运行时代码生成）
- [ ] `fuse_on_write` 函数：处理 op 的读写、中间结果复用
- [ ] 作业（高阶）：改 elemwise_fuse 的 comptime config 加一个自定义 op 类型，验证不同 op 组合产生不同 JIT key

### 第八章（待写）：跨流、不可融合与 v0.21.0

- [ ] 跨流共享的完整流程：`tag_shared_view` drain 源流 → 目标流指向同一 `Arc<GpuBuffer>`
- [ ] 不可融合的情况：
  - `ExecutionStrategy::Operations`：逐 op 执行
  - `ExecutionStrategy::Composed`：混合 fused + unfused 段
- [ ] v0.21.0 channel 重构细节：
  - 旧架构：`DeviceHandle` 内递归锁，Fusion + CubeCL 在同一锁里串行
  - 新架构：`submit()` fire-and-forget → worker 线程池；`submit_blocking()` 用于 `read_float`
  - `DeviceServiceStage::Upstream`：融合服务在 CubeCL 上游，流水线并行
- [ ] fusion log table：`RUST_LOG=burn_fusion=full` 输出融合执行表
- [ ] 作业：开两个线程各创建一个 stream，验证跨流 tensor 共享的 `tag_shared_view` 行为

---

## 与 CubeCL 专题的关系

| 维度 | Burn Fusion 专题（本专题） | CubeCL 专题 |
|------|---------------------------|------------|
| **推迟什么** | 连续 op 如何合并、何时 drain | 某次 launch 用哪份 GPU 代码、哪种 tile |
| **决策粒度** | 操作序列的融合策略 | 单次 kernel 的编译器实现 |
| **核心类型** | `MultiStream` / `Block<O>` / `FuseTrace` | `#[cube]` / `Scope` / `KernelDefinition` |
| **时机** | 运行期（读张量前 drain） | 首次 launch（JIT miss） |
| **交汇点** | FuseTrace 交给 `LaunchPlanExecutor` | CubeCL runtime 编译 + 执行 |

两专题的第 8 章最终交汇：Burn 专题 §8 覆盖 `submit()`/`submit_blocking()` 与 `DeviceServiceStage` 的 flow control；CubeCL 专题 §8 覆盖 `Backend` trait → `burn-cubecl` → CubeK → CubeCL 的调用链。两篇合起来给出从用户代码到 PTX 的完整路径。

---

## 进度

| 状态 | 文档 |
|------|------|
| ✅ 已写 | `1-client-server.md` |
| 📋 待写 | `2-operation-queue.md` … `8-cross-stream-channel.md` |
| 📎 地图 | `../summary.md` |
| 📎 本计划 | `index.md` |
| 📎 外部草稿 | [Gist](https://gist.github.com/nihalpasham/fc128f074e20d880bfd97198c2ac784b) |

---

## 系列导航（Burn 底层机制系列内）

| 篇 | 文档 | 状态 |
|:---:|------|------|
| 地图 | [../summary.md](../summary.md) | 已发布 |
| 计划 | **本文** | 新发布 |
| ONNX | [../onnx-summary.md](../onnx-summary.md) | 已发布 |
| 专题 1 | [1-client-server.md](1-client-server.md) | 已发布 |
| 专题 2–8 | `2-operation-queue.md` … | 待写 |

*Burn 底层机制系列 · 综合地图见 [README](../../../README.md)*
