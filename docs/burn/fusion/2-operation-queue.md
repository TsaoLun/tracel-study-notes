# OperationQueue：惰性执行与"推迟了什么"

> **归档说明**：本篇属于 [Fusion 专题](index.md) 的第二章（专题已归档为可选延伸阅读）。机制分析仍准确，源码锚点见各处路径。核心机制的全景见 [Fusion 系统设计](../kernel-fusion-system-design.md)。

> **导读** · 难度：中等 · 预计 ~25 分钟 + 练习 · [学习地图](../../../README.md#学习地图) 阶段 3
>
> - **读前应知道**：[第一章](1-client-server.md) 的双 client-server 链路；`Tensor::from_data` 在 Fusion 层只入队一个 `NoOp`，真正的 buffer 分配走 CubeCL 链。
> - **本篇回答**：操作入队时到底存了什么、为什么不执行；`println!` 这一行如何把入队的东西逼出来。
> - **配套练习**：[src/fusion-ch2-queue](../../../src/fusion-ch2-queue/) — 用 `BURN_FUSION_LOG` 观察入队时序与"不移除 `println!` 就不执行"。

第一章跟到 `FusionServer` 收到一个 `NoOp`：tensor 创建在 Fusion 层只占一个 id，GPU buffer 在 CubeCL 链上立即分配。本章跟进一个问题——**后续的 `* 2.0`、`+ 1.0`、`.tanh()` 这三步操作，在 Fusion 层发生了什么？为什么直到 `println!("{}", z)` 才真正跑？**

答案落在 `OperationQueue` 这个数据结构上。它是一条**单流**的操作队列，负责把操作描述存下来、但先不执行。

## `OperationQueue`：五个字段，各管一件事

`burn/crates/burn-fusion/src/stream/queue/base.rs:13`：

```rust
pub struct OperationQueue<R: FusionRuntime> {
    /// 精确 tensor ID + shape，执行 kernel 时要这些真值。
    pub(crate) global: Vec<OperationIr>,
    /// 相对表示：tensor ID/shape 被替换成局部编号，
    /// 只够判断"这俩 op 能不能融合"，不够真跑。
    pub(crate) relative: Vec<OperationIr>,
    /// global ↔ relative 的转换器（id/shape 的映射表）。
    pub(crate) converter: OperationConverter,
    /// 真正能跑的 op 包装（UnfusedOp，具体类型，非 trait object）。
    pub(crate) operations: Vec<UnfusedOp<R>>,
    /// tensor 当前状态（Read/Write/ReadWrite），用于数据流分析。
    pub(crate) variables: HashMap<TensorId, TensorStatus>,
}
```

五个字段两两配对：

| 字段 | 存什么 | 用途 |
|------|--------|------|
| `global` | 带**真** tensor ID 和 shape 的 `OperationIr` | 真跑时按这个调 kernel |
| `relative` | 同一 `OperationIr`，但 id/shape 换成局部编号 | 判定融合机会（不同 shape 的同模式可复用同一条优化） |
| `converter` | global↔relative 的映射表 | 维持两边对应关系 |
| `operations` | 与前三条**等长**，每条对应一个 `UnfusedOp<R>` | 融合失败时按这个逐 op 执行 |
| `variables` | 每个 tensor 当前的 `TensorStatus` | 给后续数据流分析（第五章）用 |

注意 `global`、`relative`、`operations` 三条 `Vec` **始终等长**——同一个 op 在三个表里各占一行。`variables` 是去重的 `HashMap`（同一个 tensor 被多 op 引用只存最新状态）。

## 入队：一次 `add` 干三件事

入队入口是 `OperationQueue::add`（`queue/base.rs:51`）：

```rust
pub fn add(&mut self, global: OperationIr, operation: UnfusedOp<R>) {
    for node in global.nodes() {
        self.variables.insert(node.id, node.status);   // 1. 更新 tensor 状态
    }
    let relative = global.to_relative(&mut self.converter);  // 2. 算相对表示
    self.relative.push(relative);                            // 3. 三表各 push 一行
    self.global.push(global);
    self.operations.push(operation);
}
```

三步对应三个表各长一行。`to_relative`（`stream/context.rs:182`）做的事是把 `OperationIr` 里的每个 `TensorIr.id` 查 `converter.tensors_global2relative`——查到就替换成局部 id，查不到就分配一个新局部 id 并登记双向映射。shape 同理（`shapes_global2relative`）。这样 `relative` 里出现的永远是"局部第 0、1、2 个 tensor"这种编号，shape 也归一化——两段 shape 相同但 tensor id 不同的 op 链，`relative` 会长得一模一样，融合引擎据此复用同一份优化方案。

> **为什么分两份**：融合判定只关心"操作的模式"（谁读谁写、op 类型序列），不关心具体 tensor 是哪一号。`relative` 把模式剥出来。真执行时还要拿真 id 调 kernel，所以 `global` 原样留着。两份各司其职，`converter` 做翻译。

### 主示例的四步入队

回到 `z = (x.clone() * 2.0 + 1.0).tanh()`，在 Fusion 层这产生四次 `add`（`clone` 也算一次 op）：

```
add #1: OperationIr::BaseFloat(Clone)       — global 写真 id, relative 写 0
add #2: OperationIr::BaseFloat(ScalarMul)   — 读 x, 写新 tensor
add #3: OperationIr::BaseFloat(ScalarAdd)   — 读上一步结果, 写新 tensor
add #4: OperationIr::Activation(Tanh)       — 读上一步结果, 写 z
```

四次入队后，`global` / `relative` / `operations` 各长 4 行，`variables` 记下每个 tensor 的最新状态。**没有任何一次 GPU 执行发生。**

## 谁触发 `add`：从用户代码到 `queue.add`

调用链（前半段在客户端，后半段在服务端线程）：

```
Tensor::mul(x, 2.0)
  → Fusion<B>::float_mul     （Fusion 层 trait 实现）
  → 构造 OperationIr::BaseFloat(ScalarMul)
  → GlobalFusionClient::register(...)
      → server.submit(move |server| server.register(stream, repr, op))   // client.rs:130
          ↓ 投递到 worker 线程
        FusionServer::register → MultiStream::register                     // server.rs:36
          → enqueue_operation
              → s.queue.add(repr, operation)                               // multi.rs:206
              → s.processor.process(segment, ..., ExecutionMode::Lazy)     // multi.rs:209
```

最后那行 `processor.process(..., ExecutionMode::Lazy)` 是关键：入队后**立刻**调一次 Processor，但传的是 `Lazy` 模式——Processor 在 `Lazy` 下最多做 `Action::Explore`（找融合机会），不触发 `Execute`。所以入队≠执行，入队只是"把 op 存进三个表，顺便让融合引擎看一眼能不能攒一起"。

`ExecutionMode` 的两个值（`stream/execution/processor.rs`）：

- `Lazy`：每入队一个 op 就调一次，只允许 Explore/Defer，不允许 Execute。
- `Sync`：drain 时用，要求把队列里剩余的 op 全部执行完。

这正是"惰性"的机械含义：`Lazy` 模式下 Processor 即使被调用也不执行。

## 什么把 op 逼出来：`println!` → drain

第三章详讲 drain，这里先看触发链。`println!("{}", z)` 调 `Tensor` 的 `Display`：

```
Tensor::fmt (Display)
  → display_fmt_impl → into_data()
  → FusionTensor::into_data()
  → client.read_tensor_float::<B>(desc, id)          // tensor.rs:177
      → server.submit_blocking(move |server| server.float_data::<B>(tensor, stream))
          ↓ 阻塞在 worker 线程上等结果
        FusionServer::float_data → read_float::<B>    // server.rs:50
          → self.drain_stream(id)                      // server.rs:56 ★
              → MultiStream::drain(handles, id)        // multi.rs:248
```

`drain_stream` 是阻塞调用（`submit_blocking`），它把当前 stream 的队列以 `ExecutionMode::Sync` 跑完，再把 z 的数据读回主机给 `Display` 用。所以**读张量值是 Fusion 的同步点**——任何要把 tensor 数据搬回主机的操作（`println!`、`into_data()`、`to_vec()`、loss 打印）都会 drain。

### 不移除 `println!` 就不执行——这是可验证的

把 `src/burn-test/src/main.rs` 里的 `println!("{z}")` 删掉，再跑 `cargo run --release`：程序正常退出，但 GPU 上**一个 kernel 都没跑**。三步操作的 `OperationIr` 留在队列里，`FusionTensor` 被 drop 时入队的 `Drop` op 也只是登记——没有任何同步点把它们逼出来。进程结束，队列随 `FusionServer` 一起释放。

这是惰性执行的硬约束：**没有同步点 = 没有执行**。`println!` 不是"打印"，它是"读 tensor 值"，读这个动作触发了 drain。

## 对比：裸 `CubeBackend` 下操作立即执行

若把 device 换成不带 Fusion 的后端（如 `NdArray` 或关掉 fusion 的 wgpu），`Tensor::mul` 直接调到 `B::float_mul`——立即在设备上执行，返回结果 tensor。没有队列、没有 `relative`、没有 `Lazy`/`Sync` 之分。代价是三步操作各跑一个 kernel，launch 开销 ×3（见 [系统设计](../kernel-fusion-system-design.md) 的开销对比）。

Fusion 层用"入队 + 延迟 drain"换来了"攒到一起再决定怎么跑"的机会——这正是融合的前提。推迟的不是计算结果，是**决策时机**：把"这三步怎么跑"的决策拖到必须出结果的那一刻，此时已经攒齐了一串 op，融合引擎才看得到模式。

---

## 小结

- `OperationQueue` 是单流操作队列，五个字段：`global`（真 id）/`relative`（局部编号）/`converter`（翻译表）/`operations`（可执行包装）/`variables`（tensor 状态）。前三条等长，一一对应。
- 入队 = `add` 三步：更新 `variables`、算 `relative`、三表各 push 一行。**不执行**。
- 入队后立刻调 `processor.process(Lazy)`，但 `Lazy` 模式下不触发 Execute——这是惰性的机械含义。
- 同步点是"读 tensor 值"：`println!` / `into_data()` / `to_vec()` 都走 `read_tensor_float` → `drain_stream` → `MultiStream::drain(Sync)`，把队列跑空。
- 删掉所有读 tensor 值的代码，op 就留在队列里永远不执行——惰性执行的硬约束。

下一章展开 `MultiStream::drain` 与 `Processor` 的 `Policy` 状态机：`Explore`/`Defer`/`Execute` 三态如何在 drain 全场中决策、`Sync` 与 `Lazy` 的行为差异。

---

## 动手改

先预测，再跑 `BURN_FUSION_LOG=full cargo run --release` 验证。改 `src/fusion-ch2-queue/src/lib.rs` 的 `main` 与测试。

1. **观察入队时序**：在三步操作之间各插一条 `eprintln!("after step N")`（主机侧 `eprintln` 立即输出，不走 Fusion），预测这些 `eprintln` 会**全部先于** fusion 日志出现——因为 fusion 日志在 drain 时才打，而 `eprintln` 在入队时就打。验证点：三条 `after step` 全部出现在 fusion execution table 之前。

2. **验证"不读不跑"**：把 `println!("{z}")` 注释掉，再加一条 `std::thread::sleep(Duration::from_secs(2))` 让程序停留。预测 fusion 日志一片空白（无 execution table），且程序正常退出。验证点：无 `[fusion]` execution table 输出。

3. **换个同步点**：把 `println!("{z}")` 换成 `let _ = z.into_data();`（只读不打印）。预测 drain 仍触发，fusion execution table 照常出现——因为 `into_data()` 同样走 `read_tensor_float`。验证点：execution table 出现，四 op 融合为一。

> 自证测试：作业 3 的对照版在 `cargo test into_data_triggers_drain`——它用 `into_data()` 读取后断言数值正确，并确认 `BURN_FUSION_LOG=full` 下 execution table 非空（弱断言：数值正确即证明 drain 发生）。

---

## 下章预告

**[第三章 · Drain 与 Processor：Policy 状态机](3-drain-processor.md)**（待写）：`MultiStream::drain` 全场；Policy 的 `Explore`/`Defer`/`Execute` 三态决策；`ExecutionMode::Sync` 与 `Lazy` 的行为差异；`Segment` 抽象如何给 Processor 独占访问队列。

---

## 系列导航

| 篇 | 文档 |
|:---:|------|
| 地图 | [../summary.md](../summary.md) |
| 计划 | [index.md](index.md) |
| 专题 1 | [1-client-server.md](1-client-server.md) |
| **专题 2** | **本文** |
| 专题 3–8 | 见 [计划表](index.md#章节目录) |

*Fusion 专题 · 源码 walkthrough · [阅读路径](../../../README.md)（可选的延伸阅读）*
