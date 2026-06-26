# Burn 的 Autodiff 系统：装饰器模式、类型状态图构建与惰性检查点

> `Autodiff<B, C>` 是一个装饰器——不像 PyTorch 把 autograd 嵌入 tensor 运行时，Burn 把 autodiff 做成可选层：`78f10aec1` 起用户通过 `Device::wgpu(...).autodiff()` 路由到 `Autodiff<B>` 后端，叶子 tensor `require_grad()` 才参与梯度图；cargo `autodiff` feature 控制是否链接 autodiff crate（推理二进制可整段排除）。反向图在 BFS 逆序执行后自动销毁。

> **导读** · 难度：中等偏难 · 预计 ~60 分钟 + 练习 · [学习地图](../../README.md#学习地图) 阶段 7
>
> - **读前应知道**：backprop 算什么、训练与推理为何不同（见 [primer · Part A](../primer.md#part-a--领域最小集)）；装饰器在类型栈最外层（[architecture.md](../architecture.md)）
> - **AI infra 通用映射**：autograd 放在架构哪一层是通用设计选择，对比 PyTorch 把 autograd 嵌入 tensor（`grad_fn`）（基线见 [primer · Part B](../primer.md#part-b--对比基线速查)）。
> - **本篇回答**：(1) 前向时梯度图如何构建；(2) 反向为何绕开 fusion 直接调内层后端；(3) 检查点策略如何在内存与重算之间取舍
> - **配套练习**：[src/autodiff-test](../../src/autodiff-test/) — 验证 `z = tanh(x*2.0+1.0)` 的梯度

## Autodiff 在框架中的位置

回顾 [Fusion 篇](kernel-fusion-system-design.md)：Burn 的后端是可组合的。默认 `Device::wgpu(...)` 展开为 `Fusion<CubeBackend<WgpuRuntime<...>>>`（推理，无 Autodiff）；`.autodiff()` 后在 dispatch 层外包 `Autodiff<...>`，Autodiff 在最外层。前向操作先经 autodiff 记录梯度图，再入 fusion 引擎排队执行。反向传播则完全绕开 fusion，直接调用内层后端。

PyTorch 的方案是将 autograd 引擎深度耦合到 tensor 类型和 C++ 运行时中。Burn 走了一条不同的路：**Autodiff 是一个装饰器（decorator），包裹在任意后端外面**。

```rust
// burn/crates/burn-autodiff/src/backend.rs:22
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}
```

`B` 是任意实现了 `Backend` trait 的后端（`CubeCL`、`NdArray` 等）。`C` 是检查点策略（`NoCheckpointing` 或 `BalancedCheckpointing`）。`Autodiff<B>` 自身也实现 `Backend`——这意味着你可以写：

```rust
type MyBackend = Autodiff<Fusion<CubeBackend<WgpuRuntime<AutoCompiler>>>>;
```

嵌套顺序决定了执行语义：`Autodiff` 在最外层，所有前向操作先经过 autodiff 记录 gradient tape，再进入 fusion 引擎优化执行。Fusion 看不到 autodiff 的 tensor 包装——`ad_enabled()` 返回 `false`（`burn/crates/burn-fusion/src/backend.rs:51`）。

这个架构的核心影响是 **autodiff 和 fusion 完全解耦**。前向融合发生在前向执行时，autodiff 记录发生在融合之前。反向传播则完全绕开 fusion 引擎，直接调用内层后端的操作。融合引擎的系统设计见 [Burn Kernel Fusion 系统设计](kernel-fusion-system-design.md)。

---

## 图构建：Tape-based 而非源码变换

### 和 PyTorch 一致：eager recording

Burn 的 autodiff 是 tape-based 的，和 PyTorch 一样。每个前向操作在执行的同时注册其反向步骤，不事后分析计算图。

每个张量操作（如 `float_add`）的实现遵循统一模式（`burn/crates/burn-autodiff/src/ops/tensor.rs:140`）：

1. 定义一个单元结构体作为反向步骤的标识（如 `struct Add;`）
2. 为该结构体实现 `Backward<B, N>` trait（`N` 是父 tensor 数量）
3. 通过类型状态构建器（`OpsPrep`）注册到图中

```rust
// 示意：每个 op 的结构
impl<B: Backend> Backward<B, 2> for Add {
    type State = (Shape, Shape);  // 向后传播时需要的状态

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _checkpointer) {
        // 链式法则：梯度流向两个父节点
        let grad = grads.consume::<B>(&ops.node);  // 消费输出节点的梯度
        binary::<B, _, _>(ops.parents, ops.node, grads,
            |grad| broadcast_shape(grad, &shape_lhs),  // 流向 lhs 的梯度
            |grad| broadcast_shape(grad, &shape_rhs),  // 流向 rhs 的梯度
        );
    }
}
```

`binary()` 和 `unary()` 辅助函数（`ops/backward.rs:49`）处理"消费当前节点的梯度 → 按需要复制到各父节点 → 注册到 Gradients 容器"的标准流程。

> ▶ **动手**：`cd src/autodiff-test && cargo test -- --nocapture`
> 验证 `z = tanh(x*2.0+1.0)` 的梯度计算。注意构造 device 时必须 `.autodiff()`，叶子 tensor 要 `require_grad()`。观察 `z.backward()` 返回的 `Gradients` 容器，以及 `x.grad(&grads)` 提取出的梯度值。运行带 `BURN_FUSION_LOG=full` 可同时观察 autodiff 触发的前向 fusion 执行。

### 核心数据结构

图被表示为一个扁平的 `HashMap<NodeId, StepBoxed>`，存在 `AutodiffServer` 中（`burn/crates/burn-autodiff/src/runtime/server.rs:31`）：

```rust
pub struct AutodiffServer {
    steps: HashMap<NodeId, StepBoxed>,           // 节点 ID → 反向步骤
    actions_builder: HashMap<NodeId, CheckpointerBuilder>,  // 检查点动作
    memory_management: GraphMemoryManagement,    // 节点生命周期追踪
}
```

**`Node`**（`graph/node.rs:42`）是图的基本单元：
```rust
pub struct Node {
    pub parents: Vec<Parent>,           // 输入张量节点
    pub order: usize,                   // 拓扑序（从叶子节点递增）
    pub id: NodeId,                     // 全局唯一 ID（AtomicU64 计数器）
    pub requirement: Requirement,       // 梯度需求标记
    pub properties: ComputingProperty,  // 计算/内存分类
    pub client: AutodiffClientImpl,     // 挂载的图客户端
}
```

**`NodeId`** 是从全局 `AtomicU64` 生成的，保证跨线程唯一。**`NodeRef`** 是 `Arc<Node>`——clone 成本是引用计数递增，支持 tensor 被多个图引用。

**`Requirement`**（`graph/requirement.rs:5`）：
- `Grad`：叶子 tensor，用户显式调用了 `.require_grad()`
- `GradInBackward`：中间节点，反向传播需要它的梯度
- `None`：不需要梯度，不参与图构建

只有 `Requirement != None` 的父节点才会被加入新节点的 `parents` 列表。这自然剪掉了不需要梯度的子图。

---

## 反向执行：BFS 分层 + 逆序执行

反向传播在 `AutodiffServer::backward()`（`runtime/server.rs:62`）中通过两步完成：

### 第一步：构建 tape

```rust
fn build_tape(&self, root: NodeRef) -> Vec<Vec<StepBoxed>> {
    // BFS 遍历从 root 开始，按 depth (order) 分层
    BreadthFirstSearch::traverse(root, &self.steps)
}
```

`BreadthFirstSearch::traverse()`（`graph/traversal.rs:22`）从根节点做 BFS，`AutodiffServer::build_tape()`（`runtime/server.rs:133`）在 BFS 回调中按 `step.depth()`（即 `order` 字段）将每个 step 分组到对应深度的 `Vec` 中。

### 第二步：逆序执行 tape

```rust
fn execute_steps(tape, mut grads, mut checkpointer) -> Gradients {
    tape.into_iter().rev().for_each(|steps| {
        steps.into_iter().for_each(|step| step.step(&mut grads, &mut checkpointer))
    });
    grads
}
```

从最深的分层开始逆序执行——深度最大的节点先执行（它们是离 root 最远的叶子），确保在被上层节点消费梯度之前，所有依赖的梯度都已计算完成。

### 梯度累积（共享权重）

当多个反向路径向同一个节点贡献梯度时，`Gradients::register()`（`grads.rs:131`）检测到目标 `NodeId` 已有梯度：

```rust
if let Some(tensor_old) = self.container.remove(&id) {
    let tensor = B::float_add(value, tensor_old);  // 累加
    self.container.register(id, tensor);
}
```

这是处理共享权重（如 weight tying）的自然方式——不需要显式的 "merge graph"，梯度自动累加。`GraphLocator` 只在父节点碰撞时合并两个 `AutodiffServer`（`runtime/graph.rs:250`），这在多设备或多计算流中才会发生。

### 分布式梯度同步

当训练跨多 GPU 时，每个设备独立计算局部梯度，需要在所有设备之间同步。Burn 将同步钩子注入 `Gradients::register()` 中（`grads.rs:131`）：每次梯度注册时检查是否需要触发 `all_reduce`。

`DistributedGradientRegistration`（`burn/crates/burn-autodiff/src/distributed.rs:45`）维护一个 `n_required_map: HashMap<NodeId, usize>`，以引用计数跟踪每个参数还有多少路梯度待注册。当计数归零（所有对该参数有贡献的路径都已完成反向传播），才提交 `submit_gradient_sync`：

```rust
// distributed.rs:45-58（示意）
fn on_register(&mut self, id: &NodeId, container: &mut TensorContainer) {
    if let Some(params) = self.sharded_parameters_map.get(id) {
        *self.n_required_map.get_mut(id).unwrap() -= 1;
        if *self.n_required == 0 {
            B::submit_gradient_sync(tensor, params);  // 触发 all_reduce
        }
    }
}
```

梯度同步通过 `Gradients` 容器上的 `on_register` 钩子在反向传播过程中内联触发，而非作为独立操作插入图中——梯度在注册完成后立即可用于同步，无需对整个梯度图做第二遍遍历。

---

## 检查点策略：计算密集 vs 内存密集

Burn 的检查点设计是这个系统中最精巧的部分。每个前向操作被分类为两种属性（`graph/node.rs:23`）：

```rust
pub enum ComputingProperty {
    ComputeBound,                                      // 保留前向输出
    MemoryBound { retro_forward: Arc<dyn RetroForward> }, // 丢弃，按需重算
    Ambiguous,
}
```

**`ComputeBound`**（如 `matmul`、`conv2d`、`embedding`、`gather`、`scatter`、`pooling`、`ctc_loss`）：前向输出在反向传播时需要且计算成本高，不能随意丢弃——保留在内存中。

**`MemoryBound`**（如 `Add`、`Mul`、`Neg`、`Exp`、`Tanh`、`Sigmoid`、`sqrt`、`abs`、`reshape`、`select`、`slice`、`permute`）：前向输出同样需要，但重新计算成本极低。反向传播时通过 `RetroForward` 闭包重新计算。

分类是由 op 实现者决定的，不是运行时启发式。`NoCheckpointing` 策略覆盖所有 op 为 `ComputeBound`（忽略 op 的标记）；`BalancedCheckpointing` 尊重标记分类。

```rust
// burn/crates/burn-autodiff/src/checkpoint/retro_forward.rs
pub trait RetroForward: Debug + Send + 'static {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId);
}
```

每个 MemoryBound 操作通过宏生成一个 `RetroForward` 实现（如 `RetroAdd` 对 `lhs + rhs`）：

```rust
retro_binary!(RetroAdd, |lhs: FloatTensor<B>, rhs: FloatTensor<B>| B::float_add(lhs, rhs));
```

在反向传播时，当一个 MemoryBound 操作的反向需要其前向输出，它调用 `checkpointer.retrieve_node_output()`（`checkpoint/base.rs:35`）。检查点器对需要重算的节点做拓扑排序，按顺序执行 `RetroForward::forward()` 重建输出，并缓存为 `State::Computed`。`n_required` 计数器跟踪一个节点被引用的次数——最后一次引用时释放。

### 两种策略

- **`NoCheckpointing`**（默认）：所有操作视为 ComputeBound。`RetroForward` 不生成。适合 GPU 内存充足的场景。
- **`BalancedCheckpointing`**（`checkpoint/strategy.rs`）：遵守 op 的分类标记。MemoryBound 的输出被丢弃，反向传播时重算。这是**粗粒度的检查点**——决策在 op 级别，不是用户手动标记分段。

与 PyTorch 的 `torch.utils.checkpoint.checkpoint()` 的区别：PyTorch 要求用户用 context manager 显式标记哪些 segment 需要检查点。Burn 的方案是**在 op 实现时注册分类，用户只选策略**——降低了使用门槛，但牺牲了灵活性（无法对特定 segment 设置自定义的检查点粒度）。

---

## 内存管理：图的生命周期

`GraphMemoryManagement`（`runtime/memory_management.rs`）通过 `Arc<Node>` 的强引用计数追踪节点是否仍可达。反向传播完成后：

1. 每个节点的 step 从 `AutodiffServer.steps` 中移除（在 `step()` 调用中消费）
2. `free_unavailable_nodes()` 清理引用计数归零的节点——包括其 `actions_builder` 和残留的 `steps` 条目
3. 因为中间梯度的 `GradInBackward` 需求在消费后被移除（`grads.consume()` 中对 `GradInBackward` 调用 `.remove()`），梯度容器在反向传播过程中逐步缩小

这个设计与 PyTorch 的 `retain_graph=True/False` 不同——Burn 的图总是在反向传播后被销毁，不提供图的持久化。这是**不支持高阶梯度**的根本原因：二阶梯度需要图在第一次反向传播后仍然存活。

---

## 不支持的特性

### 高阶梯度

Burn 没有 "grad of grad"。图在反向传播中被消费，`Gradients` 容器逐步缩小，没有机制保留图结构用于第二次反向传播。`hessian`、`double_backward` 等概念在当前实现中不存在。

### 反向融合

前向操作的融合发生在 fusion 引擎中（`Autodiff<Fusion<B>>`），反向传播直接调用内层后端。反向操作不会被融合——它们作为独立的 op 执行。在计算密集的场景中（如 transformer 的反向传播包含大量 element-wise 操作），这是一个潜在的优化空间，但当前未实现。

---

## 与 PyTorch Autograd 对比

**通用问题**：autograd 放在架构的哪一层、是否需要梯度由什么决定。任何 DL 框架都要选：把"需要梯度"做成运行时属性（动态、有开销），还是编译期类型差异（静态、推理零开销但切换要改类型）。

| 维度 | Burn | PyTorch |
|------|------|---------|
| 架构 | 装饰器 `Autodiff<B, C>`，编译期参数化 | 内置于 tensor 类型，C++ 运行时 |
| 图构建 | Tape-based，eager | Tape-based，eager |
| 图存储 | `HashMap<NodeId, StepBoxed>` 扁平结构 | `grad_fn` 指针链形成 DAG |
| 遍历 | BFS 分层 + 逆序执行 | 拓扑排序 |
| 共享权重 | `Gradients::register()` 累加梯度 | 自动在 autograd 引擎中 sum |
| 高阶梯度 | 不支持 | `create_graph=True` |
| 检查点 | op 级标记 + 策略参数化 | 手动 `checkpoint()` context |
| 后端解耦 | `Autodiff<B>` 通过 trait 泛型 | 耦合到 ATen tensor |
| 图生命周期 | 总是在反向传播中销毁 | `retain_graph=True/False` |
| 线程模型 | `GraphMutexClient`，per-device 锁 | GIL + C++ 线程锁 |

**谁该用哪个**：

- **要高阶梯度、要运行时切训练/推理、要动态灵活性** → PyTorch：`requires_grad_()` 是运行时 flag，`create_graph=True` 支持高阶梯度，`retain_graph` 控制图生命周期；代价是推理时 tensor 仍携带 `grad_fn` 指针与 autograd 引擎耦合，有运行时开销。
- **推理二进制要零 autograd 开销、后端要完全解耦、能接受编译期决定是否链接 autodiff crate** → Burn：`Device::wgpu(...)` 不调 `.autodiff()`（推理）与 `device.autodiff()`（训练）走不同 dispatch 路径，cargo `autodiff` feature 关闭时推理二进制整段排除 autodiff crate；代价是不支持高阶梯度、运行时切训练/推理需重新构造 device（PyTorch 的 `requires_grad_()` 是同一 tensor 上的运行时 flag）。

一句话：Burn 把 autodiff 做成装饰器 + cargo feature 编译期可选 + `device.autodiff()` 运行时路由，换推理时无 autograd 运行时开销与后端解耦，PyTorch 把它做成运行时属性换动态灵活与高阶梯度——选择取决于"你推理占比多大、要不要高阶梯度、能否接受编译期决定训练模式"。

---

## 限制

1. **无高阶梯度**：图在反向传播中消费。需要 Hessian 的场景无法实现。
2. **无反向融合**：反向传播的 element-wise 操作链独立执行，无法受益于 fusion 优化（融合引擎分析见 [Fusion 系统设计](kernel-fusion-system-design.md)）。这是在 `Autodiff<Fusion<B>>` 架构下的自然结果——autodiff 绕开了 fusion 层。
3. **检查点粒度固定于 op 级**：`MemoryBound` vs `ComputeBound` 是 op 级的分类，无法做 segment 级或 layer 级的手动检查点。`BalancedCheckpointing` 是一次性选择，没有 fine-tuning 的空间。
4. **图扁平化**：`HashMap<NodeId, StepBoxed>` 不保留图的层次结构。对于复杂图，BFS 重建拓扑信息有一定开销。但这对训练中计算图深度通常 < 1000 的场景是可接受的。
5. **类型状态构建器复杂性**：`OpsPrep` 使用 Rust 的类型状态模式（type-state pattern）确保在编译期强制"先标记 memory_bound → 再添加 retro_forward → 再标记 parents → 再 finish"的正确顺序。但这导致 op 实现较为冗长——每个 op 需要约 50 行 boilerplate。

---

## 关键源码入口

- 装饰器与后端 impl：`burn/crates/burn-autodiff/src/backend.rs`
- 核心 tensor 类型：`burn/crates/burn-autodiff/src/tensor.rs`
- 图数据结构：`burn/crates/burn-autodiff/src/graph/`（`node.rs`、`traversal.rs`、`requirement.rs`）
- 反向执行引擎：`burn/crates/burn-autodiff/src/runtime/server.rs`
- 操作与 Backward trait：`burn/crates/burn-autodiff/src/ops/tensor.rs`、`ops/backward.rs`
- 检查点：`burn/crates/burn-autodiff/src/checkpoint/`（`builder.rs`、`base.rs`、`retro_forward.rs`、`strategy.rs`）
- 梯度容器：`burn/crates/burn-autodiff/src/grads.rs`

---

## 本篇小结

读完你现在能回答：

- 装饰器 Autodiff 如何经 `device.autodiff()` 路由 + cargo feature 编译期可选，推理时整体排除
- 前向构图与反向 BFS 逆序执行的对应关系，以及反向为何绕开 fusion
- ComputeBound / MemoryBound 检查点策略各自的取舍

> ✓ **完成自检**：能对比 Burn 的装饰器 Autodiff 和 PyTorch 的内置 autograd——在架构位置、推理开销、高阶梯度、检查点粒度上的差异。

---

← [CubeK 架构纪律](../cubek/blueprint-routine-autotune.md) | → [全景篇](burn-systems-architecture.md)（返回入口）

动手：[src/autodiff-test/](../../src/autodiff-test/) — 运行实验观察梯度累积和检查点
