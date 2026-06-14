# Burn Autodiff：选择性包装、梯度图与 Checkpointing

## 读前须知

- **Autodiff 是什么**：Burn 的自动微分层——一个 Backend decorator（`Autodiff<B>`），只包装浮点张量，在 GPU 执行前向计算的同时在 CPU 上记录梯度图。`.backward()` 从图的叶子节点自动执行反向传播。
- **本文覆盖**：选择性包装策略（为何只包浮点）、`AutodiffTensor` 的双重身份（`primitive` + `node`）、梯度图的构建与遍历、checkpointing 策略（trade compute for memory）。本文是 Autodiff 机制的综合地图。
- **前置**：[Burn 地图](../summary.md)（类型栈、Backend trait 拆分）。本文聚焦 Autodiff 层内部的机制——梯度图如何记录、backward 如何执行、checkpointing 如何决定哪些中间结果保留。
- **机制基准**：burn 仓库 `crates/burn-autodiff/src/`。源码行号为近似值。

系列分工与导航见 [README](../../../README.md)。

---

## 架构一览

```
用户代码: loss.backward()
    ↓
Autodiff<B>::backward(tensor)
    ↓
梯度图 (Node Graph)
    ├── Node { parents, order, requirement, properties, client }
    ├── Parent { id }（有向边）
    └── Step trait（backward 执行单元）
    ↓
逆拓扑序遍历 + Gradients 累积
    ↓
Checkpointer（选择性保留/释放中间结果）
    ↓
B::FloatTensorPrimitive（底层后端的梯度张量）
```

---

## 核心结论

> `Autodiff<B>` 是一个零大小的编译期 wrapper——它通过 `BackendTypes` 的选择性覆盖只包装浮点张量（`FloatTensorPrimitive = AutodiffTensor<B>`），整数/布尔/量化张量原样透传。每个 `AutodiffTensor` 同时持有 GPU 上的数据（`primitive: B::FloatTensorPrimitive`）和 CPU 上的梯度图节点（`node: NodeRef`）。backward 沿图逆拓扑序遍历，每个 `Step` 执行该 op 的梯度计算，结果累加到 `Gradients` 中。

---

## 一、选择性包装：只包浮点，其余透传

`Autodiff<B, C>`（`crates/burn-autodiff/src/backend.rs`）对 `BackendTypes` 的实现：

```rust
impl<B: Backend, C: CheckpointStrategy> BackendTypes for Autodiff<B, C> {
    type FloatTensorPrimitive = AutodiffTensor<B>;          // 只替换这个
    type IntTensorPrimitive   = B::IntTensorPrimitive;       // 透传
    type BoolTensorPrimitive  = B::BoolTensorPrimitive;      // 透传
    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive; // 透传
}
```

**为什么只包浮点**：梯度是浮点值。整数运算（如索引计算）、布尔运算（如 mask）不需要梯度——包装它们只会增加不必要的 `node` 分配和图遍历开销。

这与 `Fusion<B>` 的包装策略形成对比——Fusion 包装全部四种张量，因为整数运算也能从融合中受益（减少 kernel launch）。两种策略的差异来自它们解决的问题不同：Autodiff 只关心"哪些操作需要记录梯度"（浮点）；Fusion 关心"哪些操作连续出现可以合并"（全部）。

---

## 二、`AutodiffTensor`：双重身份

`crates/burn-autodiff/src/tensor.rs`：

```rust
pub struct AutodiffTensor<B: Backend> {
    pub primitive: B::FloatTensorPrimitive,  // GPU 上的实际数据
    pub node: NodeRef,                       // 梯度图中的节点引用
    pub rc: NodeRefCount,                    // 图节点引用计数
}
```

两个字段代表两张"脸"：
- **`primitive`**：指向底层后端（如 `CubeBackend<WgpuRuntime>`）的浮点张量——实际数值在 GPU 上。前向计算的结果存这里。
- **`node`**：指向梯度图中的一个节点。节点记录"产生这个 tensor 的操作是什么、它的输入是谁（parents）"——足够在执行 backward 时调用正确的梯度算子。

`tensor.matmul(&other)` 同时做两件事：
1. 从 `primitive` 取出底层张量，调用 `B::float_matmul` 在 GPU 上执行矩阵乘法
2. 创建一个新的 `Node`——`parents: [self.node.id, other.node.id]`，`client` 记录 `MatmulBackward` 的梯度计算逻辑

前向计算的结果是新的 `AutodiffTensor { primitive: 矩阵乘法结果, node: 新节点 }`。

---

## 三、梯度图：Node、Parent、Step

梯度图的核心类型（`crates/burn-autodiff/src/graph/`）：

### Node

```rust
pub struct Node {
    pub parents: Vec<Parent>,           // 有向边：这个 tensor 是从哪些 tensor 算出来的
    pub order: usize,                   // 拓扑序编号
    pub id: NodeId,                     // 全局唯一 ID（AtomicU64 递增）
    pub requirement: Requirement,       // 是否需要梯度（叶子/中间/无需）
    pub properties: ComputingProperty,  // ComputeBound / MemoryBound / Ambiguous
    pub client: AutodiffClientImpl,     // 执行 backward step 的 client
}
```

### NodeRef = Arc\<Node\>

`AutodiffTensor` 中的 `node: NodeRef` 是 `Arc<Node>`——多个 tensor 可以共享同一个图节点。这在 clone 和分支操作中常见：`let y = x.clone()` 产生的新 tensor 指向同一个 `Node`，只是 `rc` 计数增加。

### Step trait

每个前向 op 对应一个 `Step` 实现——它定义了该 op 的**反向传播逻辑**：

```rust
pub trait Step {
    fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
    fn node(&self) -> NodeId;
    fn parents(&self) -> &[Parent];
    fn depth(&self) -> usize;
}
```

`Step::step` 被调用时执行：
1. 从 `grads` 中取出当前 tensor 的上游梯度
2. 从 `checkpointer` 中（或重新计算）获取前向计算时的输入
3. 计算相对于各 parent 的梯度
4. 将梯度累加回 `grads` 中对应 parent 的条目

---

## 四、Backward：逆拓扑序遍历

`AutodiffTensor::backward()`（`crates/burn-autodiff/src/tensor.rs`）触发梯度计算：

1. **构建 step 列表**：从 `self.node` 出发，沿 `parents` 边逆拓扑序遍历所有节点。每个节点的 `client` 生成对应的 `Step` trait object。
2. **逆序执行**：step 列表按深度降序排列——从输出节点（梯度为 1）开始，逐层向后传播到叶子节点。
3. **梯度累积**：每个 `Step::step` 的输出累加到 `Gradients` 中。当多个路径指向同一个 parent（如 `y = x + x`），该 parent 会收到两份梯度——`Gradients` 内部做累加而非覆盖。
4. **返回**：`Gradients` 收集所有叶子节点的梯度。用户通过 `tensor.grad(&grads)` 取出。

---

## 五、Checkpointing：用重计算换显存

`ComputingProperty` 枚举决定每个中间 tensor 的显存策略：

```rust
pub enum ComputingProperty {
    ComputeBound,                           // 计算密集型 → 保留中间结果
    MemoryBound { retro_forward: Arc<dyn RetroForward> },  // 显存密集型 → 释放，backward 时重算
    Ambiguous,                              // 不确定（留作未来 autotune）
}
```

- **ComputeBound**：前向计算的中间结果保留在显存中——backward 时直接读取，不重算。适用于计算量大但显存占用小的 op（如 elementwise）。
- **MemoryBound**：前向计算的中间结果在 forward 后释放——backward 时通过 `retro_forward` 重新计算。适用于显存占用大但重算成本低的 op（如某些 activation 的中间值）。

`CheckpointStrategy` trait（`crates/burn-autodiff/src/checkpoint/strategy.rs`）决定哪些 op 标记为 MemoryBound。默认策略 `NoCheckpointing`——所有中间结果保留，不重算。用户可在训练时选择不同的 checkpointing 策略来 trade 显存换计算。

---

## 六、Autodiff 与 Fusion 的分工

在完整训练栈 `Autodiff<Fusion<CubeBackend<WgpuRuntime>>>` 中：

| 层 | 在前向时做什么 | 在反向时做什么 |
|----|---------------|---------------|
| **Autodiff** | 拦截浮点 op，记录 Node 到梯度图 | 逆拓扑序遍历图，执行各 op 的梯度计算 |
| **Fusion** | 把连续 op 延迟入队，drain 时融合 | 不参与——反向 op 直接调内层后端，不经融合引擎 |
| **CubeBackend** | 执行融合后的 kernel（GPU 上） | 执行梯度 kernel（GPU 上） |

关键洞察：**Autodiff 的图节点在 CPU 上，Fusion 的融合在前向 GPU 路径上**。Autodiff 记录的是逻辑 op 序列，不必知道 Fusion 把哪些 op 合并了；反向路径绕开 Fusion 入队，因此当前无反向融合（见 [Autodiff 系统设计](../autodiff-system-design.md)）。

---

## 七、决策时机

| 决策 | 时机 | 层级 |
|------|------|------|
| Autodiff 是否包装（`Autodiff<B>` vs `B`） | `cargo build`（类型栈 monomorphization） | L1 |
| 每个 matmul/activation 记录为 Node | 前向计算时（CPU 侧） | 运行时 |
| `ComputingProperty`（ComputeBound vs MemoryBound） | Node 创建时（根据 op 类型判定） | 运行时 |
| 中间结果保留还是释放 | 前向计算时（Checkpointer 决定） | 运行时 |
| backward 触发 | 用户调用 `.backward()` | 运行时 |
| Step 执行顺序 | backward 时（逆拓扑序遍历） | 运行时 |
| 梯度张量的 GPU 计算与融合 | backward 时（由 Fusion + CubeCL 层处理） | 运行时 |

---

## 词汇说明表

| 术语 | 简要说明 |
|------|----------|
| **Autodiff\<B\>** | 零大小 Backend decorator（`PhantomData`），只替换 `FloatTensorPrimitive` 为 `AutodiffTensor<B>` |
| **AutodiffTensor** | `primitive`（GPU 数据）+ `node`（梯度图节点）+ `rc`（引用计数） |
| **Node** | 梯度图中的节点：parents（有向边）、order（拓扑序）、requirement、properties、client |
| **Step** | trait——定义某个前向 op 的反向传播逻辑：`step(grads, checkpointer)` |
| **Gradients** | 梯度容器：`HashMap<NodeId, B::FloatTensorPrimitive>`——支持多路径累加 |
| **Checkpointer** | 管理中间结果的保留/释放/重计算——trade compute for memory |
| **ComputingProperty** | ComputeBound（保留中间结果）/ MemoryBound（释放+重算）/ Ambiguous |
| **RetroForward** | MemoryBound 节点的重计算逻辑——backward 时重新执行前向计算 |
| **CheckpointStrategy** | trait——决定哪些 op 标记为 MemoryBound |
| **逆拓扑序遍历** | backward 的执行顺序：从输出节点沿 parents 边反向遍历，按深度降序执行 |
| **NodeRef** | `Arc<Node>`——多个 `AutodiffTensor` 可共享同一图节点 |

*Burn 底层机制 · Autodiff 地图 · 导航见 [README](../../../README.md)*
