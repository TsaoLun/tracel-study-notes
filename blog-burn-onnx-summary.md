# ONNX 模型编译：Burn ONNX 的构建时代码生成

## 读前须知

- **Burn ONNX 是什么**：在 `build.rs` 里运行的 **AOT 编译器**——把 ONNX 图翻译为可调试的 Rust 源码（`model.rs`）与权重（`model.bpk`），而非运行时加载 protobuf。
- **本文定位**：ONNX 编译器专文——以**注意力融合**为 hero pass，展开 IR 流水线、分区与测试体系。Burn 类型栈与融合流见 [blog-burn-summary.md](blog-burn-summary.md)；GPU JIT 见 [blog-cubecl-summary.md](blog-cubecl-summary.md)。
- **统计基准**：下文测试数字来自 `burn-onnx` 仓库 `crates/onnx-official-tests/expectations.toml`（ONNX v1.19.0，1615 条）；可用 [README 复验脚本](README.md#源码版本与数字校验) 刷新。

### 三篇分工

| 层次 | 文档 | 本文覆盖 |
|------|------|---------|
| 构建期 | **本文** | IR 流水线、codegen、测试 |
| 编译期 | [Burn 地图](blog-burn-summary.md) | 生成代码走 `Autodiff<Fusion<…>>` |
| 运行期 | Burn 地图 §五 | Fusion drain → burn-cubecl |
| GPU | [CubeCL 篇](blog-cubecl-summary.md) | JIT + autotune |

---

## 核心结论（读正文前的 spoiler）

> Burn ONNX 在**构建期**做运行时 loader 做不到的事：模式匹配（如 SDPA → `Attention`）、常量折叠、分区——输出是普通 Rust，运行时无 ORT、无 protobuf。与 ONNX Runtime 的区别如同 AOT 编译器 vs 解释器。生成代码与手写 Burn 模型走**同一条**融合流与 CubeCL JIT 路径。

---

若尚未建立「AOT vs loader」直觉，可先读 [Burn 地图 §七](blog-burn-summary.md#七onnx-入口构建期-aot而非运行时加载)（约 1 分钟）。下面从**最难的 pass** 直接切入。

---

## 最难的 pass：为什么注意力融合是编译器而不是 loader 的胜利

先不看流水线。先看一个具体的问题——它比其他任何东西都更能解释"为什么编译比加载难"。

PyTorch 的 `F.scaled_dot_product_attention(Q, K, V)` 在导出为 ONNX 时（尤其是旧版 opset），**不生成 `Attention` 算子**。它把注意力拆解为一串基础操作：

```
Q ──────────────────────┐
                         ├→ MatMul(Q, K^T) → [Div/Mul(scale)] → [Add(mask)]
K → Transpose(-2, -1) ──┘                                         │
                                                                   ↓
V ─────────────────────────────→ MatMul(scores, V) ←── Softmax(-1)
```

对于 ONNX Runtime，这不是问题——运行时就执行这 5 个操作呗。5 次 kernel launch，5 次显存读写中间结果。

对于 Burn ONNX，这**是一个机会**。编译器在生成代码之前**识别 SDPA 分解模式，融合为单一 `Attention` 节点**——生成代码调用 Burn 原生注意力，经 [Burn 融合流](blog-burn-summary.md) 与 [CubeK/CubeCL 内核](blog-cubecl-summary.md) 执行，而非 5 次独立 launch 的中间结果读写。

`crates/onnx-ir/src/simplify/coalesce_attention.rs`（约 1367 行）专门做这一件事。算法：

1. 找所有 `Softmax` 节点（注意力模式的"锚点"——所有 SDPA 变体都包含它）
2. 从 Softmax 向后追踪：它的输入应该来自一个可选的 `Add(mask)` → 可选的 `Div/Mul(scale)` → 一个 `MatMul`
3. 从 MatMul 向后追踪：两个输入——一个来自 Q（可能需要 `Transpose`），一个来自 `K → Transpose(-2, -1)`
4. 从 Softmax 向前追踪：它的输出应该被一个 `MatMul(scores, V)` 消费
5. 全部匹配后，用一个 `Attention { Q, K, V, scale, mask }` 节点替换这 5 个节点

难的不是标准模式。难的是**不同模型导出不同模式**：

| 变体 | Q 的变换 | K 的变换 | 出现在 |
|------|----------|----------|--------|
| 标准 | 无 | Transpose(-2, -1) | 大多数模型 |
| QK 预缩放 | Transpose([0,2,1,3]) → Mul(scale) | Transpose([0,2,3,1]) → Mul(scale) | RF-DETR |
| 仅 Q 预缩放 | Transpose → Mul(scale) | Transpose(-2, -1) | DINOv2 |
| 对称预缩放 | Mul(scale) | Transpose → Mul(scale) | DepthPro |

RF-DETR 的情况特别棘手：K 的 transpose 把 head-split 和 key-transpose 两个操作合并为一次 perm `[0,2,3,1]`（对比 Q 的 `[0,2,1,3]`）。匹配成功后，编译器需要**插入一个修正的 K Transpose** 来恢复标准的 K^T 语义。

这 1367 行做的事，运行时 loader 做不到：它**在编译期识别注意力模式**。ONNX Runtime 只看到 5 个独立操作；Burn ONNX 看到注意力，因为它是一个编译器——有时间停下来做模式匹配（`build_producer_map`、`build_consumer_map`、`is_single_use` 守卫），因为这一切发生在编译期。

注意力融合使上游套件中 **62 个 `attention_*_expanded` 测试里有 27 个** 达到 `pass`（`expectations.toml` 统计）。这不是 27 个独立 bug 的逐一修复——一个 pass 让整类生成代码质量跃升。

---

## 支撑这场胜利的流水线

注意力融合只是 **8 个 simplify pass** 中的 1 个（每轮迭代按固定顺序执行，定点循环最多 10 轮，通常 3–5 轮收敛）。它们运行在 **IR 流水线** 的 Phase 4b（`onnx-ir/src/pipeline.rs` 驱动）。总览：

| 阶段 | 模块 | 做什么 |
|:---:|------|--------|
| 1 | `initialization` | Protobuf → `RawNode`，常量初始化，mmap 外部权重 |
| 2 | `node_conversion` | 类型化节点 + `Gemm→Linear` 等早期合并 |
| 2b | 早期 constant fold | RNN 权重的 Slice→Concat 链折叠（最多 10 轮） |
| 3 | `type_inference` | 迭代形状/类型推断，`ScalarTensor` 区分 |
| 4 | `post_processing` | Identity 消除等 |
| 4b | `simplify` | **8 个 pass** 定点简化（含注意力融合） |
| 5 | `finalization` | 清理、图输出整理 |
| 6 | `convert_to_graph` | `RawNode` → `Node` 枚举（`pipeline.rs` Phase 6） |
| — | `burn-onnx` codegen | `ModelGen` / `impl_node_codegen_dispatch!` → `model.rs` |

下面按阶段说明**为什么需要这一步**，而不只是列名字。

### 你需要 Phase 1，因为 ONNX 是一个 protobuf 文件

`GraphProto` 是未类型化的 protobuf 对象。Phase 1（`phases/initialization.rs`）把它解析为 `RawNode` 的初胚，初始化器转为 `Constant` 节点。外部数据引用（>2GB 模型把权重放在 sidecar 文件中）在这里用 memory-mapped I/O 处理。

### 你需要 Phase 2，因为 protobuf 属性没有类型

`ProcessorRegistry` 查找每个操作类型的处理器，把 `{ name: "kernel_shape", ints: [3, 3] }` 这样的原始属性转为 `Conv2dConfig { kernel_shape: [3, 3], strides: [1, 1], ... }`。根据输入张量的维度确定 1d/2d/3d 变体。同时做早期模式合并（`Gemm → Linear`，`MatMul+Add → Linear`）。

### 你需要 Phase 2b，因为 PyTorch 的 RNN 导出有坏习惯

PyTorch 导出 RNN 权重时经常 `Slice`→`Concat`→`Unsqueeze` 链来重排——如果不在这里折叠为常量，Phase 3 的类型推断会被 "Dynamic" 阻塞。定点循环（最多 10 轮）处理连锁折叠。

### 你需要 Phase 3，因为只知道操作类型不知道张量形状

迭代式类型推断：收集输入偏好（某个输入是整数标量还是浮点张量）→ 同步已知类型 → 推断输出 → 检查收敛。0.21 引入的 `ScalarTensor` vs `ScalarNative` 区分在这里解决了"ONNX 里到处是 0 维张量，但 Rust 代码需要真的 `i64` 值"的问题。

### 你需要 Phase 4，因为图来自不同框架，各有各的习惯

消除 Identity 节点（`x = Identity(y)`→ 直接替换）。然后运行 **8 个 simplify pass** 的定点循环——除了注意力融合，还有：

- **Permute-reshape**：`Shape→Gather→Unsqueeze→Concat→Reshape`→`Transpose`（ONNX 中的维度重排惯用语）
- **Constant shape**：`Shape(x)→Gather(i)` 静态可解时折叠。**但裸 `Shape(x)` 被故意保留**——模型可能用静态维度导出，运行时接受动态输入
- **Constant folding**：全常量输入节点在编译期求值
- **Idempotent / Identity element elimination**：`Relu(Relu(x))`→`Relu(x)`，`x+0`→`x`
- **CSE + Dead code elimination**：合并重复节点，级联清除死节点

### 你需要 Phase 4b 是循环，因为 pass 之间有级联

CSE 合并 → 产生死节点 → DCE 清除 → 释放了被死路径独占的常量 → 下一轮 constant folding 折叠那个常量 → 可能暴露新的可 Idempotent 消除。3-5 轮收敛。

### 你需要 codegen，因为简化后的图需要变成 Rust

Phase 5–6 在 `onnx-ir` 内完成 finalization 与 `RawNode`→`Node` 转换。**Rust 源码生成**在 `burn-onnx` 的 `ModelGen`：`impl_node_codegen_dispatch!` 对 **169 种** `Node` 变体生成 match dispatch（`burn-onnx/src/burn/node_codegen.rs`）。ONNX 算子表（含 Conv1d/2d 等维度别名）见 `SUPPORTED-ONNX-OPS.md`（当前约 **168/209** import 支持）。

---

## 当图大到编译器吃不下

SDXL 的 UNet 有上万个节点。全部塞进一个 `forward()` 方法 = Rust 编译时间无法接受，甚至源码大到解析器拒绝处理。

`crates/burn-onnx/src/burn/partition.rs` 的算法做三件事：

1. **O(n) 扫描每个切分点的成本**：在节点 i 和 i+1 之间切一刀，有多少张量跨越边界。用前缀和差分数组（`delta[producer+1] += 1, delta[consumer] -= 1`）。
2. **贪心选切点**：将图切分为 64-256 节点的子模块，每个窗口内选切分成本最低的位置。
3. **常量重排**：把常量节点移到它们首个消费者的前面——避免一个权重在开头定义、在 500 个节点后才用，导致被迫跨越多个分区。

SDXL 和 Depth-Pro 的导入因为这 **437 行**（`partition.rs`）才可能。

---

## 怎么保证生成的代码是正确的？

三层测试，从锁死输出到验证数值：

1. **625 个快照测试**（`grep -r 'insta::assert_snapshot' burn-onnx --include '*.rs' | wc -l`）：`NodeCodegen` 分支用 `insta::assert_snapshot!` 锁死生成代码——改逻辑后 `cargo insta review`
2. **178 个集成测试**（`onnx-tests/tests/` 下 178 个子目录）：Python `onnx.reference.ReferenceEvaluator` 作 ground truth，Rust 逐元素对比
3. **1615 个上游测试**（ONNX v1.19.0，`expectations.toml`）：

| status | 含义 | 数量（当前） |
|--------|------|-------------|
| `pass` | codegen + compile + 数值匹配 | 722 |
| `fail-compare` | 编译运行但数值偏差 | 179 |
| `skip-compile` | codegen 成功但 Rust 不编译 | 230 |
| `skip-codegen` | codegen 失败/拒绝 | 484 |

在**能 codegen 且能编译**的集合（722+179=901）中，约 **80.1%** 数值通过（722/901）。完整 status 语义见 `crates/onnx-official-tests/README.md`。

测试门的一个特别设计：`expectations.toml`（6653 行）声明了**每个测试的期望状态**。如果代码改动让一个 `skip-codegen` 测试突然开始生成代码了，CI 告警——这里告警不是因为测试坏了，而是期望文件需要更新以反映新状态。这防止了无声的回归。

---

## 代价和收益

代价是 IR 流水线 + 8 pass 简化 + 169 个 Node codegen 变体 + 3 层测试。**168/209** ONNX 算子行 import 支持（约 80%），上游 `pass` 非 100%——见上表。

复验命令见 [README §源码版本与数字校验](README.md#源码版本与数字校验)。

收益不在于"多少模型能跑"这个百分比。在于能跑的那些模型，**运行路径与手写 Burn 代码完全相同**。

这句话值得拆开：

- **没有运行时**：不依赖 ONNX Runtime 共享库，不需要 protobuf 解析器在启动时读模型文件
- **可以调试**：`forward()` 是普通 Rust 函数，IDE 可以跳进去，调试器可以设断点
- **可以修改**：生成的代码是源码，你可以把某个 `Relu` 换成 `Gelu` 然后重新编译
- **可以嵌入**：`LoadStrategy::Embedded` 把权重编译进二进制，`no_std` 固件也能跑
- **编译期优化**：Burn 的融合引擎和 CubeCL 的 autotune 继续作用于生成的代码——因为它就是普通的 Burn 代码，不是什么黑盒运行时

通过 `burn-onnx` 导入的 Conv 并非模拟 ONNX 操作——生成的代码是 `burn::nn::Conv2d::forward()`，它穿过 [Burn 的类型栈](blog-burn-summary.md) 的 `Autodiff<Fusion<CubeBackend>>`，在运行时走融合流与调度，并在首次遇到具体形状时触发 [CubeCL 的 JIT 与 autotune](blog-cubecl-summary.md)。**从 PyTorch 导出的 ONNX 到 GPU 上的 PTX——整条链路都是 Rust，都可以追踪。**

当 AI 部署从云端 GPU 扩展到浏览器、手机、嵌入式边缘节点时，无运行时依赖、可调试、可嵌入——这些在不同环境中是实际的硬需求。

---

## 系列导航

| 文档 | 主题 | 适合 |
|------|------|------|
| [blog-burn-summary.md](blog-burn-summary.md) | Burn 底层机制地图：类型栈 + 融合流 + ONNX 入口 | 理解 Burn 全栈 |
| **本文** | ONNX→Rust AOT 编译器：6 阶段流水线、注意力融合、分区编译 | 深入 ONNX 导入 |
| [blog-cubecl-summary.md](blog-cubecl-summary.md) | CubeCL 编译器框架地图：`#[cube]`、SSA、autotune、CubeK | 理解 GPU 代码生成 |
| [blog-cubecl-plan.md](blog-cubecl-plan.md) | CubeCL 专题写作计划 + 入门引导 | 跟练 GPU kernel |
| [blog-cubecl-1.md](blog-cubecl-1.md) | CubeCL 专题 1：GELU 走通 launch | 跑第一个 kernel |

---

## 词汇说明表

| 术语 | 简要说明 |
|------|----------|
| **AOT 编译器** | 在 `build.rs` 构建期把 ONNX 译为 Rust，非运行时解释 protobuf。 |
| **coalesce_attention** | 约 1367 行的 simplify pass：SDPA 分解模式 → 单一 `Attention` 节点。 |
| **8 个 simplify pass** | 每轮迭代按固定顺序执行；定点循环最多 10 轮（`MAX_ITERATIONS`）。 |
| **expectations.toml** | 1615 条上游测试的声明式 status；见 `onnx-official-tests/README.md`。 |
| **skip-codegen / skip-compile** | 官方 status 名：codegen 失败 vs 生成 Rust 不编译。 |
| **分区编译** | 大图切为 64–256 节点子模块（`partition.rs`）。 |

*Burn 底层机制系列 · ONNX AOT 编译 · [综合地图](blog-burn-summary.md) · [CubeCL 篇](blog-cubecl-summary.md) · [系列索引](README.md)*
