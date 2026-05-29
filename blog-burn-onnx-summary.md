# 不是加载模型，是编译模型：Burn ONNX 的构建时代码生成

> ONNX Runtime 在运行时解释图。Burn ONNX 在编译期把图翻译为可调试的 Rust 源码。区别跟解释器和 AOT 编译器一样。

---

**同一张图，两种命运。**

你在 PyTorch 里训练了一个 Transformer，导出为 ONNX。用 ONNX Runtime 加载它：运行时解析 protobuf，构建内部图表示，为每个节点查表找到 kernel 实现，执行。你的二进制旁边必须带着 `libonnxruntime.so`（30MB+）。

用 Burn ONNX 加载它：在 `build.rs` 里调用 `ModelGen::new().input("model.onnx").out_dir("model/").run_from_script()`——**编译结束后**，`model.onnx` 不存在了。取而代之的是 `model.rs`（纯 Rust 源码，包含 struct 定义和 `forward()` 方法）和 `model.bpk`（权重二进制）。你在代码里写下：

```rust
let model: Model = Model::from_file("model.bpk", &device);
let output = model.forward(input);
```

没有图解释器。没有 protobuf。没有运行时查表。`model.forward()` 是你可以在调试器里逐行跟进去的 Rust 函数。你可以设断点看中间张量的值。你可以手动修改生成的代码——比如把某个 `Relu` 换成 `Gelu`，重新编译。

**这就是"AOT 编译 ONNX 模型"的含义。** 它不是"让 Rust 能加载 ONNX"——那是 ONNX Runtime 绑定的做法。它是"让 ONNX 模型变成 Rust"。

代价是，你需要一个真正的编译器。这篇文章解释这个编译器是怎么工作的。

> **与其他文档的关系**：Burn 的类型栈（`Autodiff<Fusion<…>>`）与编译期后端组合见 [blog-burn-summary.md](blog-burn-summary.md)；运行时的融合流调度与 8.2× 框架开销见同文档第五节；CubeCL 的 JIT 与 GPU 代码生成见 [blog-cubecl-summary.md](blog-cubecl-summary.md)。生成出的 `model.rs` 是普通 Burn 代码——它会穿过 `Autodiff<Fusion<…>>` 栈，走融合流，并触发 CubeCL 的 JIT 与 autotune。

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

对于 Burn ONNX，这**是一个机会**。因为输出的是 Rust 源码而非运行时图，编译器可以在生成代码之前**识别这个模式，把它融合回一个 `Attention` 节点**——直接调用 Burn 的原生注意力实现。Flash Attention（完成后）或高效融合内核，一次 kernel launch，中间结果留在寄存器或共享内存里。

`crates/onnx-ir/src/simplify/coalesce_attention.rs`（1368 行）专门做这一件事。算法：

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

这 1368 行的核心价值不是"做了一件事"——是**在编译期做了一件运行时 loader 做不到的事**。ONNX Runtime 不知道"这 5 个操作是一个注意力"——它只看到 5 个独立的操作。Burn ONNX 看到了注意力，因为它是一个编译器。它有时**间**停下来做模式匹配（`build_producer_map`、`build_consumer_map`、`is_single_use` 守卫），因为这一切发生在编译期。

注意力融合让 27 个 `attention_*_expanded` 上游测试从失败变为通过。不是修了 27 个 bug——是一个优化让 27 个测试的生成代码质量飞跃了。

---

## 支撑这场胜利的流水线

注意力融合只是 8 个简化 pass 中的 1 个。它们运行在 **6 阶段流水线**的第 4 阶段（`onnx-ir/src/pipeline.rs` 驱动）。总览：

| 阶段 | 模块 | 做什么 |
|:---:|------|--------|
| 1 | `initialization` | Protobuf → `RawNode`，常量初始化，mmap 外部权重 |
| 2 | `node_conversion` | 类型化节点 + `Gemm→Linear` 等早期合并 |
| 2b | pipeline 内联循环 | RNN 权重的 Slice→Concat 链折叠 |
| 3 | `type_inference` | 迭代形状/类型推断，`ScalarTensor` 区分 |
| 4 | `post_processing` + `simplify` | Identity 消除 + **8 轮定点简化**（含注意力融合） |
| 5 | `finalization` + codegen | `Node` 枚举 → 220+ 算子生成 `model.rs` |

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

消除 Identity 节点（`x = Identity(y)`→ 直接替换）。然后运行 8 轮定点简化——除了注意力融合，还有：

- **Permute-reshape**：`Shape→Gather→Unsqueeze→Concat→Reshape`→`Transpose`（ONNX 中的维度重排惯用语）
- **Constant shape**：`Shape(x)→Gather(i)` 静态可解时折叠。**但裸 `Shape(x)` 被故意保留**——模型可能用静态维度导出，运行时接受动态输入
- **Constant folding**：全常量输入节点在编译期求值
- **Idempotent / Identity element elimination**：`Relu(Relu(x))`→`Relu(x)`，`x+0`→`x`
- **CSE + Dead code elimination**：合并重复节点，级联清除死节点

### 你需要 Phase 4 是循环，因为 pass 之间有级联

CSE 合并 → 产生死节点 → DCE 清除 → 释放了被死路径独占的常量 → 下一轮 constant folding 折叠那个常量 → 可能暴露新的可 Idempotent 消除。3-5 轮收敛。

### 你需要 Phase 5，因为简化后的图需要变成代码

清理未引用常量 → `RawNode` 转为 `Node` 枚举 → `impl_node_codegen_dispatch!` 宏对 220+ 种算子变体生成 match dispatch。每个算子的 `forward()` 方法生成对应的 Rust 代码。

---

## 当图大到编译器吃不下

SDXL 的 UNet 有上万个节点。全部塞进一个 `forward()` 方法 = Rust 编译时间无法接受，甚至源码大到解析器拒绝处理。

`crates/burn-onnx/src/burn/partition.rs` 的算法做三件事：

1. **O(n) 扫描每个切分点的成本**：在节点 i 和 i+1 之间切一刀，有多少张量跨越边界。用前缀和差分数组（`delta[producer+1] += 1, delta[consumer] -= 1`）。
2. **贪心选切点**：将图切分为 64-256 节点的子模块，每个窗口内选切分成本最低的位置。
3. **常量重排**：把常量节点移到它们首个消费者的前面——避免一个权重在开头定义、在 500 个节点后才用，导致被迫跨越多个分区。

SDXL 和 Depth-Pro 的导入因为这 438 行代码才可能。

---

## 怎么保证生成的代码是正确的？

三层测试，从锁死输出到验证数值：

1. **790 个快照测试**：每个 `NodeCodegen` 实现都带着 `insta::assert_snapshot!`——修改代码生成逻辑后，`cargo insta review` 逐行审查生成的 Rust 代码差异
2. **178 个集成测试**：每个算子用 Python `onnx.reference.ReferenceEvaluator` 生成 ground truth，生成的 Rust 编译执行后逐元素对比
3. **1615 个上游测试**：ONNX v1.19.0 后端测试套件——717 pass / 179 数值偏差 / 215 编译失败 / 504 代码生成失败。80% 的编译集通过数值比较

测试门的一个特别设计：`expectations.toml`（6653 行）声明了**每个测试的期望状态**。如果代码改动让一个 `skip-codegen` 测试突然开始生成代码了，CI 告警——不是因为它坏了，是因为期望需要更新。这防止了无声的回归。

---

## 代价和收益

代价是 6 阶段流水线 + 8 轮简化 + 220 个算子代码生成 + 3 层测试。160/201 个算子支持，当前 80% 编译通过率——不是 100%。

收益不是 "80% 的模型能跑"。收益是"能跑的那些模型，**跑的方式和手写的 Rust 代码没有区别**。"

这句话值得拆开：

- **没有运行时**：不依赖 ONNX Runtime 共享库，不需要 protobuf 解析器在启动时读模型文件
- **可以调试**：`forward()` 是普通 Rust 函数，IDE 可以跳进去，调试器可以设断点
- **可以修改**：生成的代码是源码，你可以把某个 `Relu` 换成 `Gelu` 然后重新编译
- **可以嵌入**：`LoadStrategy::Embedded` 把权重编译进二进制，`no_std` 固件也能跑
- **编译期优化**：Burn 的融合引擎和 CubeCL 的 autotune 继续作用于生成的代码——因为它就是普通的 Burn 代码，不是什么黑盒运行时

一个通过 `burn-onnx` 导入的 Conv 不是"ONNX Conv 被模拟了"——生成的代码是 `burn::nn::Conv2d::forward()`，它穿过 [Burn 的类型栈](blog-burn-summary.md) 的 `Autodiff<Fusion<CubeBackend>>`，在运行时走融合流与调度，并在首次遇到具体形状时触发 [CubeCL 的 JIT 与 autotune](blog-cubecl-summary.md)。**从 PyTorch 导出的 ONNX 到 GPU 上的 PTX——整条链路都是 Rust，都可以追踪。**

在一个 AI 部署从"云端 GPU"扩展到"浏览器、手机、嵌入式边缘节点"的时代，这些不是 nice-to-have。它们是你能不能进入某些环境的门票。

---

## 系列导航

| 文档 | 主题 | 适合 |
|------|------|------|
| [blog-burn-summary.md](blog-burn-summary.md) | Burn 底层机制地图：类型栈 + 融合流 + ONNX 入口 | 理解 Burn 全栈 |
| **本文** | ONNX→Rust AOT 编译器：6 阶段流水线、注意力融合、分区编译 | 深入 ONNX 导入 |
| [blog-cubecl-summary.md](blog-cubecl-summary.md) | CubeCL 编译器框架地图：`#[cube]`、SSA、autotune、CubeK | 理解 GPU 代码生成 |
| [blog-cubecl-plan.md](blog-cubecl-plan.md) | CubeCL 专题写作计划 + 入门引导 | 跟练 GPU kernel |
| [blog-cubecl-1.md](blog-cubecl-1.md) | CubeCL 专题 1：GELU 走通 launch | 跑第一个 kernel |

*Burn 底层机制系列 · ONNX AOT 编译 · [综合地图](blog-burn-summary.md) · [CubeCL 篇](blog-cubecl-summary.md)*
