# 领域与基线速查：给系统工程师的 AI 框架背景

这一页给系统软件 / Rust 后端工程师补两类背景：读懂本项目文章所需的最小 NN/PyTorch 语义（Part A），以及文章里用来对比的几套外部系统（Part B）。每条只讲到"够读下去"，并标注在哪篇文章用到。你已熟悉的条目可跳过。

不需要从这里学会训练模型——目标是让文章里"为什么存在这个机制"的论证对你成立。

---

## Part A · 领域最小集

### 一个训练步是什么

一次训练迭代（training step）做三件事：**前向**——把输入张量经过一串算子（矩阵乘、加偏置、激活函数……）算出输出和 loss；**反向**——从 loss 逆着这串算子算出每个参数的梯度；**更新**——用梯度调整参数。推理（inference）只做前向。本项目的示例 `z = (x*2.0+1.0).tanh(); z.backward()` 就是"前向三个算子 + 反向"的最小骨架。用到：[全景篇](burn/burn-systems-architecture.md)、[Autodiff](burn/autodiff-system-design.md)。

### tensor op 的三类

框架执行的算子大致分三类。**element-wise**（逐元素）：对每个元素独立做同样的标量运算，如 `+`、`*`、`tanh`、`gelu`——输入输出形状相同，计算量小、几乎不复用数据。**matmul**（矩阵乘）：`C = A × B`，计算密集、数据复用高，是 NN 里最耗时的算子。**reduce**（归约）：沿某个维度聚合，如 `sum`、`mean`、`max`、softmax 里的求和。这个分类是后面很多设计的前提：element-wise 适合融合（见下），matmul 适合 autotune 选 tile。用到：[Fusion](burn/kernel-fusion-system-design.md)、[Autotune](cubecl/autotune-system-design.md)、[CubeK](cubek/blueprint-routine-autotune.md)。

### 为什么 element-wise 算子在 NN 中"多而碎"

一个典型网络层往往是"一次 matmul + 若干 element-wise"（加 bias、激活、残差相加、dropout 缩放……）。element-wise 算子单个计算量小，但数量多。每个算子若单独成为一次 GPU kernel 启动，启动开销可能与计算本身相当——这正是 kernel fusion 想消除的浪费。把"NN 里 element-wise 又多又碎"这件事记住，[Fusion](burn/kernel-fusion-system-design.md) 开篇的开销估算就有了来由。

### 反向传播（backprop）算什么

反向传播用链式法则，从 loss 出发，逆着前向算子序列，逐个算出"loss 对每个中间量和参数的偏导（梯度）"。框架需要在前向时记录算子的连接关系（一张图），反向时按相反顺序遍历这张图。训练需要梯度来更新参数；推理不需要，所以"是否构建这张图"是训练与推理的关键区别——也是 Burn 把 Autodiff 做成可在编译期排除的装饰器的动机。用到：[Autodiff](burn/autodiff-system-design.md)、[架构](architecture.md)。

### batch、shape 为什么影响"选哪个 kernel"

同一个算子，输入形状不同，最优实现也不同。matmul `A[1,4096] × B[4096,4096]`（一次 matvec）和 `A[4096,4096] × B[4096,4096]`（一次 gemm）虽是同一段逻辑，但最优的 tile 大小、线程组划分、向量化宽度完全不同；batch size 变化、不同 GPU、是否和别的算子融合，都会改变最优参数。"不存在一套参数对所有形状最优"是 autotune 存在的根本原因。用到：[Autotune](cubecl/autotune-system-design.md)、[CubeK](cubek/blueprint-routine-autotune.md)。

### 激活函数：就是 element-wise

tanh、ReLU、GELU 等激活函数是逐元素的非线性变换，属于 element-wise 算子。练习 [ch1-gelu-variants](../src/ch1-gelu-variants/) 写的就是 GELU kernel——可以把它当作"一个具体的 element-wise 算子在 GPU 上怎么落地"的样本。

---

## Part B · 对比基线速查

文章用"Burn 做 X，别人做 Y"来暴露设计权衡。下面把"别人"各讲一段，让对比能教学。

### PyTorch

最主流的深度学习框架。默认是 **eager（即时执行）**：每个 tensor 操作立刻在设备上执行。autograd 把反向能力嵌进 tensor——每个需要梯度的 tensor 携带一个 `grad_fn` 指针，指向"如何算这一步的反向"，`loss.backward()` 顺着这些指针回溯。算子分发由 **Dispatcher**（按设备 / dtype / 是否 autograd 等多级 key 查表）路由到具体实现。`torch.compile`（后端 **Inductor**）是后加的"把一段 eager 代码 trace 成图再编译融合"的路径。本项目对比点：Burn 把 autograd 做成**外层装饰器**而非 tensor 内置（[Autodiff](burn/autodiff-system-design.md)），把后端选择放在 `Device` 路由而非 Dispatcher（[架构](architecture.md)），融合用惰性队列而非 trace（[Fusion](burn/kernel-fusion-system-design.md)）。

### Triton

OpenAI 的 GPU kernel 编写语言（Python DSL）。你用类 Python 写 kernel，它 JIT 编译到 PTX。它的 autotune 让你声明一组配置（block 大小、stage 数等）组成的**参数网格**，运行时把网格里每个组合都 benchmark 一遍选最快。本项目对比点：[Autotune](cubecl/autotune-system-design.md) 中 CubeCL 用"作者枚举的有限策略 + 优先级剪枝"替代 Triton 的穷举网格，把候选数从上百压到个位/几十。

### XLA

Google 的张量编译器（驱动 JAX、TensorFlow 的一种后端）。它把整个计算图编译成中间表示 **HLO**，在编译期用 **fusion pass**（规则驱动）合并算子，然后 AOT 生成 PTX/Metal 等。本项目对比点：[Fusion](burn/kernel-fusion-system-design.md) 中 Burn 选了"运行时惰性融合 + 探索缓存"的中间路线，而非 XLA 的"编译期静态融合"——代价与适用场景不同。

### CUTLASS

NVIDIA 的开源 C++ 模板库，用模板参数（tile 形状、数据类型、流水线深度……）特化出高性能 matmul kernel。强大但模板参数组合会爆炸，且每个组合是一次独立的编译实例。本项目对比点：[CubeK](cubek/blueprint-routine-autotune.md) 用 Blueprint 纪律把"哪些参数能进编译缓存 key"限制在离散空间，正是为了避免 CUTLASS 式的组合爆炸。

### 一句话：同类问题的其他系统

这些系统反复解决同一组问题，换框架后概念可迁移：**TVM / IREE** 同样做"算子编译 + autotune"；**vLLM / TensorRT-LLM** 在推理侧做 kernel 融合与调度。读完本项目，你应能把 Burn 的机制对应到它们——映射表见 [概念索引 · 可迁移映射](concept-index.md)。

---

← 回到 [README · 学习地图](../README.md#学习地图) · 相关：[架构分析](architecture.md) · [概念索引](concept-index.md)
