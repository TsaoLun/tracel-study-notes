# Tracel 学习笔记

> 深入 [Tracel](https://github.com/tracel-ai) 开源生态的系统设计分析：Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。
>
> 单一顺序的学习路径——从头读到尾，在 `▶ 动手` 停下跑练习，然后继续。

## 适合谁读

面向系统软件 / Rust 后端工程师：对神经网络只有初步概念、用过基础 PyTorch，想借 Burn 这套可读的 Rust 代码库学习通用的 AI infra/sys。

这里的机制——惰性队列、状态机、缓存、内存池、JIT 编译管线、IR——大多是你熟悉的系统概念，落在 ML 框架的场景里。文章用 Burn 作载体，但讲的是**任何深度学习框架都要解决的问题**：算子融合、自动调参、kernel 编译、自动微分。读完能把这些机制对应到 PyTorch / XLA / Triton / TVM 等系统（见 [概念索引 · 可迁移映射](docs/concept-index.md)）。

缺口通常不在系统侧，而在领域侧：什么是 element-wise 算子、backprop 算什么、为什么 batch/shape 影响选 kernel、PyTorch/Triton/XLA/CUTLASS 各自怎么做。这些在一页 [primer](docs/primer.md) 里各用一段话讲清——先过一遍它，再进主线。

**不需要预先掌握**：CubeCL/wgpu API（文章从零展开）、Rust proc-macro（JIT 文章逐步解释 `#[cube]`）、训练模型的实操经验。

## 学习地图

按顺序读，每个阶段附难度、预计时长和产出。难度是针对系统/Rust 背景读者标的——系统原生主题（Fusion 队列、JIT 管线）对你偏易，ML 重的环节（Autodiff、全景篇的 `backward()`）才是新摩擦点。时长是单次阅读估计，不含练习首次编译（burn 全链依赖首次编译需数分钟）。

| 阶段 | 主题 | 难度 | 预计时长 | 读完能回答 |
|------|------|------|----------|------------|
| 0 | [领域与基线速查 primer](docs/primer.md) + [前置自检](#第-0-步前置自检) | 必读 | ~20–40 分钟 | NN 算子三类是什么、backprop 算什么、PyTorch/Triton/XLA/CUTLASS 各做什么 |
| 1 | [架构坐标系](docs/architecture.md) | 入门 | ~20 分钟 | 为什么 Autodiff 和 Fusion 能独立演进 |
| 2 | [全景篇](docs/burn/burn-systems-architecture.md) | 中等（初次先浏览） | ~30 分钟浏览 | 一行代码触发后经过哪几层、每层做什么 |
| 3 | [Fusion](docs/burn/kernel-fusion-system-design.md) | 偏易（系统原生） | ~50 分钟 + 练习 | 为什么三个 op 要融成一个 kernel、怎么排队触发 |
| 4 | [JIT 编译管线](docs/cubecl/jit-compilation-pipeline.md) | 中等（系统原生） | ~90 分钟 + 两个练习 | `a + b` 从 Rust 表达式到 GPU 指令经历了什么 |
| 5 | [Autotune](docs/cubecl/autotune-system-design.md) | 中等 | ~40 分钟 | CubeCL 与 Triton 在搜索空间和缓存 key 上的差异 |
| 6 | [CubeK](docs/cubek/blueprint-routine-autotune.md) | 难 | ~40 分钟 | Blueprint 纪律如何防止 JIT 缓存爆炸 |
| 7 | [Autodiff](docs/burn/autodiff-system-design.md) | 中等偏难（ML 重） | ~60 分钟 + 练习 | 装饰器 Autodiff 与 PyTorch autograd 的架构差异 |
| 8 | [回顾与索引](#8-完成后) | — | 按需 | 回查特定概念、把机制迁移到其他框架 |

每个阶段末尾有「✓ 完成标准」，能用自己的话回答再进入下一阶段。详细路径见下面「阅读路径」。

## 第 0 步：前置自检

先读 [领域与基线速查 primer](docs/primer.md)（NN 最小语义 + PyTorch/Triton/XLA/CUTLASS 各一段）。然后用下面几项自检——能答上来就跳过，答不上来回 primer 或按链接补。不深入，能理解"为什么存在这个机制"就够。

- **NN/PyTorch 最小语义**（首要）。判断标准：能说出 element-wise / matmul / reduce 三类算子的区别、为什么 element-wise 在 NN 里又多又碎、`loss.backward()` 大致算什么。这是读这些文章的领域底座，缺则文章的"为什么"论证落不了地。补：[primer · Part A](docs/primer.md#part-a--领域最小集)。

- **对比基线**（首要）。判断标准：能说出 PyTorch eager + `grad_fn`、Triton autotune 网格、XLA HLO 融合、CUTLASS 模板各是什么——文章用它们做对比来暴露 Burn 的设计权衡。补：[primer · Part B](docs/primer.md#part-b--对比基线速查)。

- **Rust trait 与泛型**（你大概率已具备）。作为 Rust 工程师这里基本无门槛。唯一要熟悉的 Burn 专有模式：默认 `Device::wgpu(...)` 展开为 `Fusion<CubeBackend<WgpuRuntime<...>>>`，`.autodiff()` 后在 dispatch 层外包 `Autodiff<...>`，在编译期单态化；用户侧 `Tensor` 不带 Backend 泛型，经 `BridgeTensor` 按 `Device` 路由。[架构](docs/architecture.md) 会展开。

- **GPU 执行模型**。判断标准：能说出 shared memory、寄存器、全局内存三者的速度和共享范围差异。Kernel 以 workgroup（thread block / CUDA block）为单位并行；workgroup 内共享快速 shared memory，workgroup 间不直接通信；寄存器是每线程最快的私有存储，全局内存（GPU DRAM）所有 workgroup 共享但延迟最高。Fusion 有效是因为中间结果不再写回全局内存再读出；Autotune 必要是因为 tile 大小要匹配这几层存储。补：[CUDA Refresher](https://developer.nvidia.com/blog/tag/cuda-refresher/) 的 Memory Hierarchy 和 Execution Model 章节。

- **Kernel launch overhead**（Fusion 篇会用到的数字）。CPU 触发 GPU kernel 需设置 grid/block 参数、传输参数、驱动调度，约 5–10 μs 量级，许多 element-wise op 的计算时间与之相当。验证：`nvprof --print-gpu-trace` 或 CUPTI API。

## Setup（首次使用）

```bash
# 必须 clone——所有练习依赖
git clone https://github.com/tracel-ai/burn.git
git clone https://github.com/tracel-ai/cubecl.git

# 可选 clone——仅 CubeK 文章需要源码参考，无练习依赖
git clone https://github.com/tracel-ai/cubek.git
git clone https://github.com/tracel-ai/burn-onnx.git
```

> 四个仓库合计约 29GB。`cubek`、`burn-onnx` 两条可以读到对应文章时再 clone。练习只依赖 `burn` 和 `cubecl`。

验证 setup：

```bash
cd src && cargo check -p burn-test -p ch1-gelu-variants
```

## 最快上手（约 30 分钟）

想先跑通一个例子建立直觉、再读理论，按这条最小路径：

1. clone `burn` 和 `cubecl` 到项目根目录（见上）。
2. 跑融合示例，观察四个操作融合成一个 kernel：

```bash
cd src/burn-test && BURN_FUSION_LOG=full cargo run --release
```

> 首次编译需数分钟（burn 全链依赖）。预期日志特征见 [burn-test/README.md](src/burn-test/README.md)。

3. 看到 `[plan] exploration completed` 或 `[plan] cache hit` 后，回到上面的「学习地图」从阶段 1 开始顺序阅读。

---

## 阅读路径

### 1. 建立坐标系

**[architecture.md](docs/architecture.md)** — 类型栈、Trait 边界与分层组合。每层解决一个系统问题；层与层通过 trait 交互——上层只知道下层"能做什么"，不知道"怎么做"。`Tensor` 经 `Device` 路由，Backend 组合在框架内部展开。

> ✓ 完成标准：能用自己的话解释"为什么 Burn 的 Autodiff 和 Fusion 可以独立演进而不会冲突"。

### 2. 全景概览
**[全景篇](docs/burn/burn-systems-architecture.md)** — 以 `z = (x*2.0+1.0).tanh(); z.backward()` 穿行四个系统（device 须 `.autodiff()`）。如果初次接触，先浏览 §1–§2（架构图和 Tensor 定义），跑一遍「最快上手」的 burn-test，然后在读完后面各系统文章后回来重读全链路时序图。

> ✓ 完成标准：能在脑子里画出一张图——"一行代码触发后，经过哪几层、每层做了什么"。

### 3. Fusion：为什么需要、怎么排队、如何竞标
**[Fusion](docs/burn/kernel-fusion-system-design.md)** — kernel launch 开销→融合收益，OperationQueue 的 dual IR，惰性执行与触发点。读到 §惰性执行末尾：

> ▶ **动手**：`cd src/burn-test && BURN_FUSION_LOG=full cargo run --release`
> 首次编译需数分钟（burn 全链依赖）。观察 [练习 README](src/burn-test/README.md) 中列出的四条日志特征。

继续读 §OperationFuser 竞标、Block 划分、GPU 内存管理（Page/Slice 三池模型）。

> ✓ 完成标准：看到 `[plan] exploration completed` 或 `[plan] cache hit`；第二次运行时观察到缓存命中。

> 📖 **延伸阅读**：[fusion/1-client-server.md](docs/burn/fusion/1-client-server.md) — from_data 到 GPU buffer 的 client-server 链路源码 walkthrough。

### 4. JIT 编译管线：宏到 GPU 二进制
**[JIT](docs/cubecl/jit-compilation-pipeline.md)** — `#[cube]` 宏展开、IR Scope 树、优化 pass。读到 §IR 优化末尾：

> ▶ **动手**：`cd src/ch1-gelu-variants && cargo test -- --nocapture`
> 写 GELU kernel 的向量化变体。测试验证不同向量化宽度产生相同计算结果。[练习 README](src/ch1-gelu-variants/README.md) 列出了每个测试观察的内容。

继续读 §代码生成（WGSL/SPIR-V/MSL）、Pipeline 缓存、GPU dispatch。读到末尾：

> ▶ **动手**：`cd src/ch2-expand-study && cargo test -- --nocapture`
> 现在你写过了 kernel，回来看 Rust `+` 如何变成 `__expand_add_method(scope, rhs)`。[练习 README](src/ch2-expand-study/README.md) 列出每个测试对应的 IR 特征。

> ✓ 完成标准：能解释 `a + b` 在 `#[cube]` 函数中经历了什么——从 Rust 表达式到 IR 操作到 GPU 指令。

> 📖 **延伸阅读**：[cubecl/1-gelu-launch.md](docs/cubecl/1-gelu-launch.md) — GELU 从 `#[cube]` 到 GPU launch 的完整 walkthrough；[cubecl/2-expand.md](docs/cubecl/2-expand.md) — `#[cube]` 宏展开内部机制。

### 5. Autotune：选最快的实现
你理解了 kernel 如何编译和启动。Autotune 回答的问题是：**在多个候选 kernel 变体中，选哪个来编译和启动。** 同一个 matmul，1024×4096 和 4096×1024 的最优 tile 大小不同——怎么在首次执行时选出最快者，并缓存结果。

**[Autotune](docs/cubecl/autotune-system-design.md)** — 策略枚举 vs Triton 参数网格、优先级提前终止、anchor 量化缓存。

> ✓ 完成标准：能对比 CubeCL 和 Triton 的 autotune 在"搜索空间定义"和"缓存 key 设计"上的根本差异。

### 6. CubeK：防止 Kernel 爆炸
**[CubeK](docs/cubek/blueprint-routine-autotune.md)** — Blueprint-Routine-Autotuner 三层纪律。JIT 管线的 `KernelId` 哈希决定了编译缓存 key 的维度——CubeK 用 Blueprint 纪律限制哪些参数可以进入这个 key，用 Routine 的离散化防止组合爆炸。

> ✓ 完成标准：能解释"如果把 M 放进 Blueprint，JIT 缓存会怎样爆炸"以及 CUTLASS 的等价问题是什么。

### 7. Autodiff：梯度怎么算
**[Autodiff](docs/burn/autodiff-system-design.md)** — 回顾 Fusion 篇：默认 `Device::wgpu(...)` 不含 Autodiff，`.autodiff()` 后 `Autodiff<Fusion<B>>` 中 Autodiff 在最外层。读到 §图构建结束后：

> ▶ **动手**：`cd src/autodiff-test && cargo test -- --nocapture`
> 验证 `z = tanh(x*2.0+1.0)` 的梯度。[练习 README](src/autodiff-test/README.md) 列出了观察要点和两个自行验证的问题。

继续读 §检查点策略（ComputeBound/MemoryBound）、BFS 逆序执行、分布式梯度同步。

> ✓ 完成标准：能对比 Burn 的装饰器 Autodiff 和 PyTorch 的内置 autograd——在架构位置、推理开销、高阶梯度、检查点粒度上的差异。

### 8. 完成后

- [全景篇](docs/burn/burn-systems-architecture.md) 重读——现在你能理解全链路时序图的每个环节
- [概念索引](docs/concept-index.md) — 按需回查特定主题
- [源码版本管理](docs/SOURCE-VERSION.md) — API 依赖矩阵和已知漂移
- [写作计划与进度（ROADMAP）](docs/ROADMAP.md) — 已完成内容与计划中的章节教程

---

## 可选延伸

| 延伸阅读 | 说明 |
|----------|------|
| [fusion/1-client-server.md](docs/burn/fusion/1-client-server.md) | Fusion client-server 源码 walkthrough |
| [cubecl/1-gelu-launch.md](docs/cubecl/1-gelu-launch.md) | GELU kernel 完整生命周期 walkthrough |
| [cubecl/2-expand.md](docs/cubecl/2-expand.md) | `#[cube]` 宏展开内部机制 |
| [automatic-kernel-fusion.md](docs/appendix/automatic-kernel-fusion.md) | 旧博客中文翻译（项目起源——这篇文章催生了整个分析和重构） |

## 练习速查

| 步骤 | 练习 | 命令 |
|------|------|------|
| 3. Fusion | `src/burn-test` | `BURN_FUSION_LOG=full cargo run --release` |
| 4. JIT（先） | `src/ch1-gelu-variants` | `cargo test -- --nocapture` |
| 4. JIT（后） | `src/ch2-expand-study` | `cargo test -- --nocapture` |
| 7. Autodiff | `src/autodiff-test` | `cargo test -- --nocapture` |

---

## 源码版本

| 仓库 | commit | 日期 |
|------|--------|------|
| burn | `78f10aec1` | 2026-06-10 |
| cubecl | `35b861d0` | 2026-06-12 |
| burn-onnx | `846b2452` | 2026-06-11 |
| cubek | `4ccfc4f2` | 2026-06-16 |

详细的 API 依赖矩阵和已知漂移见 [docs/SOURCE-VERSION.md](docs/SOURCE-VERSION.md)。

## 仓库结构

```
docs/                           src/
├── primer.md                   ├── Cargo.toml
├── architecture.md             ├── burn-test/          (Fusion)
├── concept-index.md            ├── autodiff-test/      (Autodiff)
├── SOURCE-VERSION.md           ├── ch1-gelu-variants/  (JIT)
├── ROADMAP.md                  ├── ch2-expand-study/   (JIT)
├── burn/                       └── ...（计划中骨架见 ROADMAP）
│   ├── burn-systems-architecture.md
│   ├── kernel-fusion-system-design.md
│   ├── autodiff-system-design.md  burn/       (gitignored)
│   ├── summary.md              cubecl/     (gitignored)
│   └── fusion/ (1-client-server)  cubek/      (gitignored)
├── cubecl/                      burn-onnx/  (gitignored)
│   ├── autotune-system-design.md
│   ├── jit-compilation-pipeline.md
│   ├── summary.md
│   ├── 1-gelu-launch.md
│   └── 2-expand.md
├── cubek/
│   ├── blueprint-routine-autotune.md
│   └── summary.md
└── appendix/
```

[CLAUDE.md](CLAUDE.md) · 文档以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可发布。
