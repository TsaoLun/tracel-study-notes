# Tracel 学习笔记

> 深入 [Tracel](https://github.com/tracel-ai) 开源生态的系统设计分析：Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。
>
> 单一顺序的学习路径——从头读到尾，在 `▶ 动手` 停下跑练习，然后继续。

## 阅读前提

需要了解——每个概念后面附了快速学习资源。不深入，能理解"为什么存在这个机制"就够。

**Rust trait 与泛型**：Burn 的类型栈 `Autodiff<Fusion<CubeBackend<WgpuRuntime>>>` 通过 trait 嵌套实现编译期后端组合。需要理解：trait 是什么（接口）、泛型参数如何单态化（编译期为每种具体类型生成独立代码）、`PhantomData` 的作用（标记类型关系但不持有值）。
- [Rust Book §10](https://doc.rust-lang.org/book/ch10-00-generics.html)（泛型+trait）和 [Nomicon: PhantomData](https://doc.rust-lang.org/nomicon/phantom-data.html)

**GPU 执行模型**：Kernel 在 GPU 上以 workgroup（也叫 thread block/CUDA block）为单位并行执行。workgroup 内共享一块快速的 shared memory（对应用户管理的 L1 cache），workgroup 间无法直接通信。寄存器是每个线程最快速的私有存储，全局内存（GPU DRAM）所有 workgroup 共享但访问延迟最高。Fusion 之所以有效，是因为中间结果不再写回全局内存再读出来；Autotune 之所以必要，是因为 tile 大小需要匹配这几层存储的大小和带宽。
- [CUDA Refresher: GPU Computing Ecosystem](https://developer.nvidia.com/blog/tag/cuda-refresher/) 的 Memory Hierarchy 和 Execution Model 章节

**Kernel launch overhead**：CPU 触发 GPU kernel 执行需设置 grid/block 参数→传输参数→驱动调度。5-10 μs 量级，许多 element-wise op 的计算时间与之相当——单独 launch 低效，融合后一次 launch 跑多个 op 才有收益。Fusion 文章开篇的数字直接来自这个开销。
- 验证：`nvprof --print-gpu-trace` 或 CUPTI API

**自动微分（Autodiff）**：反向传播通过链式法则计算梯度。前向时记录每个 op 的输入→输出关系，反向时从输出端逆序传播。需要在概念上理解"前向图"和"反向图"的对应关系。
- [CS231n 反向传播笔记](https://cs231n.github.io/optimization-2/) Introduction + 前两节（计算图 + 链式法则），5-10 分钟

**不需要了解**：CubeCL/wgpu API（文章从零展开）、Rust proc-macro（JIT 文章逐步解释 `#[cube]`）。

## Setup（首次使用）

```bash
# 必须 clone——所有练习依赖
git clone https://github.com/tracel-ai/burn.git
git clone https://github.com/tracel-ai/cubecl.git

# 可选 clone——仅 CubeK 文章需要源码参考，无练习依赖
git clone https://github.com/tracel-ai/cubek.git
git clone https://github.com/tracel-ai/burn-onnx.git
```

> 四个仓库合计约 29GB。可选 clone 的两条可以在读到 CubeK 文章时再决定。

验证 setup：

```bash
cd src && cargo check -p burn-test -p ch1-gelu-variants
```

---

## 阅读路径

### 1. 建立坐标系

**[architecture.md](docs/architecture.md)** — 类型栈、Trait 边界与分层组合。每层解决一个系统问题；层与层通过 trait 交互——上层只知道下层"能做什么"，不知道"怎么做"。

> ✓ 完成标准：能用自己的话解释"为什么 Burn 的 Autodiff 和 Fusion 可以独立演进而不会冲突"。

### 2. 全景概览
**[全景篇](docs/burn/burn-systems-architecture.md)** — 以 `z = (x*2.0+1.0).tanh(); z.backward()` 穿行四个系统。如果初次接触，先浏览 §1–§2（架构图和 Tensor 定义），然后在读完后面各系统文章后回来重读全链路时序图。

> ✓ 完成标准：能在脑子里画出一张图——"一行代码触发后，经过哪几层、每层做了什么"。

### 3. Fusion：为什么需要、怎么排队、如何竞标
**[Fusion](docs/burn/kernel-fusion-system-design.md)** — kernel launch 开销→融合收益，OperationQueue 的 dual IR，惰性执行与触发点。读到 §惰性执行末尾：

> ▶ **动手**：`cd src/burn-test && RUST_LOG=burn_fusion=trace cargo run --release`
> 首次编译需数分钟（burn 全链依赖）。观察 [练习 README](src/burn-test/README.md) 中列出的四条日志特征。

继续读 §OperationFuser 竞标、Block 划分、GPU 内存管理（Page/Slice 三池模型）。

> ✓ 完成标准：看到 `[plan] exploration completed` 或 `[plan] cache hit`；第二次运行时观察到缓存命中。

> 📖 **延伸阅读**：[fusion/1-client-server.md](docs/burn/fusion/1-client-server.md) — from_data 到 GPU buffer 的 client-server 链路源码 walkthrough。

### 4. JIT 编译管线：宏到 GPU 二进制
**[JIT](docs/cubecl/jit-compilation-pipeline.md)** — `#[cube]` 宏展开、IR Scope 树、优化 pass。读到 §IR 优化末尾：

> ▶ **动手**：`cd src/ch1-gelu-variants && cargo test -- --nocapture`
> 写 GELU kernel 的三种变体。三个测试验证不同向量化宽度产生相同计算结果。[练习 README](src/ch1-gelu-variants/README.md) 列出了三个测试的差异。

继续读 §代码生成（WGSL/SPIR-V/MSL）、Pipeline 缓存、GPU dispatch。读到末尾：

> ▶ **动手**：`cd src/ch2-expand-study && cargo test -- --nocapture`
> 现在你写过了 kernel，回来看 Rust `+` 如何变成 `__expand_add_method(scope, rhs)`。[练习 README](src/ch2-expand-study/README.md) 列出每个测试对应的 IR 特征。

> ✓ 完成标准：能解释 `a + b` 在 `#[cube]` 函数中经历了什么——从 Rust 表达式到 IR 操作到 GPU 指令。

> 📖 **延伸阅读**：[cubecl/1-gelu-launch.md](docs/cubecl/1-gelu-launch.md) — GELU 从 `#[cube]` 到 GPU launch 的完整 walkthrough。

### 5. Autotune：选最快的实现
你理解了 kernel 如何编译和启动。Autotune 回答的问题是：**在多个候选 kernel 变体中，选哪个来编译和启动。** 同一个 matmul，1024×4096 和 4096×1024 的最优 tile 大小不同——怎么在首次执行时选出最快者，并缓存结果。

**[Autotune](docs/cubecl/autotune-system-design.md)** — 策略枚举 vs Triton 参数网格、优先级提前终止、anchor 量化缓存。

> ✓ 完成标准：能对比 CubeCL 和 Triton 的 autotune 在"搜索空间定义"和"缓存 key 设计"上的根本差异。

### 6. CubeK：防止 Kernel 爆炸
**[CubeK](docs/cubek/blueprint-routine-autotune.md)** — Blueprint-Routine-Autotuner 三层纪律。JIT 管线的 `KernelId` 哈希决定了编译缓存 key 的维度——CubeK 用 Blueprint 纪律限制哪些参数可以进入这个 key，用 Routine 的离散化防止组合爆炸。

> ✓ 完成标准：能解释"如果把 M 放进 Blueprint，JIT 缓存会怎样爆炸"以及 CUTLASS 的等价问题是什么。

### 7. Autodiff：梯度怎么算
**[Autodiff](docs/burn/autodiff-system-design.md)** — 回顾 Fusion 篇：`Autodiff<Fusion<B>>` 中 Autodiff 在最外层。读到 §图构建结束后：

> ▶ **动手**：`cd src/autodiff-test && cargo test -- --nocapture`
> 验证 `z = tanh(x*2.0+1.0)` 的梯度。[练习 README](src/autodiff-test/README.md) 列出了观察要点和两个自行验证的问题。

继续读 §检查点策略（ComputeBound/MemoryBound）、BFS 逆序执行、分布式梯度同步。

> ✓ 完成标准：能对比 Burn 的装饰器 Autodiff 和 PyTorch 的内置 autograd——在架构位置、推理开销、高阶梯度、检查点粒度上的差异。

### 8. 完成后

- [全景篇](docs/burn/burn-systems-architecture.md) 重读——现在你能理解全链路时序图的每个环节
- [概念索引](docs/concept-index.md) — 按需回查特定主题
- [源码版本管理](docs/SOURCE-VERSION.md) — API 依赖矩阵和已知漂移

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
| 3. Fusion | `src/burn-test` | `RUST_LOG=burn_fusion=trace cargo run --release` |
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
| cubek | `c6a0bf40` | 2026-06-12 |

## 仓库结构

```
docs/                           src/
├── architecture.md             ├── Cargo.toml
├── concept-index.md            ├── burn-test/          (Fusion)
├── SOURCE-VERSION.md           ├── autodiff-test/      (Autodiff)
├── burn/                       ├── ch1-gelu-variants/  (JIT)
│   ├── burn-systems-architecture.md ├── ch2-expand-study/    (JIT)
│   ├── kernel-fusion-system-design.md ├── ch3-trait-study/
│   ├── autodiff-system-design.md  ├── fusion-ch2-queue/
│   ├── summary.md                  └── fusion-ch3-drain/
│   └── fusion/ (1-client-server)
├── cubecl/                      burn/       (gitignored)
│   ├── autotune-system-design.md cubecl/     (gitignored)
│   ├── jit-compilation-pipeline.md cubek/      (gitignored)
│   ├── summary.md               burn-onnx/  (gitignored)
│   ├── 1-gelu-launch.md
│   └── 2-expand.md
├── cubek/
│   ├── blueprint-routine-autotune.md
│   └── summary.md
└── appendix/
```

[CLAUDE.md](CLAUDE.md) · 文档以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可发布。
