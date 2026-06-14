# Tracel 学习笔记

> 深入 [Tracel](https://github.com/tracel-ai) 开源生态的系统设计分析：Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。
>
> 单一顺序的学习路径——从头读到尾，在 `▶ 动手` 停下跑练习，然后继续。

## 阅读路径

### 1. 建立坐标系

**[architecture.md](docs/architecture.md)**（15 分钟）— 四项目共享的设计哲学：决策推迟（L1 编译期 → L2 JIT 时 → L3 首次执行）。读完知道 Tracel 生态的组件为什么可以自由组合。

### 2. 全景概览

**[全景篇](docs/burn/burn-systems-architecture.md)** — 以 `z = (x*2.0+1.0).tanh(); z.backward()` 穿行四个系统。如果初次接触，可以先浏览 §1–§2（架构 + Tensor 定义），然后在读完后面各系统文章后回来重读全链路时序图。

### 3. Fusion：为什么需要、怎么排队、如何竞标

**[Fusion](docs/burn/kernel-fusion-system-design.md)** — kernel launch 开销→融合收益，OperationQueue 的 dual IR，惰性执行与触发点。读到 §惰性执行末尾：

> ▶ **动手**：`cd src/burn-test && RUST_LOG=burn_fusion=trace cargo run --release`
> 观察 `[stream]`、`[plan]`、`[explorer]` 日志行。

继续读 §OperationFuser 竞标、Block 划分、GPU 内存管理（Page/Slice 三池模型）。

> 📖 **延伸阅读**：[fusion/1-client-server.md](docs/burn/fusion/1-client-server.md) — from_data 到 GPU buffer 的 client-server 链路源码 walkthrough。与本篇的设计分析互补，可按需选读。

### 4. JIT 编译管线：宏到 GPU 二进制

**[JIT](docs/cubecl/jit-compilation-pipeline.md)** — `#[cube]` 宏展开、IR Scope 树、优化 pass。读到 §IR 优化末尾：

> ▶ **动手**：`cd src/ch1-gelu-variants && cargo test -- --nocapture`
> 写 GELU kernel 的三种变体。先建立"我能写一个 kernel"的直觉，再看编译器内部。

继续读 §代码生成（WGSL/SPIR-V/MSL）、Pipeline 缓存、GPU dispatch。读到末尾：

> ▶ **动手**：`cd src/ch2-expand-study && cargo test -- --nocapture`
> 现在你写过了 kernel，回来看 Rust `+` 如何变成 `__expand_add_method(scope, rhs)`。

### 5. Autotune：选最快的实现

你理解了 kernel 如何编译和启动。Autotune 回答的问题是：**在多个候选 kernel 变体中，选哪个来编译和启动。** 同一个 matmul，1024×4096 和 4096×1024 的最优 tile 大小不同——怎么在首次执行时选出最快者，并缓存结果。

**[Autotune](docs/cubecl/autotune-system-design.md)** — 策略枚举 vs Triton 参数网格、优先级提前终止、anchor 量化缓存。全文概念密集。

> 📖 **延伸阅读**：[cubecl/1-gelu-launch.md](docs/cubecl/1-gelu-launch.md) — GELU 从 `#[cube]` 到 GPU launch 的完整 walkthrough。与本篇的 JIT 管线互补。

### 6. CubeK：防止 Kernel 爆炸

**[CubeK](docs/cubek/blueprint-routine-autotune.md)** — Blueprint-Routine-Autotuner 三层纪律。JIT 管线的 `KernelId` 哈希决定了编译缓存 key 的维度——CubeK 用 Blueprint 纪律限制哪些参数可以进入这个 key，用 Routine 的离散化防止组合爆炸。

### 7. Autodiff：梯度怎么算

**[Autodiff](docs/burn/autodiff-system-design.md)** — 回顾 Fusion 篇：`Autodiff<Fusion<B>>` 中 Autodiff 在最外层，前向操作先经 autodiff 记录梯度图，再入 fusion 排队。读到 §图构建结束后：

> ▶ **动手**：`cd src/autodiff-test && cargo test -- --nocapture`
> 验证 `z = tanh(x*2.0+1.0)` 的梯度，观察 `Gradients` 容器的消费。

继续读 §检查点策略（ComputeBound/MemoryBound）、BFS 逆序执行、分布式梯度同步。

### 8. 完成后

- 用 [概念索引](docs/concept-index.md) 按需回查特定主题
- [全景篇](docs/burn/burn-systems-architecture.md) 重读——现在你能理解全链路时序图的每个环节
- [源码版本管理](docs/SOURCE-VERSION.md) 记录 API 依赖和漂移状态

## 可选延伸

| 延伸阅读 | 说明 |
|----------|------|
| [fusion/1-client-server.md](docs/burn/fusion/1-client-server.md) | Fusion client-server 源码 walkthrough |
| [cubecl/1-gelu-launch.md](docs/cubecl/1-gelu-launch.md) | GELU kernel 完整生命周期 walkthrough |
| [cubecl/2-expand.md](docs/cubecl/2-expand.md) | `#[cube]` 宏展开内部机制 |

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

设置参考仓库：

```bash
git clone https://github.com/tracel-ai/burn.git
git clone https://github.com/tracel-ai/burn-onnx.git
git clone https://github.com/tracel-ai/cubecl.git
git clone https://github.com/tracel-ai/cubek.git
```

---

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
