# Tracel 学习笔记

> 深入 [Tracel](https://github.com/tracel-ai) 开源生态的系统设计分析：Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。
>
> 单一顺序的学习路径：从头读到尾，在需要动手时停下，跑练习，然后继续。

## 阅读路径

以一次 GPU training step 的全链路为线索。每篇文章的特定段落末尾有 `▶ 动手` 提示——停在那里，去跑对应的练习。

### 1. 建立坐标系

**[architecture.md](docs/architecture.md)** — 四项目共享的设计哲学：决策推迟（编译期 L1 → JIT 时 L2 → 首次执行 L3）。读完你知道 Tracel 生态的组件为什么可以自由组合。

### 2. 建立全貌

**[全景篇](docs/burn/burn-systems-architecture.md)** — 以 `z = (x*2.0+1.0).tanh(); z.backward()` 穿行四个系统。

### 3. Fusion：为什么需要、怎么排队

**[Fusion §为什么需要](docs/burn/kernel-fusion-system-design.md#为什么需要-kernel-fusion)** 到 **[Fusion §惰性执行](docs/burn/kernel-fusion-system-design.md#惰性执行操作如何排队)** — kernel launch 开销 vs 融合收益的数字，OperationQueue 的 dual IR。

> ▶ **动手**：`cd src/burn-test && RUST_LOG=burn_fusion=trace cargo run --release`
> 观察 `[stream]`、`[plan]`、`[explorer]` 日志行——看到操作入队、Policy 决策和探索过程。

**[Fusion §竞标 + §Block 划分](docs/burn/kernel-fusion-system-design.md#融合引擎operationfuser-的竞争探索)** — OperationFuser 如何竞标，Block 如何按 tensor 依赖划分。

> ▶ **跟练**：[fusion/1-client-server.md](docs/burn/fusion/1-client-server.md) — 理解 `from_data` 到 GPU buffer 的完整 client-server 链路。

**[Fusion §GPU 内存管理](docs/burn/kernel-fusion-system-design.md#gpu-内存管理page--slice-模型)** — Page/Slice 模型、三池分离、`ALLOC_AFTER_FREE` 延迟驱逐。

### 4. JIT 编译管线：宏到 GPU 二进制

**[JIT §宏展开 + §IR 设计](docs/cubecl/jit-compilation-pipeline.md#第一步cube-过程宏--从-rust-到-ir)** 到 **[JIT §IR 优化](docs/cubecl/jit-compilation-pipeline.md#第三步ir-优化--多次-pass-循环收敛)** — `#[cube]` 如何变成 Scope 树，`ConstOperandSimplify` 等优化 pass。

> ▶ **动手**：`cd src/ch2-expand-study && cargo test -- --nocapture`
> 观察 Rust `+` 如何变成 `__expand_add_method(scope, rhs)`。

**[JIT §代码生成 + §Launch](docs/cubecl/jit-compilation-pipeline.md#第四步多平台代码生成)** 到末尾 — WGSL/SPIR-V/MSL 三后端，Pipeline 缓存，GPU dispatch。

> ▶ **动手**：`cd src/ch1-gelu-variants && cargo test -- --nocapture`
> 写 GELU kernel 的三种变体，理解 `#[cube]` 函数的完整生命周期。

### 5. Autotune：选最快的实现

**[Autotune 全文](docs/cubecl/autotune-system-design.md)** — 策略枚举 vs Triton 参数网格、优先级提前终止、anchor 量化缓存、与 Fusion 的交互。全文无练习——概念密集，建议一口气读完。

### 6. CubeK：防止 Kernel 爆炸

**[CubeK 全文](docs/cubek/blueprint-routine-autotune.md)** — Blueprint-Routine-Autotuner 三层纪律、与 CUTLASS 的对比。无练习。

### 7. Autodiff：梯度怎么算

**[Autodiff §装饰器 + §图构建](docs/burn/autodiff-system-design.md#autodiff-在框架中的位置)** 到 **[Autodiff §图构建末尾](docs/burn/autodiff-system-design.md#图构建tape-based-与-walkthrough)** — `Autodiff<B, C>` 装饰器、`Backward` trait 注册、前向 trace。

> ▶ **动手**：`cd src/autodiff-test && cargo test -- --nocapture`
> 观察 `z.backward()` 后的梯度计算，和 Gradients 容器的消费过程。

**[Autodiff §检查点 + §反向 + §分布式](docs/burn/autodiff-system-design.md#检查点策略计算密集-vs-内存密集)** 到末尾 — `ComputingProperty` 分类、BFS 逆序执行、`on_register` 分布式同步。

### 8. 完成后

用 [概念索引](docs/concept-index.md) 按需回查特定主题。[源码版本管理](docs/SOURCE-VERSION.md) 记录了每篇文章的 API 依赖和漂移状态。

---

## 练习速查

| 步骤 | 练习 | 命令 |
|------|------|------|
| 3. Fusion §惰性 | `src/burn-test` | `RUST_LOG=burn_fusion=trace cargo run --release` |
| 3. Fusion §竞标 | `docs/burn/fusion/1-client-server.md` | 纯读（教程章节） |
| 4. JIT §宏 | `src/ch2-expand-study` | `cargo test -- --nocapture` |
| 4. JIT §Launch | `src/ch1-gelu-variants` | `cargo test -- --nocapture` |
| 7. Autodiff §图 | `src/autodiff-test` | `cargo test -- --nocapture` |

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
│   ├── fusion/ (1-client-server)
│   └── onnx/                    burn/       (gitignored)
├── cubecl/                      cubecl/     (gitignored)
│   ├── autotune-system-design.md cubek/      (gitignored)
│   ├── jit-compilation-pipeline.md burn-onnx/  (gitignored)
│   ├── summary.md
│   ├── 1-gelu-launch.md
│   └── 2-expand.md
├── cubek/
│   ├── blueprint-routine-autotune.md
│   └── summary.md
└── appendix/
```

技术写作规范见 [CLAUDE.md](CLAUDE.md)。文档以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可发布。
