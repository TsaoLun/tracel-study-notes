# Tracel 学习笔记

> 深入 [Tracel](https://github.com/tracel-ai) 开源生态的源码级机制分析：Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。
>
> 在 Rust 类型系统、编译器流水线、GPU JIT 的层面解释"为什么这么设计"和"怎么做到的"。

---

## 快速开始

文档按两种形式组织：

| 类型 | 文件 | 职能 |
|------|------|------|
| **系统设计** | `*-system-design.md` | 从代码中提取设计决策与权衡：为什么这么设计、和主流方案的区别、限制。每篇对应一个核心系统。 |
| **章节教程** | `N-title.md` | 跟练教程、逐机制展开源码、带作业练习。 |

**推荐入口：[全景篇](docs/burn/burn-systems-architecture.md)** — 以一行 `z = (x*2.0+1.0).tanh(); z.backward()` 穿行 Fusion → Autotune → JIT → Autodiff 四个核心系统。

---

## 背景：Tracel 生态

| 项目 | 一句话 | 仓库 |
|------|--------|------|
| **Burn** | Rust 深度学习框架——用 trait 嵌套替代运行时 dispatch | [tracel-ai/burn](https://github.com/tracel-ai/burn) |
| **CubeCL** | 多平台 GPU 计算编译器——`#[cube]` 写 kernel，JIT 到 CUDA/HIP/WGPU/CPU | [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) |
| **CubeK** | 基于 CubeCL 的成品算子库（matmul、attention、convolution 等） | [tracel-ai/cubek](https://github.com/tracel-ai/cubek) |
| **Burn-ONNX** | ONNX→Rust AOT 编译器——构建期把模型翻译为可调试的 Rust 源码 | [tracel-ai/burn-onnx](https://github.com/tracel-ai/burn-onnx) |

```
入口 A：手写 Rust 模型 ──→ Burn（Autodiff + Fusion + Backend trait）
入口 B：ONNX 模型 ──→ build.rs / ModelGen（AOT）──→ 生成的 Rust + bpk ──→ Burn
                              ↓
                    burn-cubecl → CubeK 成品算子
                              ↓
                    CubeCL（#[cube] + IR + JIT + autotune）
                              ↓
                    CUDA / HIP / WGPU / CPU …
```

---

## 系统设计文章

五篇文章从设计决策出发，覆盖 Burn/CubeCL 技术栈的四个核心系统。每篇独立可读，合读形成完整的训练 step 全链路认知。

| 项目 | 文章 | 内容 |
|------|------|------|
| 全栈 | [burn-systems-architecture.md](docs/burn/burn-systems-architecture.md) | **全景篇（推荐入口）**：一行代码穿行 Fusion → Autotune → JIT → Autodiff |
| Burn | [kernel-fusion-system-design.md](docs/burn/kernel-fusion-system-design.md) | 惰性队列融合：OperationFuser 竞标、Stream/MultiStream 隔离、Page/Slice 内存 |
| Burn | [autodiff-system-design.md](docs/burn/autodiff-system-design.md) | 装饰器 Autodiff：类型状态图构建、BFS 逆序执行、分布式梯度同步 |
| CubeCL | [autotune-system-design.md](docs/cubecl/autotune-system-design.md) | 策略枚举 vs 参数网格、优先级提前终止、anchor 量化缓存 |
| CubeCL | [jit-compilation-pipeline.md](docs/cubecl/jit-compilation-pipeline.md) | `#[cube]` → IR → 优化 → WGSL/SPIR-V/MSL → GPU dispatch |

## 章节教程与导航

```
docs/
├── architecture.md                ← 跨项目架构主线："决策推迟"
│
├── burn/                          ← Burn 框架
│   ├── burn-systems-architecture.md ← 全景篇
│   ├── kernel-fusion-system-design.md ← Fusion 系统设计
│   ├── autodiff-system-design.md   ← Autodiff 系统设计
│   ├── summary.md                  ← 导航页
│   ├── onnx-summary.md             ← ONNX AOT
│   ├── fusion/                     ← Fusion 章节教程
│   │   ├── index.md                ← 8 章计划
│   │   └── 1-client-server.md      ← 章节：from_data → GPU buffer
│   └── onnx/                       ← ONNX 计划
│       └── index.md
│
├── cubecl/                        ← CubeCL 编译器
│   ├── autotune-system-design.md   ← Autotune 系统设计
│   ├── jit-compilation-pipeline.md ← JIT 编译管线
│   ├── summary.md                  ← 导航页
│   ├── index.md                    ← 8 章计划
│   ├── 1-gelu-launch.md            ← 章节：GELU 走通 launch
│   └── 2-expand.md                 ← 章节：expand 机制
│
├── cubek/                         ← CubeK
│   └── summary.md                  ← Blueprint-Routine-Autotuner
│
└── appendix/                      ← 附录
    └── automatic-kernel-fusion.md  ← 旧博客翻译
```

### 跨项目

| 类型 | 文档 | 主题 |
|------|------|------|
| 架构 | [architecture.md](docs/architecture.md) | 决策推迟：编译期 → JIT 时 → 首次执行——四项目共性 |

### Burn

| 类型 | 文档 | 主题 | 状态 |
|------|------|------|:---:|
| 全景 | [burn-systems-architecture.md](docs/burn/burn-systems-architecture.md) | Fusion → Autotune → JIT → Autodiff 全链路 | ✅ |
| 设计 | [kernel-fusion-system-design.md](docs/burn/kernel-fusion-system-design.md) | 惰性队列融合 | ✅ |
| 设计 | [autodiff-system-design.md](docs/burn/autodiff-system-design.md) | 装饰器 Autodiff | ✅ |
| 导航 | [summary.md](docs/burn/summary.md) | 项目索引 | ✅ |
| 地图 | [onnx-summary.md](docs/burn/onnx-summary.md) | ONNX→Rust AOT | ✅ |
| 计划 | [fusion/index.md](docs/burn/fusion/index.md) | Fusion 8 章计划 | ✅ |
| 章节 | [fusion/1-client-server.md](docs/burn/fusion/1-client-server.md) | from_data → GPU buffer | ✅ |
| 计划 | [onnx/index.md](docs/burn/onnx/index.md) | ONNX 6 章计划 | ✅ |
| 附录 | [appendix/automatic-kernel-fusion.md](docs/appendix/automatic-kernel-fusion.md) | 旧博客翻译 | ✅ |

### CubeCL

| 类型 | 文档 | 主题 | 状态 |
|------|------|------|:---:|
| 设计 | [autotune-system-design.md](docs/cubecl/autotune-system-design.md) | 策略枚举 + 优先级剪枝 | ✅ |
| 设计 | [jit-compilation-pipeline.md](docs/cubecl/jit-compilation-pipeline.md) | IR 设计 + 多平台代码生成 | ✅ |
| 导航 | [summary.md](docs/cubecl/summary.md) | 项目索引 | ✅ |
| 计划 | [index.md](docs/cubecl/index.md) | 8 章计划 | ✅ |
| 章节 | [1-gelu-launch.md](docs/cubecl/1-gelu-launch.md) | GELU walkthrough | ✅ |
| 章节 | [2-expand.md](docs/cubecl/2-expand.md) | expand 机制 | ✅ |

### CubeK

| 类型 | 文档 | 主题 | 状态 |
|------|------|------|:---:|
| 地图 | [cubek/summary.md](docs/cubek/summary.md) | Blueprint-Routine-Autotuner | ✅ |

---

## 仓库结构

```
tracel-study-notes/
├── README.md
├── CLAUDE.md                        ← 技术写作规范
├── .gitignore
│
├── docs/                            ← 所有分析文档
│   ├── architecture.md              ← 跨项目架构主线
│   ├── burn/
│   │   ├── burn-systems-architecture.md  ← 全景篇
│   │   ├── kernel-fusion-system-design.md ← Fusion 系统设计
│   │   ├── autodiff-system-design.md ← Autodiff 系统设计
│   │   ├── summary.md                ← 导航页
│   │   ├── onnx-summary.md
│   │   ├── fusion/
│   │   │   ├── index.md
│   │   │   └── 1-client-server.md
│   │   └── onnx/
│   │       └── index.md
│   ├── cubecl/
│   │   ├── autotune-system-design.md ← Autotune 系统设计
│   │   ├── jit-compilation-pipeline.md ← JIT 编译管线
│   │   ├── summary.md                ← 导航页
│   │   ├── index.md
│   │   ├── 1-gelu-launch.md
│   │   └── 2-expand.md
│   ├── cubek/
│   │   └── summary.md
│   └── appendix/
│       └── automatic-kernel-fusion.md ← 旧博客翻译
│
├── src/                             ← 示例与作业（Cargo workspace）
│   ├── Cargo.toml
│   ├── burn-test/
│   ├── ch1-gelu-variants/
│   ├── ch2-expand-study/
│   ├── ch3-trait-study/
│   ├── fusion-ch2-queue/
│   └── fusion-ch3-drain/
│
├── burn/          (gitignored)      ← tracel-ai/burn 参考源码
├── cubecl/        (gitignored)      ← tracel-ai/cubecl 参考源码
├── cubek/         (gitignored)      ← tracel-ai/cubek 参考源码
└── burn-onnx/     (gitignored)      ← tracel-ai/burn-onnx 参考源码
```

文档中源码引用使用各仓库根下的路径 + 符号名（如 `crates/burn-autodiff/src/backend.rs` 中的 `BackendTypes for Autodiff`）；行号仅作近似。跟练前 clone 参考仓库：

```bash
git clone https://github.com/tracel-ai/burn.git
git clone https://github.com/tracel-ai/burn-onnx.git
git clone https://github.com/tracel-ai/cubecl.git
git clone https://github.com/tracel-ai/cubek.git
```

---

## 源码版本

| 仓库 | commit | 说明 |
|------|--------|------|
| **burn** | `cfa867f13` (2026-06-05) | 五篇系统设计文章的源码基准 |
| **cubecl** | `ba103c7f` (2026-06-04) | JIT 管线和 autotune 的源码基准 |
| **burn-onnx** | main | 测试统计以 `expectations.toml` 为准 |
| **cubek** | main | matmul autotune 候选以源码为准 |

复验 burn-onnx 测试统计（在已 clone 的 `burn-onnx/` 下）：

```bash
grep -r 'insta::assert_snapshot' burn-onnx --include '*.rs' | wc -l
grep '^status' burn-onnx/crates/onnx-official-tests/expectations.toml | sort | uniq -c
```

---

## 写作约定

- **源码路径写全**：`crates/burn-fusion/src/stream/multi.rs`（符号 `MultiStream`），行号作近似参考
- **术语首次出现括号简注**，完整释义在各文档末尾的词汇说明表
- **系列导航**：每篇末尾有导航表，可跳转到任意相关文档
- **章节末尾有作业**，用于验证理解
- 技术写作规范详见 [CLAUDE.md](CLAUDE.md)

---

## 许可

文档内容以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可发布。所引用的 Tracel 项目源码各按其自有许可证（Apache 2.0 / MIT）。
