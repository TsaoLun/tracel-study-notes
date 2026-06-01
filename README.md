# Tracel 学习笔记

> 深入 [Tracel](https://github.com/tracel-ai) 开源生态的源码级机制分析：Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。
>
> 不是 API 教程。是在 Rust 类型系统、编译器流水线、GPU JIT 的层面解释"为什么这么设计"和"怎么做到的"。

---

## 快速开始

文档按**主题 → 类型**组织。每个主题目录下：

| 类型 | 文件 | 职能 |
|------|------|------|
| **summary** | `summary.md` | 心智模型 + 架构连接 + 设计动机。读完知道"为什么这么设计"。不逐机制展开。（每篇 15–25 分钟） |
| **plan** | `index.md` | 专题写作计划 + 入门引导：示例是什么、建议阅读顺序、跟跑方式。 |
| **chapter** | `N-title.md` | 跟练教程、逐机制展开源码、带作业。 |

**建议先读 summary 建立全局视图，再读 chapter 跟练具体机制。**

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

## 文档

```
docs/
├── burn/                        ← Burn 框架底层机制
│   ├── summary.md               （地图）类型栈 + 融合流 + 框架开销
│   ├── onnx-summary.md          （地图）ONNX→Rust AOT 编译器
│   └── fusion/                  ← Burn Fusion 运行时专题
│       ├── index.md             （计划）8 章写作计划 + 入门引导
│       └── 1-client-server.md   （章节）双客户端-服务器：from_data → GPU buffer
│
└── cubecl/                      ← CubeCL 编译器
    ├── summary.md               （地图）#[cube] 宏展开、SSA、autotune
    ├── index.md                 （计划）8 章写作计划 + 入门引导
    ├── 1-gelu-launch.md         （章节）GELU 走通一条 launch
    └── 2-expand.md              （章节）expand：+ → __expand_add_method → IR
```

### Burn

| 类型 | 文档 | 主题 | 决策时机 | 状态 |
|------|------|------|----------|:---:|
| 地图 | [summary.md](docs/burn/summary.md) | 类型栈 + 融合流 + 框架开销 | `rustc` 单态化 + 训练 loop | ✅ |
| 地图 | [onnx-summary.md](docs/burn/onnx-summary.md) | ONNX→Rust AOT 编译器 | `cargo build` / `build.rs` | ✅ |
| 计划 | [fusion/index.md](docs/burn/fusion/index.md) | Fusion 运行时 8 章写作计划 | — | ✅ |
| 章节 | [fusion/1-client-server.md](docs/burn/fusion/1-client-server.md) | 双客户端-服务器：from_data → GPU buffer | — | ✅ |
| 章节 | fusion/2-operation-queue.md … fusion/8-cross-stream-channel.md | OperationQueue … channel 重构 | — | 📋 |

### CubeCL

| 类型 | 文档 | 主题 | 决策时机 | 状态 |
|------|------|------|----------|:---:|
| 地图 | [summary.md](docs/cubecl/summary.md) | `#[cube]` 宏展开、SSA 定点循环、autotune | 首次 kernel launch | ✅ |
| 计划 | [index.md](docs/cubecl/index.md) | 8 章写作计划 + 入门引导 | — | ✅ |
| 章节 | [1-gelu-launch.md](docs/cubecl/1-gelu-launch.md) | GELU 走通一条 launch | — | ✅ |
| 章节 | [2-expand.md](docs/cubecl/2-expand.md) | expand：`+` → `__expand_add_method` → IR | — | ✅ |
| 章节 | 3-trait-impl.md … 8-cubek-burn.md | trait、comptime、拓扑、JIT、autotune、CubeK/Burn | — | 📋 |

---

## 仓库结构

```
tracel-study-notes/
├── README.md
├── CLAUDE.md                        ← 技术写作规范
├── .gitignore
│
├── docs/                            ← 所有机制分析文档
│   ├── burn/
│   │   ├── summary.md
│   │   ├── onnx-summary.md
│   │   └── fusion/
│   │       ├── index.md
│   │       └── 1-client-server.md
│   └── cubecl/
│       ├── summary.md
│       ├── index.md
│       ├── 1-gelu-launch.md
│       └── 2-expand.md
│
├── src/                             ← 示例与作业（Cargo workspace）
│   ├── Cargo.toml
│   ├── README.md
│   ├── burn-test/                   ← Fusion 专题跟练：融合示例
│   ├── ch1-gelu-variants/           ← CubeCL 专题 1 作业
│   └── ch2-expand-study/            ← CubeCL 专题 2 作业
│
├── burn/          (gitignored)      ← tracel-ai/burn 参考源码
├── burn-onnx/     (gitignored)      ← tracel-ai/burn-onnx 参考源码
├── cubecl/        (gitignored)      ← tracel-ai/cubecl 参考源码
└── cubek/         (gitignored)      ← tracel-ai/cubek 参考源码
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

| 仓库 | 机制基准 | 说明 |
|------|----------|------|
| **burn** | v0.21.0（融合 channel 重构） | channel 重构对应该版本 |
| **burn-onnx** | main | 测试统计以 `expectations.toml` 为准 |
| **cubecl** / **cubek** | main | TileKind 等以 cubek 源码为准 |

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
