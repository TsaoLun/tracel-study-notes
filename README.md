# Tracel 学习笔记

> 深入 [Tracel](https://github.com/tracel-ai) 开源生态的系统设计分析：Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。
>
> 从设计决策出发——和主流方案的区别是什么、为什么这么做、限制在哪里。

---

## 知识图谱

```
                        ┌─────────────────────────────────────────┐
                        │         architecture.md                  │
                        │  "决策推迟：编译期 → JIT → 首次执行"     │
                        │          （跨项目设计哲学地图）           │
                        └──────────────┬──────────────────────────┘
                                       │ 先读这个建立坐标系
                                       ▼
                        ┌─────────────────────────────────────────┐
                        │      burn-systems-architecture.md        │
                        │         "一行代码穿行四个系统"            │
                        │          （推荐入口，15-20 分钟）          │
                        └──┬─────────┬─────────┬─────────┬────────┘
                           │         │         │         │
              ┌────────────▼──┐ ┌────▼──────┐ ┌▼──────────▼──┐ ┌▼─────────────┐ ┌─────────────┐
              │ Fusion 系统设计│ │Autotune   │ │ JIT 编译管线 │ │CubeK Blueprint│ │ Autodiff    │
              │               │ │ 系统设计   │ │              │ │ 架构纪律    │ │ 系统设计    │
              │ "融合怎么做的" │ │"最快实现  │ │ "#[cube]到GPU│ │"怎么防kernel│ │ "梯度怎么算"│
              │ 竞标·隔离·内存│ │ 怎么选的"  │ │ 二进制的管线"│ │  爆炸"       │ │ 装饰器·BFS  │
              └───────┬───────┘ └─────┬──────┘ └──────┬───────┘ └──────┬──────┘ └──────┬──────┘
                      │               │               │                │                │
          ┌───────────▼───┐  ┌────────▼──────┐ ┌──────▼───────┐       │       ┌────────▼──────┐
          │ src/burn-test  │  │               │ │src/ch1-gelu- │       │       │src/autodiff-  │
          │ "上手融合日志" │  │               │ │   variants   │       │       │    test       │
          └───────────────┘  └───────────────┘ │"GELU作业"    │       │       │ "梯度验证"    │
                                               │src/ch2-expand│       │       └───────────────┘
                                               │ "宏展开作业" │       │
                                               └──────────────┘       │

                          ┌─────────────────┐
                          │ 章节跟练教程      │
                          │ cubecl/1-gelu-   │ ← 跟着 #[cube] 写出第一个 GPU kernel
                          │   launch.md      │
                          │ cubecl/2-expand.md│ ← 看 Rust `+` 如何变成 IR 指令
                          │ burn/fusion/1-   │ ← 从 from_data 到 GPU buffer
                          │   client-server  │
                          └─────────────────┘
```

## 阅读路径

### 路径 A：建立系统全貌（60 分钟）

1. **[architecture.md](docs/architecture.md)**（15 分钟）— 跨项目设计哲学地图。理解 "决策推迟" 如何在三层上运作。
2. **[全景篇](docs/burn/burn-systems-architecture.md)**（20 分钟）— 一行代码穿行四个核心系统，建立全链路认知。
3. 任选一篇系统设计文章深入（25 分钟/篇）。

### 路径 B：直入主题

对特定系统感兴趣，直接读对应的系统设计文章。每篇末尾有 `← 上一篇 | 下一篇 →` 导航。

### 路径 C：跟练源码（动手）

读完系统设计文章后，做对应的章节练习题：

| 读了 | 去练 | 说明 |
|------|------|------|
| JIT 编译管线 | `src/ch1-gelu-variants/` | 写 GELU kernel 的三种变体 |
| JIT 编译管线 | `src/ch2-expand-study/` | 观察 `#[cube]` 宏展开的 IR |
| Fusion 系统设计 | `src/burn-test/` | `RUST_LOG=burn_fusion=trace` 观察融合 |
| Autodiff 系统设计 | `src/autodiff-test/` | 观察梯度图构建和反向传播 |

---

## 文档

### 系统设计文章（5 篇）

| # | 文章 | 内容 | 练习 |
|---|------|------|:--:|
| 0 | [architecture.md](docs/architecture.md) | 四项目共享的设计哲学：决策推迟 | — |
| 1 | [全景篇](docs/burn/burn-systems-architecture.md) | Fusion → Autotune → JIT → CubeK → Autodiff 全链路 | — |
| 2 | [Fusion](docs/burn/kernel-fusion-system-design.md) | 惰性队列融合：竞标机制、Stream 隔离、Page/Slice 内存 | burn-test |
| 3 | [Autotune](docs/cubecl/autotune-system-design.md) | 策略枚举 vs 参数网格、优先级剪枝、anchor 缓存 | — |
| 4 | [JIT 编译管线](docs/cubecl/jit-compilation-pipeline.md) | `#[cube]` → IR → WGSL/SPIR-V/MSL → GPU | ch1, ch2 |
| 5 | [CubeK](docs/cubek/blueprint-routine-autotune.md) | Blueprint/Routine/Autotuner 三层纪律：如何防止 kernel 组合爆炸 | — |
| 6 | [Autodiff](docs/burn/autodiff-system-design.md) | 装饰器模式、类型状态图构建、BFS 逆序 | autodiff-test |

### 章节教程

| 项目 | 已完成 | 计划中 |
|------|--------|--------|
| CubeCL | [1-gelu-launch](docs/cubecl/1-gelu-launch.md), [2-expand](docs/cubecl/2-expand.md) | [8 章计划](docs/cubecl/index.md) |
| Burn Fusion | [1-client-server](docs/burn/fusion/1-client-server.md) | [8 章计划](docs/burn/fusion/index.md) |
| Burn ONNX | — | [6 章计划](docs/burn/onnx/index.md) |

### 工具文档

| 文档 | 说明 |
|------|------|
| [concept-index.md](docs/concept-index.md) | 概念索引：~70 个关键概念 → 文章位置 |
| [SOURCE-VERSION.md](docs/SOURCE-VERSION.md) | 源码基准、API 依赖矩阵、漂移检查清单 |

### 项目地图

| 项目 | 导航页 | 章节计划 |
|------|--------|----------|
| Burn | [summary.md](docs/burn/summary.md) | fusion/, onnx/ |
| CubeCL | [summary.md](docs/cubecl/summary.md) | index.md |
| CubeK | [summary.md](docs/cubek/summary.md) | — |

---

## 源码版本

| 仓库 | commit | 日期 |
|------|--------|------|
| burn | `cfa867f13` | 2026-06-05 |
| cubecl | `ba103c7f` | 2026-06-04 |
| burn-onnx | main | — |
| cubek | main | — |

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
docs/          分析文档                src/         Cargo workspace 练习
├── architecture.md                  ├── burn-test/
├── burn/                            ├── ch1-gelu-variants/
│   ├── burn-systems-architecture.md ├── ch2-expand-study/
│   ├── kernel-fusion-system-design.md ├── ch3-trait-study/
│   ├── autodiff-system-design.md    ├── fusion-ch2-queue/
│   ├── summary.md                   └── fusion-ch3-drain/
│   ├── fusion/ (教程)
│   └── onnx/   (计划)              burn/       (gitignored 参考源码)
├── cubecl/                         cubecl/     (gitignored 参考源码)
│   ├── autotune-system-design.md   cubek/      (gitignored 参考源码)
│   ├── jit-compilation-pipeline.md  burn-onnx/  (gitignored 参考源码)
│   ├── summary.md
│   ├── 1-gelu-launch.md
│   └── 2-expand.md
├── cubek/summary.md
└── appendix/
```

技术写作规范见 [CLAUDE.md](CLAUDE.md)。文档以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可发布。
