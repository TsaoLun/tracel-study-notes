# CubeCL 导航

CubeCL 是 Tracel 生态的多平台 GPU 编译器。本项目围绕系统设计分析展开，涵盖 JIT 编译管线和自动调参。

## 系统设计文章

从代码中提取设计决策与权衡：

| 文章 | 内容 |
|------|------|
| [autotune-system-design.md](autotune-system-design.md) | 策略枚举 vs 参数网格、优先级提前终止、anchor 量化缓存、与 Triton 对比 |
| [jit-compilation-pipeline.md](jit-compilation-pipeline.md) | `#[cube]` → IR → 优化 → WGSL/SPIR-V/MSL → GPU dispatch 的完整管线 |

## 章节教程

跟练教程，逐步展开源码机制：

| 章节 | 内容 |
|------|------|
| [index.md](index.md) | CubeCL 8 章写作计划 |
| [1-gelu-launch.md](1-gelu-launch.md) | GELU kernel：从 `#[cube]` 到 launch 的完整 walkthrough |
| [2-expand.md](2-expand.md) | expand：Rust `+` 如何变成 `__expand_add_method` |

## 跨项目

- [架构主线](../architecture.md) — Tracel 生态共享的"决策推迟"设计哲学
