# Burn 导航

Burn 是 Tracel 生态的深度学习框架。本项目围绕系统设计分析展开，涵盖融合引擎、自动微分和全链路架构。

## 系统设计文章

从代码中提取设计决策与权衡，适合后端/Infra 工程师建立系统认知：

| 文章 | 内容 |
|------|------|
| [burn-systems-architecture.md](burn-systems-architecture.md) | **全景篇**（推荐入口）：一行代码穿行 Fusion → Autotune → JIT → Autodiff |
| [kernel-fusion-system-design.md](kernel-fusion-system-design.md) | 惰性队列融合：OperationFuser 竞标、Stream/MultiStream 隔离、Page/Slice 内存模型 |
| [autodiff-system-design.md](autodiff-system-design.md) | 装饰器 Autodiff：类型状态图构建、BFS 逆序执行、分布式梯度同步 |

## 章节教程

跟练教程，逐步展开源码机制：

| 章节 | 内容 |
|------|------|
| [fusion/index.md](fusion/index.md) | Fusion 运行时 8 章计划 |
| [fusion/1-client-server.md](fusion/1-client-server.md) | Client-Server 架构：from_data 到 GPU buffer |

## 跨项目

- [架构主线](../architecture.md) — Tracel 生态共享的"决策推迟"设计哲学
- [ONNX 导入](onnx-summary.md) — ONNX 模型 AOT 编译为 Rust

---

→ 推荐入口：[全景篇](burn-systems-architecture.md) · [所有文章导航](../../README.md)
