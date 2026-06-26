# 示例与作业

本目录是 Cargo workspace，包含跟练示例和作业骨架代码。

## 前置条件

项目根目录下需已 clone burn 和 cubecl 仓库：

```bash
git clone https://github.com/tracel-ai/burn.git
git clone https://github.com/tracel-ai/cubecl.git
```

## 可运行练习

五个完整可运行的练习，对应阅读路径中的 `▶ 动手` callout：

| 练习 | 对应文档 | 运行方式 |
|------|----------|----------|
| **burn-test**（Fusion） | [kernel-fusion-system-design.md](../docs/burn/kernel-fusion-system-design.md) · [fusion/1-client-server.md](../docs/burn/fusion/1-client-server.md) | `cd src/burn-test && BURN_FUSION_LOG=full cargo run --release` |
| **fusion-ch2-queue**（Fusion） | [fusion/2-operation-queue.md](../docs/burn/fusion/2-operation-queue.md) | `cd src/fusion-ch2-queue && BURN_FUSION_LOG=full cargo run --release` |
| **autodiff-test**（Autodiff） | [autodiff-system-design.md](../docs/burn/autodiff-system-design.md) | `cd src/autodiff-test && cargo test -- --nocapture` |
| **ch1-gelu-variants**（JIT） | [1-gelu-launch.md](../docs/cubecl/1-gelu-launch.md) | `cd src/ch1-gelu-variants && cargo test -- --nocapture` |
| **ch2-expand-study**（JIT） | [2-expand.md](../docs/cubecl/2-expand.md) | `cd src/ch2-expand-study && cargo test -- --nocapture` |

每个练习的预期输出与验证点见各自的 README。

## 计划中骨架

以下 crate 是占位骨架，对应章节教程尚未写（见 [docs/ROADMAP.md](../docs/ROADMAP.md)）。当前**不可运行**，仅保留 workspace 结构：

| 骨架 crate | 对应（计划中）章节 | 主题 |
|------------|--------------------|------|
| `ch3-trait-study` | CubeCL ch3 | trait / impl 与 `#[define]` |
| `fusion-ch3-drain` | Fusion ch3 | Drain / Processor / Policy 状态机 |

> 章节写完时再为骨架补全可运行代码和 binary target。完整的写作进度见 [docs/ROADMAP.md](../docs/ROADMAP.md)。

## burn-test 预期日志

以 `BURN_FUSION_LOG=full` 运行时，关注三类日志（完整逐行解读见 [burn-test/README.md](burn-test/README.md)）：

- `[explorer]` — Explorer 探索融合机会
- `[stream]` — StreamOptimizer 注册 / 停止
- `[plan]` — Policy 决策（cache hit / exploration completed）

四个操作（Clone, ScalarMul, ScalarAdd, Tanh）被融合为**一个** `elemwise_fuse` kernel。
