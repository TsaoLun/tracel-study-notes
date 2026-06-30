# 写作计划与进度（ROADMAP）

本文是项目的单一进度页：哪些内容已完成、哪些在计划中。主阅读路径（[README](../README.md#学习地图)）只经过已完成内容，计划中的章节教程不在主线上，按兴趣选读。

## 已完成

### 系统设计文章（7 篇，主线）

从源码提取设计决策与权衡，互相用 `← | →` 链成闭环。

- [architecture.md](architecture.md) — 类型栈、Trait 边界与分层组合
- [burn-systems-architecture.md](burn/burn-systems-architecture.md) — 全景篇：一行代码穿行四个系统
- [kernel-fusion-system-design.md](burn/kernel-fusion-system-design.md) — Fusion 惰性队列融合
- [jit-compilation-pipeline.md](cubecl/jit-compilation-pipeline.md) — JIT 编译管线
- [autotune-system-design.md](cubecl/autotune-system-design.md) — Autotune 策略枚举与量化缓存
- [blueprint-routine-autotune.md](cubek/blueprint-routine-autotune.md) — CubeK 三层纪律
- [autodiff-system-design.md](burn/autodiff-system-design.md) — Autodiff 装饰器与图构建

### 章节教程（4 篇，可选延伸）

逐步展开源码机制的 walkthrough。

- [fusion/1-client-server.md](burn/fusion/1-client-server.md) — from_data 到 GPU buffer 的 client-server 链路
- [fusion/2-operation-queue.md](burn/fusion/2-operation-queue.md) — OperationQueue：惰性执行与"推迟了什么"
- [cubecl/1-gelu-launch.md](cubecl/1-gelu-launch.md) — GELU kernel 从 `#[cube]` 到 launch
- [cubecl/2-expand.md](cubecl/2-expand.md) — `#[cube]` 宏展开内部机制

### 练习 crate（5 个，完整可运行）

见 [src/README.md](../src/README.md)：`burn-test`、`fusion-ch2-queue`、`autodiff-test`、`ch1-gelu-variants`、`ch2-expand-study`。

## 计划中

以下章节教程与练习骨架尚未完成。详细的逐章计划保留在各专题的（已归档）`index.md` 中。

### Burn Fusion 章节教程

详细计划见 [fusion/index.md](burn/fusion/index.md)。

| 计划文件 | 标题 | 对应练习骨架 |
|----------|------|--------------|
| `fusion/3-drain-processor.md` | Drain 与 Processor：Policy 状态机 | `src/fusion-ch3-drain` |
| `fusion/4-block-scoring.md` | 增量融合：Block 注册与 Builder 评分 | `fusion-ch4-blocks`（待建） |
| `fusion/5-fuse-block-builder.md` | FuseBlockBuilder：数据流分析 | `fusion-ch5-builder`（待建） |
| `fusion/6-fuse-trace-launch.md` | 从 FuseTrace 到 kernel launch | `fusion-ch6-launch`（待建） |
| `fusion/7-elemwise-fuse.md` | `elemwise_fuse` kernel | `fusion-ch7-elemwise`（待建） |
| `fusion/8-cross-stream-channel.md` | 跨流共享与 channel 重构 | `fusion-ch8-cross-stream`（待建） |

### CubeCL 章节教程

详细计划见 [cubecl/index.md](cubecl/index.md)。

| 计划文件 | 标题 | 对应练习骨架 |
|----------|------|--------------|
| `cubecl/3-trait-impl.md` | trait / impl 与 `#[define]` | `src/ch3-trait-study` |
| `cubecl/4-comptime.md` | comptime 与 JIT 缓存键 | `ch4-comptime-study`（待建） |
| `cubecl/5-topology.md` | 拓扑与四轴 | `ch5-topology-study`（待建） |
| `cubecl/6-jit-pipeline.md` | JIT 管线：Scope → PTX/WGSL | `ch6-jit-pipeline`（待建） |
| `cubecl/7-vectorization-autotune.md` | vectorization 与 autotune | `ch7-autotune-study`（待建） |
| `cubecl/8-cubek-burn.md` | CubeK 纪律与 Burn 边界 | `ch8-cubek-burn`（待建） |

### Burn-ONNX 章节教程

Burn-ONNX 的系统设计分析尚未开始。详细计划见 [onnx/index.md](burn/onnx/index.md)，专题地图见 [onnx-summary.md](burn/onnx-summary.md)。

涵盖：Protobuf → IR、类型推断、注意力融合、simplify 定点循环、finalization → codegen、测试体系与 ModelGen（6 章）。

### 练习骨架 crate

`src/ch3-trait-study`、`src/fusion-ch3-drain` 当前是占位骨架（不可运行），对应章节写完时补全。其余待建 crate 在对应章节写作时创建，模板见已存在的骨架。

---

← [README](../README.md) · [源码版本管理](SOURCE-VERSION.md) · [概念索引](concept-index.md)
