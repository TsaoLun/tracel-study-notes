# 示例与作业

本目录是 Cargo workspace，包含跟练示例和作业骨架代码。

## 前置条件

项目根目录下需已 clone burn 和 cubecl 仓库：

```bash
git clone https://github.com/tracel-ai/burn.git
git clone https://github.com/tracel-ai/cubecl.git
```

## 文件索引

| 项目 | 对应文档 | 路径 | 运行方式 |
|------|----------|------|----------|
| **burn-test** | [docs/burn/fusion/1-client-server.md](../docs/burn/fusion/1-client-server.md) | [burn-test/](burn-test/) | `cd src/burn-test && BURN_FUSION_LOG=full cargo run --release` |
| ch1-gelu-variants | [docs/cubecl/1-gelu-launch.md](../docs/cubecl/1-gelu-launch.md) | [ch1-gelu-variants/](ch1-gelu-variants/) | `cd src/ch1-gelu-variants && cargo test -- --nocapture` |
| ch2-expand-study | [docs/cubecl/2-expand.md](../docs/cubecl/2-expand.md) | [ch2-expand-study/](ch2-expand-study/) | `cd src/ch2-expand-study && cargo test -- --nocapture` |
| ch3-trait-study | [docs/cubecl/3-trait-impl.md](../docs/cubecl/index.md#章节目录)（待写） | [ch3-trait-study/](ch3-trait-study/) | `cd src/ch3-trait-study && cargo test -- --nocapture` |
| fusion-ch2-queue | [docs/burn/fusion/2-operation-queue.md](../docs/burn/fusion/index.md#章节目录)（待写） | [fusion-ch2-queue/](fusion-ch2-queue/) | `cd src/fusion-ch2-queue && BURN_FUSION_LOG=full cargo run --release` |
| fusion-ch3-drain | [docs/burn/fusion/3-drain-processor.md](../docs/burn/fusion/index.md#章节目录)（待写） | [fusion-ch3-drain/](fusion-ch3-drain/) | `cd src/fusion-ch3-drain && BURN_FUSION_LOG=full cargo run --release` |

### 待建骨架（对应未来章节）

| 计划 crate | 对应文档 | 章节主题 |
|------------|----------|----------|
| `ch4-comptime-study` | CubeCL ch4 | comptime 与 JIT 缓存键 |
| `ch5-topology-study` | CubeCL ch5 | 拓扑与四轴并行 |
| `ch6-jit-pipeline` | CubeCL ch6 | JIT 管线：Scope → PTX/WGSL |
| `ch7-autotune-study` | CubeCL ch7 | vectorization 与 autotune |
| `ch8-cubek-burn` | CubeCL ch8 | CubeK 纪律与 Burn 边界 |
| `fusion-ch4-blocks` | Fusion ch4 | 增量融合：Block 注册与 Builder 评分 |
| `fusion-ch5-builder` | Fusion ch5 | FuseBlockBuilder：数据流分析 |
| `fusion-ch6-launch` | Fusion ch6 | FuseTrace → kernel launch |
| `fusion-ch7-elemwise` | Fusion ch7 | `elemwise_fuse` kernel |
| `fusion-ch8-cross-stream` | Fusion ch8 | 跨流共享与 channel 重构 |
| `onnx-ch1-ir` | ONNX ch1 | Protobuf → IR |

> 以上 crate 在对应章节写完时创建——Cargo.toml 模板见已存在的骨架。

## burn-test 预期日志

以 `BURN_FUSION_LOG=full` 运行时，你会看到类似以下输出：

- `[explorer]` — Explorer 探索融合机会
- `[stream]` — StreamOptimizer 注册/停止
- `[plan]` — Policy 决策（cache hit / exploration completed）

四个操作（Clone, ScalarMul, ScalarAdd, Tanh）被融合为**一个** `elemwise_fuse` kernel。
