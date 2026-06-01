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
| **burn-test** | [docs/burn/fusion/1-client-server.md](../docs/burn/fusion/1-client-server.md) | [burn-test/](burn-test/) | `cd src/burn-test && RUST_LOG=burn_fusion=trace cargo run --release` |
| ch1-gelu-variants | [docs/cubecl/1-gelu-launch.md](../docs/cubecl/1-gelu-launch.md) | [ch1-gelu-variants/](ch1-gelu-variants/) | `cd src/ch1-gelu-variants && cargo test -- --nocapture` |
| ch2-expand-study | [docs/cubecl/2-expand.md](../docs/cubecl/2-expand.md) | [ch2-expand-study/](ch2-expand-study/) | `cd src/ch2-expand-study && cargo test -- --nocapture` |

## burn-test 预期日志

以 `RUST_LOG=burn_fusion=trace` 运行时，你会看到类似以下输出：

- `[explorer]` — Explorer 探索融合机会
- `[stream]` — StreamOptimizer 注册/停止
- `[plan]` — Policy 决策（cache hit / exploration completed）

四个操作（Clone, ScalarMul, ScalarAdd, Tanh）被融合为**一个** `elemwise_fuse` kernel。
