# burn-test: 观察 Fusion 引擎

验证 [Fusion 系统设计](../../docs/burn/kernel-fusion-system-design.md) 中描述的惰性排队和融合机制。

## 运行

```bash
cd src/burn-test
RUST_LOG=burn_fusion=trace cargo run --release
```

## 预期输出

数值：`z = tanh((x*2.0+1.0))`，对 x=[[2,3],[4,5]] 的计算结果。

## 观察点

设置 `RUST_LOG=burn_fusion=trace` 后在日志中找：

1. `[stream]` 行 — 哪些操作被 StreamOptimizer 注册/拒绝。看到 `[stream] op 0 Init`、`op 1 NumericFloat(MulScalar)` 等入队记录。

2. `[plan]` 行 — Policy 的决策。"cache hit" 说明已有融合方案缓存命中，"exploration completed" 说明首次探索完成。

3. `New execution plan` 行 — 融合引擎产生了新方案（操作数 + trigger 数）。

4. `elemwise_fuse` 条目 — 生成的金属/WGSL 着色器名称。四个操作（Clone、MulScalar、AddScalar、Tanh）被融合为一个 elemwise_fuse kernel。

## 理解要点

- 第一次 `cargo run` 是冷启动：Policy 无缓存，Explorer 首次探索融合方案。第二次运行同一 binary：`ExecutionPlanStore` 有缓存，`[plan] cache hit` 直接执行。
- 对比 `RUST_LOG=burn_fusion=info` 和 `=trace` 的日志量差异——trace 级别展示了完整的探索过程。
