# autodiff-test: 观察 Autodiff 梯度计算

验证 [Autodiff 系统设计](../../docs/burn/autodiff-system-design.md) 中描述的梯度图构建和反向传播。

## 运行

```bash
cd src/autodiff-test
cargo test -- --nocapture

# 同时观察融合日志（autodiff 触发的前向 fusion）
RUST_LOG=burn_fusion=trace cargo run --release
```

## 测试内容

`autodiff_gradient_matches_manual` — 对 `z = tanh(x*2.0+1.0)` 手动计算 `∂z/∂x = (1 - tanh²(2x+1)) × 2`，与 Burn 的 autodiff 结果逐元素对比。x = [[2,3],[4,5]]。

## 观察点

1. **`require_grad()`** — `x.require_grad()` 标记叶子节点需要梯度（`Requirement::Grad`）。

2. **`z.backward()` 返回 `Gradients`** — `backward()` 消费了 `z`，返回包含所有注册梯度的容器。注意 `backward` 之后的 `z` 不能再被使用。

3. **`x.grad(&grads)`** — 从 `Gradients` 容器中按 `NodeId` 查找 `x` 的梯度。如果 x 没有 require_grad 或 backward 未覆盖到此节点，返回 `None`。

4. **前向执行时机** — backward 会触发前向 drain——`z` 的前向操作（`*2.0`、`+1.0`、`tanh`）在 `z.backward()` 时才真正在 GPU 上执行。设 `RUST_LOG=burn_fusion=trace` 可以观察到。

## 理解要点

- 修改代码，在 `z.backward()` 之前加 `println!("{}", z)`——这会提前触发前向执行。然后观察 `backward()` 是否仍能正常工作（应该可以，因为图在前向执行时已构建）。
- 尝试对 `z` 调用两次 `backward()`——第二次会失败，因为图已在第一次 backward 中被消费。这验证了 Burn 不支持高阶梯度。
