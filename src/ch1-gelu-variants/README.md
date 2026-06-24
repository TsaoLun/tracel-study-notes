# ch1-gelu-variants: GELU kernel 的向量化与 comptime 变体

跟练 [JIT 编译管线](../../docs/cubecl/jit-compilation-pipeline.md) 中描述的 `#[cube]` 宏和 kernel launch，对应章节教程 [1-gelu-launch.md](../../docs/cubecl/1-gelu-launch.md)。

> **对应的 NN 概念**：GELU 是一个激活函数，属于 element-wise（逐元素）算子——见 [primer · 算子三类](../../docs/primer.md#part-a--领域最小集)。这个练习是"一个具体 element-wise 算子在 GPU 上怎么落地"的样本。

## 运行

```bash
cd src/ch1-gelu-variants
cargo test -- --nocapture
```

> 在 CPU runtime（`cubecl::cpu::CpuRuntime`）上跑，无需 GPU。首次编译需要本地已 clone `cubecl`（见根目录 [README](../../README.md#setup首次使用)）。

## 三个测试

所有变体共用同一个 kernel `gelu_array<F, N>`（调用 `gelu_scalar`），通过类型参数 `N`（向量宽度）和 `#[comptime]` 参数改变行为。

| 测试 | 作业 | 观察点 |
|------|------|--------|
| `homework_1_vector_sizes` | 向量化宽度 vs CubeDim | `launch_vector1`（`vector_size=1` → `CubeDim::new_1d(8)`）与 `launch_vector4`（`vector_size=4` → `CubeDim::new_1d(2)`）算同一份 GELU，输出数值相同、线程数不同 |
| `homework_2_comptime_constant` | `comptime!` 常量 vs `#[comptime]` 参数 | `gelu_array_scaled`（函数体内 `comptime!` 常量，不改 launch 签名）对比 `gelu_array_comptime_param`（多一个 `#[comptime] bool` 参数，进入 `KernelId` / JIT 缓存键） |
| `homework_2_comparison` | 对比总结 | 打印两种 comptime 用法在"签名 / 缓存键 / 适用场景"上的差异表 |

## 预期输出

`cargo test -- --nocapture` 通过时，3 个测试全部 `ok`，stdout 含类似：

```
=== 作业 1：vector_size 与 CubeDim ===
vector_size=1 → CubeDim::new_1d(8), output=[...]
vector_size=4 → CubeDim::new_1d(2), output=[...]
验证：两次输出的数值应一致（GELU 结果相同），但 CubeDim 不同。
...
=== 作业 2A：comptime! 常量 ===
=== 作业 2B：#[comptime] bool 参数 ===
=== 作业 2 对比总结 ===
```

## 验证点

- `homework_1_vector_sizes` 两次打印的 `output=` 数值应逐元素一致——向量化宽度只改并行度，不改计算结果。
- `homework_2_comptime_constant` 中，2A 的 launch 签名与原始 `gelu_array` 相同，2B 多传一个 `scaled` 参数。这对应文档里"`#[comptime]` 参数进 JIT 缓存键"的论点。

## 理解要点

- 同一个 `#[cube]` 函数可以通过类型参数 `N: Size` 生成不同向量化宽度的 kernel——不同的类型参数对应不同的 monomorphized 实例。
- 运行 `cargo expand --lib`（需要 `cargo install cargo-expand`）可以看到 `#[cube]` 宏展开后的完整代码。下一个练习 [ch2-expand-study](../ch2-expand-study/) 专门观察这一步。
