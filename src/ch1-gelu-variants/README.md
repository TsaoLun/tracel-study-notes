# ch1-gelu-variants: 写 GELU kernel 三种变体

验证 [JIT 编译管线](../../docs/cubecl/jit-compilation-pipeline.md) 中描述的 `#[cube]` 宏和 kernel launch。

## 运行

```bash
cd src/ch1-gelu-variants
cargo test -- --nocapture
```

## 三个测试

| 测试 | 函数 | 向量化 | CubeDim |
|------|------|--------|---------|
| `test_scalar` | `launch_scalar` | 标量（1 元素/work item） | (32,1,1) — 32 threads |
| `test_vector2` | `launch_vector2` | vec2（2 元素/work item） | (16,1,1) — 16 threads, 每个处理 2 元素 |
| `test_vector4` | `launch_vector4` | vec4（4 元素/work item） | (8,1,1) — 8 threads, 每个处理 4 元素 |

三种变体计算完全相同的 GELU 函数，产生相同输出，但使用不同的并行度。

## 观察点

1. `#[cube(launch)]` 宏做了什么——打开 `lib.rs`，看 `gelu_scalar`、`gelu_vector2`、`gelu_vector4` 三个函数。它们用完全相同的 Rust 代码描述 GELU，仅靠类型参数（`f32`、`vectorization::Float2`、`vectorization::Float4`）改变并行度。

2. `CubeDim` 的调整——标量用 (32,1,1)，vec4 用 (8,1,1)。因为 vec4 每个 work item 处理 4 个元素，只需 1/4 的线程数。

3. 测试用相同的输入验证三个变体产生相同的输出。这验证了不同向量化宽度只是实现细节，不改变计算结果。

## 理解要点

- 同一个 `#[cube]` 函数可以通过类型参数生成不同向量化宽度的 kernel。这是 CubeCL "comptime 特化" 的基础——不同的类型参数对应不同的 monomorphized 实例。
- 运行 `cargo expand`（需要 `cargo install cargo-expand`）可以看到 `#[cube]` 宏展开后的完整代码。
