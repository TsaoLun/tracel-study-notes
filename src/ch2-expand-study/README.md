# ch2-expand-study: 观察 `#[cube]` 宏展开

验证 [JIT 编译管线](../../docs/cubecl/jit-compilation-pipeline.md) 中描述的表达式→IR 映射。

## 运行

```bash
cd src/ch2-expand-study
cargo test -- --nocapture
```

## 三个测试

| 测试 | 内容 | 观察点 |
|------|------|--------|
| `test_expand_gelu` | GELU 的宏展开产物 | 看 `__expand_add_method`、`__expand_mul_method` 等调用 |
| `test_expand_arithmetic` | `a + b * c` 的 IR 展开 | 看 `scope.register(Operation::Arithmetic(Arithmetic::Add(...)))` |
| `test_expand_branch` | `if` 的 IR 展开 | 看 `scope.child()` 嵌套 Scope 和 `Branch::If` |

## 观察点

1. **IR 不是文本输出，是运行中的 Rust 代码**——`cargo test` 在 CPU 上执行 `#[cube]` 函数的 expand 阶段，生成的 IR 以程序化方式构建（`scope.register(...)`），不是打印字符串。

2. **`a + b` 的真实形态**——在展开代码中找 `__expand_add_method`。这个方法是 CubeCL 在 `Float`/`Int` 等类型上为每个操作符自动生成的分发函数。

3. **`if` 分支的 Scope 树**——在 if 展开中找 `scope.child()`。每个控制流分支创建自己的子 Scope，形成嵌套树而非平铺的基本块——这是 CubeCL IR 的核心设计选择。

## 理解要点

- 运行 `cargo expand --lib`（需要 `cargo install cargo-expand`）可以看到完整的宏展开产物。对比原始 `#[cube]` 函数和展开后的 Rust 代码。
- `IntoExpand` trait 是表达式→IR 映射的关键：每个支持的类型实现了 `into_expand`，将 Rust 表达式转换为 `scope.register(...)` 调用。
