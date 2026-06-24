# ch2-expand-study: 观察 `#[cube]` 宏展开与 IR

跟练 [JIT 编译管线](../../docs/cubecl/jit-compilation-pipeline.md) 中描述的表达式→IR 映射，对应章节教程 [2-expand.md](../../docs/cubecl/2-expand.md)。

> **对应的 NN 概念**：`a + b * c` 是 element-wise 算术——观察的是框架如何把一行 Rust 表达式变成 GPU IR，而非 NN 语义本身。算子分类见 [primer · Part A](../../docs/primer.md#part-a--领域最小集)。

## 运行

```bash
cd src/ch2-expand-study
cargo test -- --nocapture
```

> 在 CPU runtime 上跑，无需 GPU。首次编译需要本地已 clone `cubecl`。

## 五个测试

围绕 `a + b * c` 这个表达式展开（作业 2 是核心，作业 1/3 是源码阅读引导）。

| 测试 | 作业 | 内容 |
|------|------|------|
| `homework_2_verify` | 作业 2 步骤一 | CPU launch `arith_kernel`，**断言** `2 + 3 * 4 == 14` |
| `homework_2_ir_dump` | 作业 2 步骤二 | 不 launch，直接 `ArithKernel::define()` 打印 expand 填入的 Scope IR |
| `homework_2_ir_analysis` | 作业 2 步骤三 | 打印概念题参考答案（为什么 Mul 先于 Add 注册） |
| `homework_1_doc` | 作业 1 | 打印 `expression.rs` 中 FunctionCall vs MethodCall 展开差异的阅读指引 |
| `homework_3_into_expand` | 作业 3（选做） | 打印 `IntoExpand` trait 实现的追踪指引（源码路径 + grep 命令） |

## 预期输出

5 个测试全部 `ok`。`homework_2_verify` 是唯一带数值断言的测试，stdout 含：

```
✓ 数值验证通过：2 + 3 * 4 = 14
```

`homework_2_ir_dump` 打印一段 Scope IR 文本，在其中能找到 Mul 对应的 `Operation`（先注册）、Add 对应的 `Operation`，以及中间变量如何把 mul 结果传给 add。

## 验证点

- `homework_2_verify` 通过 → expand 生成的 kernel 在 CPU 上算出了正确数值（Rust 优先级 `a + (b * c)`）。
- 对照 `homework_2_ir_dump` 的输出确认：**Mul 先于 Add 注册**。原因是 rustc 先求值 `b * c`，generate 展开 `+` 时先 `to_tokens` 右子树。这印证了文档里"两层转换把 Operation 注册推迟到 trait 分发"的结论。

## 理解要点

- IR 不是文本输出，而是运行中的 Rust 代码：`cargo test` 在 CPU 上执行 `#[cube]` 函数的 expand 阶段，IR 以 `scope.register(...)` 程序化构建。
- 运行 `cargo expand --lib`（需要 `cargo install cargo-expand`）可以看到完整宏展开产物。在其中找 `__expand_add_method`——CubeCL 为每个操作符自动生成的分发函数。
- `IntoExpand` trait 是表达式→IR 映射的关键：每个支持的类型实现 `into_expand`，将 Rust 表达式转换为 `scope.register(...)` 调用（作业 3 追踪）。
