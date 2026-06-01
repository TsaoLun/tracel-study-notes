# CubeCL 专题 · 第二章：expand——`+` 如何变成 `__expand_add_method`

> **本章锚点**：GELU 示例中 `x / Vector::new(sqrt2)` 这行代码，从 Rust 语法树到 IR 里的 `Operation::Arithmetic(Div, …)`，中间经过两层转换。  
> **读完能干什么**：能读 `cubecl-macros/src/generate/expression.rs` 中的 `Expression::to_tokens` 匹配臂，解释为什么表达式不是「AST 直连 Operation」；能用 `ArithKernel::define()` 打印 expand 生成的 Scope。

> **前置**：[第一章](1-gelu-launch.md)（launch 调用链、`expand` 何时被调用）。术语见 [summary 词汇表](summary.md#词汇说明表)。

---

## 本章在系列中的位置

| 文档 | 你得到什么 |
|------|------------|
| [专题一](1-gelu-launch.md) | launch 调用链：`launch_unchecked` → `define()` → `expand` 在哪被调用 |
| **本章** | expand 内部：表达式如何经两层方法调用最终向 `Scope` 注册 `Operation` |
| [专题三](index.md#第三章待写新增) | trait/impl 与 `#[define]`——泛型 kernel 如何注册 |

---

## 两层总图

`#[cube]` 函数里的 `a + b`，从 proc-macro 到 IR 经过两层：

```
第 1 层（parse）：Rust 源码
    → Expression::from_expr → Expression::Binary { left: a, operator: Add, right: b }
    （cubecl-macros/src/parse/expression.rs）

第 2 层（generate）：Expression::to_tokens
    → a.into_expand(scope).__expand_add_method(scope, b.into_expand(scope))
    （cubecl-macros/src/generate/expression.rs）
```

第 2 层的代码是在 expand 函数里**真正执行的 Rust 代码**——它在 JIT 时运行，通过 `__expand_add_method` → `binary_expand` → `scope.register(Instruction(Arithmetic::Add, output))` 把指令写入 IR。

**关键洞察**：proc-macro 不直接把 `a + b` 翻译成 `Operation::Arithmetic(Add, …)`。它翻译成一个**方法调用链**——`into_expand` 把值包装成 `NativeExpand`，`__expand_add_method` 在该包装上注册 Operation。两层之间存在一个**方法分发层**，这使不同类型的值（标量、向量、tensor）可以有不同的 expand 行为。

---

## 第一层：parse——Rust 语法树 → Expression 枚举

`cubecl-macros/src/parse/expression.rs` 中的 `Expression::from_expr` 负责这一步。以 `a + b` 为例（`Expr::Binary`，约 line 39）：

```rust
Expr::Binary(binary) => {
    let operator = parse_binop(&binary.op)?;          // syn::BinOp → Operator::Add
    let right = Self::from_expr(*binary.right, context)?;
    let left = Self::from_expr(*binary.left, context)?;

    if left.is_const() && right.is_const() {
        // 两个都是编译期常量 → 不生成 IR，直接算 Rust 值
        let left = left.as_const(context).unwrap();
        let right = right.as_const(context).unwrap();
        Expression::Verbatim { tokens: quote![(#left + #right)] }
    } else {
        // 至少一个运行时值 → 生成 expand 调用
        Expression::Binary {
            left: Box::new(left),
            operator: parse_binop(&binary.op)?,
            right: Box::new(right),
            span,
        }
    }
}
```

要点：若 `a` 和 `b` 都是编译期常量（`comptime!` 中的值），直接生成 Rust 加法；若涉及运行时变量，才构造 `Expression::Binary` 走 expand 路径。

`parse_binop`（`parse/operator.rs`）做 `syn::BinOp` → `Operator` 的映射：

| Rust | Operator 枚举 |
|------|:------------:|
| `+` | `Add` |
| `-` | `Sub` |
| `*` | `Mul` |
| `/` | `Div` |
| `%` | `Rem` |
| `==` | `Eq` |
| `&&` | `And` |
| `\|\|` | `Or` |

完整定义见 `cubecl-macros/src/operator.rs`，覆盖算术、比较、布尔、位运算的 assign 和非 assign 变体。

---

## 第二层：generate——Expression → IR 方法调用

`Expression::to_tokens`（`cubecl-macros/src/generate/expression.rs`，约 line 27）的核心分发逻辑。以 `Binary { operator: Add, … }` 为例（约 line 56–78）：

```rust
Expression::Binary { left, operator, right, span, .. } => {
    // operator.op_name() → "add"
    let op = format_ident!("__expand_{}_method", operator.op_name());
    // left → into_expand(left.to_tokens(context))
    let left = into_expand(left.to_tokens(context));
    // right → into_expand(right.to_tokens(context))
    let right = into_expand(right.to_tokens(context));
    let expand = with_span(context, *span,
        quote![#left.#op(scope, #rhs)]
    );
    quote! {{#expand}}
}
```

生成的代码等价于：

```rust
{
    let _left = IntoExpand::into_expand(a, scope);
    let _right = IntoExpand::into_expand(b, scope);
    // 比较操作 (Eq, Lt, …) 的 rhs 传引用 &_right，算术操作为移动
    _left.__expand_add_method(scope, _right)
}
```

`into_expand` 函数（同文件 line 21–24）：

```rust
fn into_expand(tokens: TokenStream) -> TokenStream {
    let into_expand = prelude_type("IntoExpand");
    quote![#into_expand::into_expand(#tokens, scope)]
}
```

它的作用是：把任意值转为「可调用 expand 方法」的形态。对于运行时变量，`into_expand` 返回 `NativeExpand<T>`（携带 `scope` 上下文和对应的 `Variable`）。

---

## `NativeExpand<T>` 与方法链

`NativeExpand<T>`（`cubecl-core/src/frontend/element/base.rs`）是 expand 阶段的统一包装：

```rust
pub struct NativeExpand<T: ?Sized> {
    pub expand: Variable,     // IR 中的变量句柄
    _type: PhantomData<T>,    // 编译期类型标记
}
```

它的 `IntoExpand` 实现是恒等的——已经是 expand 形态，不再转换：

```rust
impl<T: ?Sized> IntoExpand for NativeExpand<T> {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self::Expand { self }
}
```

`__expand_add_method` 通过 `impl_core_binop!` 宏生成（`cubecl-core/src/frontend/operation/binary.rs`，约 line 223–253，line 282 调用）：

```rust
impl_core_binop!(Add, add, Arithmetic::Add);
// 展开为：
//   trait CubeAdd { fn __expand_add_method(…) -> NativeExpand<Self>; }
//   impl<T: Add<Output=T> + CubePrimitive> CubeAdd for T {}
//   impl<T: CubePrimitive> AddExpand for NativeExpand<T> {
//       fn __expand_add_method(self, scope, rhs) -> Self {
//           binary_expand(scope, self.into(), rhs.into(), Arithmetic::Add).into()
//       }
//   }
```

`binary_expand`（`cubecl-core/src/frontend/operation/base.rs`，约 line 19）是最终向 `Scope` 注册指令的地方：

```rust
pub(crate) fn binary_expand<F, Op>(scope: &Scope, lhs: Variable, rhs: Variable, func: F) -> Variable
where F: Fn(BinaryOperands) -> Op, Op: Into<Operation>,
{
    let item = lhs.value_type();
    let output = scope.create_local(item);           // 分配输出变量
    let op = func(BinaryOperands { lhs, rhs });       // Arithmetic::Add { lhs, rhs }
    scope.register(Instruction::new(op, output));     // 注册到 Scope
    output
}
```

---

## 完整旅程：`a + b` 从源码到 IR

以 GELU 的 `x / Vector::new(sqrt2)` 为例（其中 `/` 走同一路径，`operator` 为 `Div`，对应 `__expand_div_method` → `Arithmetic::Div`）：

```
源码: x / sqrt2_vec

1. parse: Expression::from_expr
   → Expression::Binary {
       left: Variable(x),
       operator: Div,
       right: FunctionCall(Vector::new, [sqrt2]),
       span: …
     }

2. generate: Expression::to_tokens 生成的 Rust 代码（在 expand 函数内）:
   → IntoExpand::into_expand(x, scope)
       .__expand_div_method(scope,
         IntoExpand::into_expand(Vector::new(…), scope))

3. JIT 时 expand 执行:
   x.into_expand(scope) → NativeExpand<f32> { expand: Variable(id=3), … }
   sqrt2_vec → NativeExpand<Vector<f32,4>> { expand: Variable(id=7), … }

   x.__expand_div_method(scope, sqrt2_vec)
     → binary_expand(scope,
         var(id=3),           // lhs: Variable
         var(id=7),           // rhs: Variable
         Arithmetic::Div,     // op 构造器
       )

4. binary_expand:
   - let output = scope.create_local(Type::Scalar(f32));  // 分配新 Variable(id=10)
   - scope.register(Instruction(Arithmetic::Div {
         lhs: var(3), rhs: var(7)
     }, output=var(10)));
   - return var(10)

5. 返回值继续传递给下一个操作（* 或 +）
```

---

## `if` 的 expand：不是直接翻译

条件分支的 expand 走 `cubecl_ir::branch` 模块（约 line 287–339）：

```rust
// 简单 if（无 else）
Expression::If { condition, then_block, else_branch: None, .. } => {
    let condition = into_expand(condition.to_tokens(context));
    let then_block = then_block.to_tokens(context);
    quote! {
        #path::branch::if_expand(scope, #condition, |scope| #then_block);
    }
}

// if/else（作为表达式）
Expression::If { condition, then_block, else_branch: Some(else_branch), .. }
    if then_block.ret.is_some() && else_branch.needs_terminator() => {
    // 有返回值 → expr 版本
    quote! {
        #path::branch::if_else_expr_expand(scope, #condition, |scope| (#then_block))
            .or_else(scope, |scope| (#else_branch))
    }
}

// 编译期常量 if
Expression::If { condition, .. } if condition.is_const() => {
    // comptime 分支 → 直接生成普通 Rust if，不走 IR
    quote![if #as_const #then_block #else_branch]
}
```

`if_expand` 向 `Scope` 注册 `Operation::Branch(…)`，包含 then/else 各自的基本块。闭包 `|scope| { … }` 捕获 `scope`，确保 then/else 块内的 IR 指令注册到正确的控制流区域。

---

## 短路 `&&` / `||`：展开为条件分支

`a && b` 和 `a || b` 在 CubeCL IR 中没有直接的「逻辑与/或」Operation。它们被**消解为 `if/else`**（约 line 31–55）：

```rust
Expression::Binary {
    operator: op @ (Operator::Or | Operator::And), …
} if !left.is_always_pure() || !right.is_always_pure() => {
    let (then_block, else_block) = match op {
        // a || b  →  if a { true } else { b }
        Operator::Or => (quote![true], right),
        // a && b  →  if a { b } else { false }
        Operator::And => (right, quote![false]),
        _ => unreachable!(),
    };
    quote! {
        branch::if_else_expr_expand(scope, #left, |scope| #then_block)
            .or_else(scope, |scope| #else_block)
    }
}
```

这解释了为什么 `&&`/`||` 在 kernel 里**天然短路**——它们被展开为条件分支，右侧只在需要时才执行（通过闭包的惰性求值）。

> 注意：**纯**（pure，无副作用）操作数不受此规则约束，会走普通的 `__expand_and_method`/`__expand_or_method` 路径。

---

## 原子操作的注册模式

其他关键表达式类型及其展开模式：

| Rust 表达式 | expand 方法 | 注册的操作 |
|------------|-------------|-----------|
| `-a` | `__expand_neg_method` | `Arithmetic::Neg(a)` |
| `!a` | `__expand_not_method` | `Bitwise::Not(a)` |
| `a = b` | `__expand_assign_method` | `assign::expand(scope, value, target)` |
| `a[i]` | `__expand_index_method` | `Memory::Index(list, index)` |
| `a.field` | 直接展开（编译期） | 字段访问，无运行时 Operation |
| `func::<T>(args)` | `func::expand::<T>(scope, args)` | 取决于函数定义 |
| `a.method(args)` | `a.__expand_method_method(scope, args)` | 取决于方法定义 |
| `for i in range { … }` | `branch::for_expand(scope, range, unroll, \|…\| …)` | 循环体注册到独立 scope |
| `break` | `branch::break_expand(scope)` | 跳出循环 |

---

## 只看 IR，不 launch：`define()`

跟练时不必走完整 launch 管线。宏为每个 `#[cube(launch)]` kernel 生成 **`{Name}Kernel` 结构体** 与 **`CubeKernel::define()`**——在 Host CPU 上运行 `expand`，把指令填入 `Scope`，**不提交后端编译**。

跟练骨架（`src/ch2-expand-study/`）用法：

```rust
use cubecl::prelude::*;

let client = cubecl::cpu::CpuRuntime::client(&Default::default());
let settings = KernelSettings::default().cube_dim(CubeDim::new_1d(1));

let kernel = arith_kernel::ArithKernel::<f32, cubecl::cpu::CpuRuntime>::new(
    settings,
    client,
    BufferCompilationArg { inplace: None }, // 每个 buffer 参数各一份
    BufferCompilationArg { inplace: None },
    BufferCompilationArg { inplace: None },
    BufferCompilationArg { inplace: None },
);

println!("{}", kernel.define().body); // Scope 实现了 Display
```

宏还支持 `#[cube(launch, create_dummy_kernel)]`，会额外生成 `create_dummy_kernel(...)` 辅助函数；但 **`define()` 仍需要带 `device_properties` 的 `ComputeClient`**，跟练时直接 `ArithKernel::new` + `define()` 更直观。

CubeCL 还提供 `#[cube(create_dummy_kernel)]` 属性（见 `cubecl-macros` 文档），用于测试 harness——与上述 `define()` 路径目的一致。

---

## 当 expand 不是方法调用时

部分表达式直接对应 cubecl-core 中的独立函数（`Expression::FunctionCall`，约 line 173）：

```rust
Expression::FunctionCall { func, args, associated_type: None, .. } => {
    let args = map_args(args, context);  // 每个参数加 .into()
    quote![#path::expand #generics(scope, #(#args),*)]
}
```

例如 `plane_sum(input)` 被翻译为 `plane_sum::expand(scope, input.into())`——调用模块级 `expand` 函数而非类型方法，因为 `plane_sum` 是 CubeCL 前端函数。

---

## 小结

1. **两层转换**：parse 层（Rust AST → `Expression` 枚举）→ generate 层（`Expression` → `__expand_*_method` 调用）。proc-macro 不直接把 AST 节点翻译为 IR Operation。
2. **方法链**：`into_expand` 把值转为 `NativeExpand<T>`（携带 `scope` + `Variable`），然后 `__expand_*_method` 在 `Scope` 中注册 `Instruction`。
3. **核心函数**：`binary_expand` 分配输出变量、构造 `BinaryOperands`、调用 `scope.register(Instruction(op, output))`。
4. **控制流**：`if` 展开为 `branch::if_expand`（注册 `Operation::Branch`），闭包确保 then/else 块各自独立向 `Scope` 注册指令。
5. **短路逻辑**：`&&`/`||` 被消解为 `if/else` 表达式，天然惰性求值。

---

## 作业

> 可运行骨架：[src/ch2-expand-study/](src/ch2-expand-study/)（`cd src/ch2-expand-study && cargo test -- --nocapture`）。

1. 在 `cubecl-macros/src/generate/expression.rs` 中找到 `Expression::FunctionCall`（两个匹配臂）和 `Expression::MethodCall` 的处理分支，写一段注释说明两者展开方式的差异。（提示：自由函数用 `{path}::expand` 或 `__expand_{name}`；方法调用用 `receiver.__expand_{method}_method`。）

2. 在骨架中补全/运行 `a + b * c` kernel：
   - **步骤一** `homework_2_verify`：CPU launch 验证 `2 + 3 × 4 = 14`
   - **步骤二** `homework_2_ir_dump`：`ArithKernel::new` + `define()` 打印 Scope，找出 `*`（`binding(25)` mul）先于 `+`（`binding(26)` add），以及 `binding(25)` 如何传给 add
   - **步骤三** `homework_2_ir_analysis`：阅读概念题参考答案

3. （选做）跟踪 `into_expand` 在 `cubecl-core/src/frontend/element/base.rs` 中的 trait 定义。列出至少 3 种实现了 `IntoExpand` 的类型，说明它们各自 `into_expand` 的行为差异。

---

## 下章预告

**[第三章 · trait/impl 与 `#[define]`](index.md#章节目录)**（待写）：`Float` 泛型 kernel 的 expand 如何生成；`__expand_{method}` 命名规则在 trait 方法上的变化；CubeK 常见 `#[define(Lhs, Rhs)]` 签名。

---

*Burn 底层机制 · CubeCL 专题 · 第二章 · [系列索引](../../README.md)*
