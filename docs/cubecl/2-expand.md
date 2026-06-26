# CubeCL 专题 · 第二章：expand——`+` 如何变成 `__expand_add_method`

> **本章锚点**：GELU 示例中 `x / Vector::new(sqrt2)` 这行代码，从 Rust 语法树到 IR 里的 `Operation::Arithmetic(Div, …)`，中间经过两层转换。  
> **读完能干什么**：能读 `cubecl-macros/src/generate/expression.rs` 中的 `Expression::to_tokens` 匹配臂，解释为什么表达式不是「AST 直连 Operation」；能用 `ArithKernel::define()` 打印 expand 生成的 Scope；能读懂 CubeCL IR 文本格式。

> **前置**：[第一章](1-gelu-launch.md)（launch 调用链、`expand` 何时被调用）。术语见 [JIT 编译管线](jit-compilation-pipeline.md)。

---

## 本章在系列中的位置

| 文档 | 你得到什么 |
|------|------------|
| [专题一](1-gelu-launch.md) | launch 调用链：`launch_unchecked` → `define()` → `expand` 在哪被调用 |
| **本章** | expand 内部：表达式如何经两层方法调用最终向 `Scope` 注册 `Operation` |
| [专题三](index.md#第三章待写新增) | trait/impl 与 `#[define]`——泛型 kernel 如何注册 |

---

## 先看产物：`a + b * c` 生成的 IR 长什么样

在讨论"怎么做到的"之前，先看 expand 最终产出的东西。以下是在 `src/ch2-expand-study/` 中运行 `cargo test homework_2_ir_dump -- --nocapture` 捕获的完整 Scope 文本（完整文件：[../artifacts/arith-ir.txt](../artifacts/arith-ir.txt)）：

```
{
    binding(0) = buffer_len(global(0)) : (array<f32, global<0>>) -> (u32)
    ptr<slice>(1) = aggregate(global(0), u32(0), binding(0)) : () -> (array<f32, global<0>>)
    ...
    binding(19) = load(binding(18)) : (ptr<f32, global<1>>) -> (f32)    ← b[ABSOLUTE_POS]
    binding(24) = load(binding(23)) : (ptr<f32, global<2>>) -> (f32)    ← c[ABSOLUTE_POS]
    binding(25) = binding(19) * binding(24) : (f32, f32) -> (f32)       ← Mul 先注册
    binding(26) = binding(14) + binding(25) : (f32, f32) -> (f32)       ← Add 后注册
    ...
    store(binding(30), binding(26))
}
```

注意第 25 行和第 26 行：`binding(25) = binding(19) * binding(24)`（Mul）在 `binding(26) = binding(14) + binding(25)`（Add）**之前**。这是 Rust 运算符优先级的结果——`a + b * c` 中 `b * c` 先求值，proc-macro 忠实地保留了表达式的嵌套顺序。本章的核心就是解释：这一行 IR `binding(25) = binding(19) * binding(24)` 是从 Rust `a[ABSOLUTE_POS] + b[ABSOLUTE_POS] * c[ABSOLUTE_POS]` 中的 `b * c` 出发，经过两层转换才最终以 `scope.register(Instruction(Arithmetic::Mul, output))` 的形式写入 Scope 的。

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

## 为什么用两层？——设计动机

这里有一个值得展开的设计问题：**proc-macro 为什么不直接把 AST 节点一对一映射为 `Operation` 枚举构造调用，而要生成一个方法链？**

以 `a + b` 为例，"直译"方案是：

```rust
// 假想的「直译」方案——实际不存在
scope.register(Operation::Arithmetic(Add, BinaryOperands {
    lhs: a.into_expand(scope).expand,  // ← 需要先拿到 Value
    rhs: b.into_expand(scope).expand,
}), output);
```

直译方案需要在 proc-macro 内部决定 3 件事：
1. 用哪个 `Operation` 变体（`Arithmetic`、`Bitwise`、`Comparison`……）
2. 如何构造操作数（`BinaryOperands` 还是 `UnaryOperands`）
3. 输出的 `Value` 如何分配

而实际的两层方案**把这三件事的决策推迟到 trait 分发层**：

```rust
// 实际生成的代码（在 expand 函数内）
LeftExpandType::into_expand(left, scope).__expand_add_method(scope,
    RightExpandType::into_expand(right, scope))
```

推迟带来的好处：

1. **类型自适应**：`f32` 的 `__expand_add_method` 注册 `Arithmetic::Add(f32, f32)`；`Vector<f32, 4>` 的同一方法注册 `Arithmetic::Add(f32x4, f32x4)`——proc-macro **不需要知道**操作数类型，Rust 的 trait 分发在 expand 执行时自动选对。

2. **操作语义可扩展**：新增一种类型（如 `BF16`）只需在 `cubecl-core` 里为它实现 `__expand_add_method` 等一系列方法，proc-macro 代码不需要改动。如果 proc-macro 直译 `Operation::Arithmetic(Add, ...)`，每加一种类型就要改宏代码。

3. **简化 proc-macro 逻辑**：proc-macro 不需要理解"这个操作应该生成算术指令还是位运算指令"——它只需要知道"二元表达式 → `__expand_{op}_method`"，剩下的由 trait 分发解决。这使得 `expression.rs` 中的 `Binary` 分支可以写成一行：`left.__expand_add_method(scope, right)`，而不用写 `match (left_type, right_type) { (f32, f32) => ..., (Vector<f32,4>, Vector<f32,4>) => ..., ... }`。

4. **允许操作间的中间表示**：`into_expand` 返回的 `NativeExpand<T>` 不是裸 `Value`——它携带类型标记 `PhantomData<T>` 和 scope 上下文。后续的 `__expand_add_method` 可以据此在 `Scope` 里分配正确类型的输出 `Value`，并选择正确的 `Operation` 变体。

以 Rust 编译器的术语来说，这相当于在 proc-macro（前端）和 cubecl-core（后端）之间放了一个 **trait-based IR builder**：前端生成方法调用，后端实现这些方法。

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

它的作用是：把任意值转为「可调用 expand 方法」的形态。对于运行时值，`into_expand` 返回 `NativeExpand<T>`（携带 `scope` 上下文和对应的 `Value`）。

---

## `NativeExpand<T>` 与方法链

`NativeExpand<T>`（`cubecl-core/src/frontend/element/base.rs`）是 expand 阶段的统一包装：

```rust
pub struct NativeExpand<T: ?Sized> {
    pub expand: Value,        // IR 中的值句柄
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
pub(crate) fn binary_expand<F, Op>(scope: &Scope, lhs: Value, rhs: Value, func: F) -> Value
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

## 完整旅程：`a + b * c` 从源码到 IR——验证

以作业 kernel `output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS] * c[ABSOLUTE_POS]` 为例：

```
源码: a[ABSOLUTE_POS] + b[ABSOLUTE_POS] * c[ABSOLUTE_POS]

1. parse: Expression::from_expr
   → Expression::Binary {
       left: IndexAccess(a, ABSOLUTE_POS),
       operator: Add,
       right: Expression::Binary {
           left: IndexAccess(b, ABSOLUTE_POS),
           operator: Mul,
           right: IndexAccess(c, ABSOLUTE_POS)
       }
     }

2. generate: Expression::to_tokens 对 Add 节点：
   - 先 to_tokens 右子树（Mul 节点）
       → b.__expand_mul_method(scope, c) 先生成
   - 然后 left 的 into_expand
       → a.into_expand(scope)
   - 最后组合
       → a.into_expand(scope).__expand_add_method(scope, b.__expand_mul_method(scope, c))

3. JIT 时 expand 实际执行（按 IR 顺序）：
   binding(19) = load(b[ABSOLUTE_POS])   # b 的值
   binding(24) = load(c[ABSOLUTE_POS])   # c 的值
   binding(25) = binding(19) * binding(24)  # ← Mul 先执行
   binding(14) = load(a[ABSOLUTE_POS])   # a 的值
   binding(26) = binding(14) + binding(25)  # ← Add 后执行
   store(output[ABSOLUTE_POS], binding(26))
```

跟练时运行 `cargo test homework_2_ir_dump -- --nocapture` 可亲眼看到这段 IR 输出（完整产物见 [../artifacts/arith-ir.txt](../artifacts/arith-ir.txt)）。

---

## 从 IR 到 WGSL：expand 的产物去哪了

上面生成的 Scope（IR）下一步会进入 `cubecl-opt` 的定点循环优化，然后由 `WgslCompiler` 生成着色器代码。以 `arith_kernel` 为例，一个简化的对应关系：

```
CubeCL IR                                   WGSL
──────────────────────────────────────────  ────────────────────────────
global(0), global(1), ...                  @group(0) @binding(0) var<storage> buffer_0: ...
AbsolutePos                                let id = global_id.x * ... + local_id.x
binding(19) = load(b[...])                 let l_19 = buffer_1[id];
binding(24) = load(c[...])                 let l_24 = buffer_2[id];
binding(25) = binding(19) * binding(24)    let l_25 = l_19 * l_24;
binding(26) = binding(14) + binding(25)    let l_26 = l_14 + l_25;
store(output[...], binding(26))            buffer_3[id] = l_26;
```

> 完整的 GELU kernel WGSL 产物见 [Nihal Pasham 的 artifact gist](https://gist.github.com/nihalpasham/0ed25f2dbcb08278f79d6ceabf38a60b)，展示了 `#[cube]` kernel 经 WGSL 编译后的完整着色器代码。仓库本地的 WGSL 产物生成命令待参考仓库编译完成后补充至 `docs/artifacts/`。

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

上面这段会打印出本章开头展示的 IR 文本。宏还支持 `#[cube(launch, create_dummy_kernel)]`，会额外生成 `create_dummy_kernel(...)` 辅助函数；但 **`define()` 仍需要带 `device_properties` 的 `ComputeClient`**，跟练时直接 `ArithKernel::new` + `define()` 更直观。

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

## 本章决策时机

本章的核心机制是**两层转换**。每层的决策时机不同：

| 决策 | 时机 | 在哪个 crate |
|------|------|-------------|
| Rust AST → `Expression` 枚举（parse） | proc-macro 展开时（`cargo build`） | `cubecl-macros` |
| 编译期常量折叠（`left.is_const() && right.is_const()`） | proc-macro 展开时（`cargo build`） | `cubecl-macros`（parse 层） |
| `&&` / `||` 消解为 if/else | proc-macro 展开时（`cargo build`） | `cubecl-macros`（generate 层） |
| `Expression` → `__expand_*_method` 调用（generate） | proc-macro 展开时（`cargo build`） | `cubecl-macros`（generate 层） |
| `__expand_*_method` 实际执行 → `scope.register(Instruction(...))` | 首次 JIT miss 时（`define()` 内） | `cubecl-core`（host CPU 上执行） |
| `binary_expand` 分配输出 Value | 首次 JIT miss 时 | `cubecl-core` |
| `Operation::Branch(...)` 注册 | 首次 JIT miss 时 | `cubecl_ir::branch` |
| SSA 定点循环优化 | 首次 JIT miss 时（`define()` 之后） | `cubecl-opt` |
| IR → WGSL/CUDA/SPIR-V 代码生成 | 首次 JIT miss 时（优化之后） | `cubecl-wgpu` / `cubecl-cuda` / … |

**关键洞察**：proc-macro 做的事（两层转换）和 expand 做的事（注册 Instruction）发生在**完全不同的时间**。proc-macro 在 `cargo build` 时就完成了——它生成了一段 Rust 代码（`__expand_add_method` 调用链）。这段代码在首次 launch 的 JIT miss 时才真正执行，向 `Scope` 注册 `Instruction`。理解这个时间差，就理解了为什么 `comptime!` 里的代码能在 expand 执行时读取当前硬件属性——它运行在 JIT 编译的 host 端，不是 `cargo build` 时。

这也是"两层转换"的设计动机之一：parse + generate 在编译期完成，产出的是**一段可执行的 Rust 代码**而非静态 IR。这段代码在 JIT 时执行，可以根据运行时信息（当前硬件、comptime 参数）动态决定生成什么 IR。如果 proc-macro 直接产出 IR（直译方案），`comptime` 就不可能了——IR 在 `cargo build` 时就固定了。

---

## 小结

1. **两层转换**：parse 层（Rust AST → `Expression` 枚举）→ generate 层（`Expression` → `__expand_*_method` 调用）。proc-macro 不直接把 AST 节点翻译为 IR Operation。
2. **方法链**：`into_expand` 把值转为 `NativeExpand<T>`（携带 `scope` + `Value`），然后 `__expand_*_method` 在 `Scope` 中注册 `Instruction`。
3. **核心函数**：`binary_expand` 分配输出变量、构造 `BinaryOperands`、调用 `scope.register(Instruction(op, output))`。
4. **控制流**：`if` 展开为 `branch::if_expand`（注册 `Operation::Branch`），闭包确保 then/else 块各自独立向 `Scope` 注册指令。
5. **短路逻辑**：`&&`/`||` 被消解为 `if/else` 表达式，天然惰性求值。
6. **设计动机**：两层转换不是多此一举——它把"对什么类型注册什么 Operation"的决策从 proc-macro 推迟到 Rust trait 分发，使 proc-macro 保持简单、类型可扩展、comptime 成为可能。

---

## 作业

> 可运行骨架：[src/ch2-expand-study/](../../src/ch2-expand-study/)（`cd src/ch2-expand-study && cargo test -- --nocapture`）。

1. 在 `cubecl-macros/src/generate/expression.rs` 中找到 `Expression::FunctionCall`（两个匹配臂）和 `Expression::MethodCall` 的处理分支，写一段注释说明两者展开方式的差异。（提示：自由函数用 `{path}::expand` 或 `__expand_{name}`；方法调用用 `receiver.__expand_{method}_method`。）

2. 在骨架中补全/运行 `a + b * c` kernel：
   - **步骤一** `homework_2_verify`：CPU launch 验证 `2 + 3 × 4 = 14`
   - **步骤二** `homework_2_ir_dump`：`ArithKernel::new` + `define()` 打印 Scope，找出 `*`（`binding(25)` mul）先于 `+`（`binding(26)` add），以及 `binding(25)` 如何传给 add
   - **步骤三** `homework_2_ir_analysis`：阅读概念题参考答案

3. （选做）跟踪 `into_expand` 在 `cubecl-core/src/frontend/element/base.rs` 中的 trait 定义。列出至少 3 种实现了 `IntoExpand` 的类型，说明它们各自 `into_expand` 的行为差异。

4. （选做）阅读 [../artifacts/arith-ir.txt](../artifacts/arith-ir.txt) 的完整 IR 内容，尝试将每行 IR 映射回对应的 Rust 源码表达式。验证 Mul 在 Add 之前的 IR 顺序。

---

## 下章预告

**[第三章 · trait/impl 与 `#[define]`](index.md#章节目录)**（待写）：`Float` 泛型 kernel 的 expand 如何生成；`__expand_{method}` 命名规则在 trait 方法上的变化；CubeK 常见 `#[define(Lhs, Rhs)]` 签名。

---

*CubeCL 专题 · 源码 walkthrough · [阅读路径](../../README.md)（可选的延伸阅读）*
