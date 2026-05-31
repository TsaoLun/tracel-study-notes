//! CubeCL 专题 · 第二章作业：expand 机制
//!
//! 对应文档：../../blog-cubecl-2.md
//!
//! 运行方式（在 homework/ch2-expand-study/ 下）：
//!   cargo test -- --nocapture
//!
//! 前置条件：项目根目录下已 clone cubecl 仓库（../../cubecl/）
//!
//! 三道作业：
//!   作业 1：源码阅读 — 在 expression.rs 中添加注释
//!   作业 2：可运行   — 补全 kernel 后 launch 验证，再观察 IR
//!   作业 3：选做     — 追踪 IntoExpand trait 实现

use cubecl::prelude::*;

// ---------------------------------------------------------------------------
// 作业 2：补全 kernel 并在 launch 中观察 a + b * c 的 IR
// ---------------------------------------------------------------------------
// kernel 取 3 个输入数组和 1 个输出数组，每个 unit 算一个元素。

#[cube(launch)]
fn arith_kernel<F: Float>(
    a: &[F],
    b: &[F],
    c: &[F],
    output: &mut [F],
) {
    if ABSOLUTE_POS < a.len() {
        // ① todo!：实现 output[ABSOLUTE_POS] = a + b * c
        todo!("output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS] * c[ABSOLUTE_POS]");
    }
}

/// 作业 2 步骤一：补全 kernel 后，取消下方 #[ignore] 并跑 cargo test。
///   用 CPU runtime 验证数值正确性（a=2, b=3, c=4 → output=14）。
#[test]
#[ignore] // ② 补全 kernel 后删除此行
fn homework_2_verify() {
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    let a = &[2.0f32];
    let b = &[3.0f32];
    let c = &[4.0f32];

    let a_handle = client.create_from_slice(a);
    let b_handle = client.create_from_slice(b);
    let c_handle = client.create_from_slice(c);
    let output_handle = client.empty(a.len() * core::mem::size_of::<f32>());

    unsafe {
        arith_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(a.len() as u32),
            &a_handle,
            &b_handle,
            &c_handle,
            &output_handle,
        );
    }

    let output: Vec<f32> = client.read_one(&output_handle).unwrap();
    assert_eq!(output[0], 14.0, "2 + 3 * 4 should be 14");
    println!("✓ 数值验证通过：2 + 3 * 4 = {}", output[0]);
}

/// 作业 2 步骤二：完成数值验证后，用 println 回答以下问题（无需改代码）：
///
///   问题 A：表达式 a + b * c 在 expand 阶段，哪个操作先被注册到 Scope？
///   问题 B：`a + b` 中的 `+` 被 translate 为哪个 __expand_*_method？
///   问题 C：为什么 `b * c` 中的 `*` 先于 `+` 注册，尽管 `a + …` 写在前面？
///
///   提示：回忆 Rust 运算符优先级规则，再对照 blog-cubecl-2.md 中
///         Expression::Binary 的展开逻辑。expand 生成的 IR 指令顺序
///         与 Rust 求值顺序一致。
#[test]
fn homework_2_ir_analysis() {
    println!("=== 作业 2 步骤二：IR 分析 ===");
    println!();
    println!("问题 A：*（mul）先注册。因为 rustc 先求值 b * c，");
    println!("         再求值 a + (b * c)。expand 按求值顺序注册指令。");
    println!();
    println!("问题 B：__expand_add_method");
    println!();
    println!("问题 C：运算符优先级——* 高于 +。在 Expression::Binary");
    println!("        的 parse 阶段，b * c 是 + 的 right 操作数，");
    println!("        其 to_tokens 先生成 mul 的 register，");
    println!("        然后才是 add 的 register。");
    println!();
    println!("（以上为参考答案；补全 kernel 并通过 homework_2_verify 后阅读。）");
}

// ---------------------------------------------------------------------------
// 作业 1：FunctionCall vs MethodCall 展开差异（源码阅读 + 注释）
// ---------------------------------------------------------------------------
// 打开 cubecl/crates/cubecl-macros/src/generate/expression.rs：
//   Expression::FunctionCall → 约 line 173
//   Expression::MethodCall  → 约 line 222
// 在对应位置上方添加注释，说明两者的展开模式差异。

#[test]
fn homework_1_doc() {
    println!("作业 1：在 cubecl-macros/src/generate/expression.rs 中完成注释");
    println!("  FunctionCall → 展开为 path::expand::<G>(scope, arg1.into(), ...)");
    println!("  MethodCall  → 展开为 receiver.__expand_{{method}}_method(scope, args...)");
    println!("  差异：FunctionCall 用模块级 ::expand 自由函数；");
    println!("        MethodCall 的 expand 挂在 receiver 实例上。");
}

// ---------------------------------------------------------------------------
// 作业 3（选做）：IntoExpand trait 类型实现追踪（源码阅读）
// ---------------------------------------------------------------------------
// 在 cubecl-core/src/frontend/element/base.rs 中：
//   1. 找到 trait IntoExpand 定义（约 line 58）
//   2. 搜索 impl.*IntoExpand，列出至少 3 种实现

#[test]
fn homework_3_into_expand() {
    let base_rs = "../../cubecl/crates/cubecl-core/src/frontend/element/base.rs";
    println!("作业 3：追踪 IntoExpand trait 实现");
    println!("  trait 定义：{base_rs}（约 line 58）");
    println!("  搜索：grep -n 'impl.*IntoExpand' {base_rs}");
    println!();
    println!("  目标：找到至少 3 种实现，记录行号和 into_expand 行为。");
    println!("  提示：关注 NativeExpand<T>（恒等）、&T（透传）、");
    println!("        元组 (A,B)（递归调用）的差异。");
}
