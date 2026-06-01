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
//!   作业 2：launch 验证数值 + `define()` 打印 IR
//!   作业 3：选做     — 追踪 IntoExpand trait 实现

use cubecl::prelude::*;

// ---------------------------------------------------------------------------
// 作业 2：a + b * c — launch 验证 + IR 观察
// ---------------------------------------------------------------------------

#[cube(launch)]
fn arith_kernel<F: Float>(
    a: &[F],
    b: &[F],
    c: &[F],
    output: &mut [F],
) {
    if ABSOLUTE_POS < a.len() {
        // 作业：对照 IR，确认 Mul 先于 Add 注册（Rust 优先级：b * c 是 + 的右子树）
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS] * c[ABSOLUTE_POS];
    }
}

fn buffer_arg<R: Runtime>(handle: cubecl::server::Handle, len: usize) -> BufferArg<R> {
    unsafe { BufferArg::from_raw_parts(handle, len) }
}

/// 作业 2 步骤一：CPU launch 验证 a=2, b=3, c=4 → 14
#[test]
fn homework_2_verify() {
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    let a = &[2.0f32];
    let b = &[3.0f32];
    let c = &[4.0f32];

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));
    let c_handle = client.create_from_slice(f32::as_bytes(c));
    let output_handle = client.empty(a.len() * core::mem::size_of::<f32>());

    arith_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(a.len() as u32),
        buffer_arg(a_handle.clone(), a.len()),
        buffer_arg(b_handle, b.len()),
        buffer_arg(c_handle, c.len()),
        buffer_arg(output_handle.clone(), a.len()),
    );

    let bytes = client.read_one(output_handle).unwrap();
    let output = f32::from_bytes(&bytes);
    assert_eq!(output[0], 14.0, "2 + 3 * 4 should be 14");
    println!("✓ 数值验证通过：2 + 3 * 4 = {}", output[0]);
}

/// 作业 2 步骤二：不 launch GPU，直接 `define()` 打印 expand 填入的 Scope
#[test]
fn homework_2_ir_dump() {
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    let settings = KernelSettings::default().cube_dim(CubeDim::new_1d(1));
    let none = || BufferCompilationArg { inplace: None };

    let kernel = arith_kernel::ArithKernel::<f32, cubecl::cpu::CpuRuntime>::new(
        settings,
        client,
        none(),
        none(),
        none(),
        none(),
    );
    let def = kernel.define();

    println!("=== 作业 2：expand 生成的 Scope（IR 文本）===");
    println!("{}", def.body);
    println!();
    println!("在输出中找出：");
    println!("  - Mul 对应的 Operation（应先于 Add）");
    println!("  - Add 对应的 Operation");
    println!("  - 中间 Variable 如何把 mul 结果传给 add");
}

/// 作业 2 步骤三：概念题（对照 IR dump 阅读）
#[test]
fn homework_2_ir_analysis() {
    println!("=== 作业 2 概念题参考答案 ===");
    println!();
    println!("问题 A：*（Mul）先注册。rustc 先求值 b * c，再求 a + (b * c)。");
    println!();
    println!("问题 B：`a + (b * c)` 里的 + → __expand_add_method");
    println!();
    println!("问题 C：parse 后 AST 为 Add(a, Mul(b,c))。generate 展开 + 时");
    println!("        先 to_tokens 右子树 Mul，故 mul 的 register 先于 add。");
    println!();
    println!("（先跑 homework_2_ir_dump 对照 Scope 文本。）");
}

// ---------------------------------------------------------------------------
// 作业 1：FunctionCall vs MethodCall 展开差异（源码阅读 + 注释）
// ---------------------------------------------------------------------------
// 打开 cubecl/crates/cubecl-macros/src/generate/expression.rs：
//   Expression::FunctionCall → 约 line 173（associated_type: None）
//                          → 约 line 202（associated_type: Some）
//   Expression::MethodCall  → 约 line 222

#[test]
fn homework_1_doc() {
    println!("作业 1：在 cubecl-macros/src/generate/expression.rs 中完成注释");
    println!();
    println!("  FunctionCall (无关联类型) → path::expand::<G>(scope, arg.into(), ...)");
    println!("  FunctionCall (有关联类型) → ty_path::__expand_{{name}}(scope, args...)");
    println!("  MethodCall               → receiver.__expand_{{method}}_method(scope, args...)");
    println!();
    println!("  差异：自由函数走模块级 ::expand 或关联 expand；");
    println!("        方法调用挂在 receiver 的 __expand_*_method 上。");
}

// ---------------------------------------------------------------------------
// 作业 3（选做）：IntoExpand trait 类型实现追踪（源码阅读）
// ---------------------------------------------------------------------------

#[test]
fn homework_3_into_expand() {
    let base_rs = "../../cubecl/crates/cubecl-core/src/frontend/element/base.rs";
    println!("作业 3：追踪 IntoExpand trait 实现");
    println!("  trait 定义：{base_rs}（line 58）");
    println!("  搜索：grep -n 'impl.*IntoExpand' {base_rs}");
    println!();
    println!("  目标：至少 3 种实现，记录行号与 into_expand 行为。");
    println!("  提示：NativeExpand<T>（line 561，恒等）、&T（line 63，透传引用）、");
    println!("        元组 (A,B,…)（line 670，递归 into_expand）。");
}
