//! CubeCL 专题 · 第一章作业：GELU 变体跟练
//!
//! 对应文档：../../blog-cubecl-1.md
//!
//! 运行方式（在 homework/ch1-gelu-variants/ 下）：
//!   cargo test -- --nocapture
//!
//! 前置条件：项目根目录下已 clone cubecl 仓库（../../cubecl/）
//!
//! 两道作业：
//!   作业 1：修改 input 为 8 元素，对比 vector_size=1 与 4 的 CubeDim 推导
//!   作业 2：加 comptime! 常量观察 launch 签名不变；对比 #[comptime] 参数的影响

use cubecl::prelude::*;

// =========================================================================
// 原始 gelu kernel（与 cubecl/examples/gelu/src/lib.rs 一致，供参考）
// =========================================================================

#[cube(launch_unchecked)]
fn gelu_array<F: Float, N: Size>(input: &[Vector<F, N>], output: &mut [Vector<F, N>]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar(input[ABSOLUTE_POS]);
    }
}

#[cube]
fn gelu_scalar<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let sqrt2 = F::new(comptime!(2.0f32.sqrt()));
    let tmp = x / Vector::new(sqrt2);
    x * (Vector::erf(tmp) + Vector::one()) / Vector::new(F::new(2.0f32))
}

// =========================================================================
// 作业 1：vector_size 与 CubeDim 推导
// =========================================================================
// 原始 gelu 示例：4 元素，vector_size=4 → CubeDim::new_1d(1)。
// 本作业：8 元素，分别用 vector_size=1 和 4，推导 CubeDim。
//
// 公式：CubeDim 统计 unit 个数 = input.len() / vector_size
//   vector_size=1 → unit 个数 = 8 / 1 = 8
//   vector_size=4 → unit 个数 = 8 / 4 = 2

fn launch_vector1<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1.0, 0.0, 1.0, 5.0, -2.0, 3.0, -0.5, 0.5];
    let vector_size = 1;

    // ① todo!：推导 CubeDim
    //    input.len()=8, vector_size=1 → CubeDim::new_1d(8 / 1) = ?
    let cube_dim = todo!("CubeDim::new_1d(input.len() as u32 / vector_size)");

    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create_from_slice(f32::as_bytes(input));

    unsafe {
        gelu_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dim,
            vector_size,
            BufferArg::from_raw_parts(input_handle, input.len()),
            BufferArg::from_raw_parts(output_handle.clone(), input.len()),
        )
    };

    let bytes = client.read_one(output_handle).unwrap();
    let output = f32::from_bytes(&bytes);
    println!("vector_size=1 → CubeDim::new_1d(8), output={output:?}");
}

fn launch_vector4<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1.0, 0.0, 1.0, 5.0, -2.0, 3.0, -0.5, 0.5];
    let vector_size = 4;

    // ② todo!：推导 CubeDim
    //    input.len()=8, vector_size=4 → CubeDim::new_1d(8 / 4) = ?
    let cube_dim = todo!("CubeDim::new_1d(input.len() as u32 / vector_size)");

    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create_from_slice(f32::as_bytes(input));

    unsafe {
        gelu_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dim,
            vector_size,
            BufferArg::from_raw_parts(input_handle, input.len()),
            BufferArg::from_raw_parts(output_handle.clone(), input.len()),
        )
    };

    let bytes = client.read_one(output_handle).unwrap();
    let output = f32::from_bytes(&bytes);
    println!("vector_size=4 → CubeDim::new_1d(2), output={output:?}");
}

#[test]
fn homework_1_vector_sizes() {
    let device = Default::default();
    println!("=== 作业 1：vector_size 与 CubeDim ===");
    launch_vector1::<cubecl::cpu::CpuRuntime>(&device);
    launch_vector4::<cubecl::cpu::CpuRuntime>(&device);
    println!("验证：两次输出的数值应一致（GELU 结果相同），但 CubeDim 不同。");
}

// =========================================================================
// 作业 2：comptime! 常量 vs #[comptime] 参数
// =========================================================================
// 步骤 A：在 gelu_scalar 中加一个 comptime! 常量（如 scale 因子）。
//         观察：无需改 launch_unchecked 的签名和调用方式。
// 步骤 B：对比——若将 scale 改为 #[comptime] launch 参数，
//         launch 签名和缓存键都会变化。

// A. 加 comptime! 常量的 gelu（用于步骤 A）
#[cube]
fn gelu_scalar_scaled<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let sqrt2 = F::new(comptime!(2.0f32.sqrt()));
    // ③ todo!：加一个 comptime! 常量，例如让 GELU 结果放大 1.5 倍
    //    提示：let scale = comptime!(1.5f32);
    //          然后在最后乘上 F::new(scale)
    let _scale = todo!("comptime!(1.5f32) — 在 expand 时求值，不改变 launch 签名");

    let tmp = x / Vector::new(sqrt2);
    x * (Vector::erf(tmp) + Vector::one()) / Vector::new(F::new(2.0f32))
}

#[cube(launch_unchecked)]
fn gelu_array_scaled<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar_scaled(input[ABSOLUTE_POS]);
    }
}

// B. 对比：用 #[comptime] 参数的 gelu（用于步骤 B，展示差异）
#[cube]
fn gelu_scalar_comptime_param<F: Float, N: Size>(
    x: Vector<F, N>,
    _scale: F, // ← 这个参数不作为 GPU 运行时参数传入
) -> Vector<F, N> {
    let sqrt2 = F::new(comptime!(2.0f32.sqrt()));
    let tmp = x / Vector::new(sqrt2);
    x * (Vector::erf(tmp) + Vector::one()) / Vector::new(F::new(2.0f32))
}

#[cube(launch_unchecked)]
fn gelu_array_comptime_param<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    #[comptime] scale: F, // ← #[comptime] 改变了 launch 签名！
) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar_comptime_param(input[ABSOLUTE_POS], scale);
    }
}

#[test]
fn homework_2_comptime_constant() {
    let device = Default::default();
    let client = cubecl::cpu::CpuRuntime::client(&device);
    let input = &[-1.0, 0.0, 1.0, 5.0];
    let vector_size = 4;

    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create_from_slice(f32::as_bytes(input));

    println!("=== 作业 2A：comptime! 常量 ===");
    println!("  gelu_scalar_scaled 在函数体内用 comptime!(1.5) 计算 scale");
    println!("  launch_unchecked 的签名不变——无需额外参数。");
    unsafe {
        gelu_array_scaled::launch_unchecked::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(input.len() as u32 / vector_size as u32),
            vector_size,
            BufferArg::from_raw_parts(input_handle.clone(), input.len()),
            BufferArg::from_raw_parts(output_handle.clone(), input.len()),
        )
    };
    let bytes = client.read_one(output_handle.clone()).unwrap();
    let output = f32::from_bytes(&bytes);
    println!("  scale=1.5 output={output:?}");

    // ④ todo!：步骤 B — 调用带 #[comptime] scale 参数的 launch
    //    观察：launch 调用必须多传一个 scale 参数。
    //    这改变了 JIT 缓存键——scale 取不同值会生成不同的编译产物。
    println!();
    println!("=== 作业 2B：#[comptime] 参数 ===");
    println!("  #[comptime] scale 参数出现在 launch 签名中");
    println!("  每次 scale 变化 → JIT 缓存 miss → 重新编译");
    println!("  （取消下方注释以观察编译差异）");

    // 取消注释以运行：
    // unsafe {
    //     gelu_array_comptime_param::launch_unchecked::<f32, cubecl::cpu::CpuRuntime>(
    //         &client,
    //         CubeCount::Static(1, 1, 1),
    //         CubeDim::new_1d(input.len() as u32 / vector_size as u32),
    //         vector_size,
    //         // ⑤ todo!：这里需要多传 #[comptime] scale 参数
    //         // 比较：comptime! 常量不需要改这里，#[comptime] 参数需要
    //         1.5f32, // scale
    //         BufferArg::from_raw_parts(input_handle, input.len()),
    //         BufferArg::from_raw_parts(output_handle.clone(), input.len()),
    //     )
    // };
    println!("  （步骤 B 为观察性练习，不强制运行——重点是理解签名差异。）");
}

// =========================================================================
// 作业 2 对比总结（思考题，无代码）
// =========================================================================

#[test]
fn homework_2_comparison() {
    println!("=== 作业 2 对比总结 ===");
    println!();
    println!("  comptime! 常量                 #[comptime] 参数");
    println!("  ─────────────────────────────  ────────────────────────────────");
    println!("  在函数体内部定义                在 launch 签名中声明");
    println!("  不改变 launch_unchecked 签名    改变 launch 签名（多一个参数）");
    println!("  不改变 JIT 缓存键              改变 JIT 缓存键（多一个维度）");
    println!("  适用：固定不变的派生值          适用：调用方希望运行时指定的结构参数");
    println!("  示例：sqrt2, scale=1.5          示例：plane: bool, blueprint");
    println!();
    println!("  （以上为参考答案；完成作业 2A 后阅读。）");
}
