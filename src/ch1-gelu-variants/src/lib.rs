//! CubeCL 专题 · 第一章作业：GELU 变体跟练
//!
//! 对应文档：../../../docs/cubecl/1-gelu-launch.md
//!
//! 运行方式（在 src/ch1-gelu-variants/ 下）：
//!   cargo test -- --nocapture
//!
//! 前置条件：项目根目录下已 clone cubecl 仓库（../../cubecl/）
//!
//! 两道作业：
//!   作业 1：修改 input 为 8 元素，对比 vector_size=1 与 4 的 CubeDim 推导
//!   作业 2：加 comptime! 常量观察 launch 签名不变；对比 #[comptime] 参数的影响

use cubecl::num_traits::One;
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
// 公式：CubeDim 统计 unit 个数 = input.len() / vector_size
//   vector_size=1 → 8 / 1 = 8
//   vector_size=4 → 8 / 4 = 2

fn launch_vector1<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1.0, 0.0, 1.0, 5.0, -2.0, 3.0, -0.5, 0.5];
    let vector_size = 1;

    // ① 推导：CubeDim::new_1d(8)
    let cube_dim = CubeDim::new_1d(input.len() as u32 / vector_size as u32);

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

    // ② 推导：CubeDim::new_1d(2)
    let cube_dim = CubeDim::new_1d(input.len() as u32 / vector_size as u32);

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

// A. comptime! 常量：不改变 launch 签名
#[cube]
fn gelu_scalar_scaled<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let sqrt2 = F::new(comptime!(2.0f32.sqrt()));
    let scale = comptime!(1.5f32);
    let tmp = x / Vector::new(sqrt2);
    let y = x * (Vector::erf(tmp) + Vector::one()) / Vector::new(F::new(2.0f32));
    y * Vector::new(F::new(scale))
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

// B. #[comptime] 参数：改变 launch 签名与 JIT 缓存键（与 blog 预告第四章一致，用 bool）
#[cube]
fn gelu_scalar_comptime_param<F: Float, N: Size>(
    x: Vector<F, N>,
    #[comptime] scaled: bool,
) -> Vector<F, N> {
    let sqrt2 = F::new(comptime!(2.0f32.sqrt()));
    let tmp = x / Vector::new(sqrt2);
    let mut y = x * (Vector::erf(tmp) + Vector::one()) / Vector::new(F::new(2.0f32));
    if scaled {
        y = y * Vector::new(F::new(comptime!(1.5f32)));
    }
    y
}

#[cube(launch_unchecked)]
fn gelu_array_comptime_param<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    #[comptime] scaled: bool,
) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar_comptime_param(input[ABSOLUTE_POS], scaled);
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
    println!("  comptime scale=1.5 → output={output:?}");
    println!("  launch_unchecked 签名与原始 gelu 相同（无额外 comptime 参数）。");

    println!();
    println!("=== 作业 2B：#[comptime] bool 参数 ===");
    unsafe {
        gelu_array_comptime_param::launch_unchecked::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(input.len() as u32 / vector_size as u32),
            vector_size,
            BufferArg::from_raw_parts(input_handle, input.len()),
            BufferArg::from_raw_parts(output_handle.clone(), input.len()),
            true, // scaled — 多出的 #[comptime] 参数，进入 KernelId / JIT 缓存键
        )
    };
    let bytes = client.read_one(output_handle).unwrap();
    let output = f32::from_bytes(&bytes);
    println!("  #[comptime] scaled=true → output={output:?}");
    println!("  对比 2A：launch 多传 scaled；scaled 变化 → 不同 JIT 产物。");
}

#[test]
fn homework_2_comparison() {
    println!("=== 作业 2 对比总结 ===");
    println!();
    println!("  comptime! 常量                 #[comptime] 参数");
    println!("  ─────────────────────────────  ────────────────────────────────");
    println!("  在函数体内部定义                在 launch 签名中声明");
    println!("  不改变 launch_unchecked 签名    改变 launch 签名（多一个参数）");
    println!("  不改变 JIT 缓存键              改变 JIT 缓存键（多一个维度）");
    println!("  适用：固定不变的派生值          适用：调用方指定的结构参数");
    println!("  示例：sqrt2, scale=1.5          示例：scaled: bool, blueprint");
}

// =========================================================================
// 自证测试：对应 README「动手改」作业 2（改 vector_size=2）
// 读者做完开放作业后跑 `cargo test homework_vector2_check` 验证预测。
// =========================================================================

fn launch_vector2<R: Runtime>(device: &R::Device) -> (CubeDim, Vec<f32>) {
    let client = R::client(device);
    let input = &[-1.0, 0.0, 1.0, 5.0, -2.0, 3.0, -0.5, 0.5];
    let vector_size = 2;
    // 预测：8 / 2 = 4 个 unit
    let cube_dim = CubeDim::new_1d(input.len() as u32 / vector_size as u32);

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
    let output = f32::from_bytes(&bytes).to_vec();
    (cube_dim, output)
}

#[test]
fn homework_vector2_check() {
    let device = Default::default();
    let (cube_dim, output_v2) = launch_vector2::<cubecl::cpu::CpuRuntime>(&device);

    // 预测 1：CubeDim::new_1d(4) —— 8 元素 / vector_size=2 = 4 个 unit
    assert_eq!(cube_dim.x, 4, "vector_size=2, 8 元素 → CubeDim::new_1d(4)");

    // 预测 2：数值与 vector_size=1 一致（向量化只改并行度，不改结果）
    let client = cubecl::cpu::CpuRuntime::client(&device);
    let input = &[-1.0, 0.0, 1.0, 5.0, -2.0, 3.0, -0.5, 0.5];
    let out_h = client.empty(input.len() * core::mem::size_of::<f32>());
    let in_h = client.create_from_slice(f32::as_bytes(input));
    unsafe {
        gelu_array::launch_unchecked::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(input.len() as u32),
            1,
            BufferArg::from_raw_parts(in_h, input.len()),
            BufferArg::from_raw_parts(out_h.clone(), input.len()),
        )
    };
    let output_v1 = f32::from_bytes(&client.read_one(out_h).unwrap()).to_vec();

    for (i, (a, b)) in output_v2.iter().zip(output_v1.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "idx {i}: v2={a} v1={b} 应一致");
    }
    println!("✓ vector_size=2: CubeDim.x=4, 数值与 vector_size=1 一致");
}
