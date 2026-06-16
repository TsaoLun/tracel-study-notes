//! Burn Fusion 专题 · 主示例
//!
//! 三行操作（clone, *, +, tanh）生成一个融合 kernel。
//! 通过 BURN_FUSION_LOG 环境变量观察融合日志。
//!
//! ## 运行
//!
//! ```bash
//! cd src/burn-test
//! BURN_FUSION_LOG=full cargo run --release
//! ```
//!
//! ## 预期日志（burn_fusion=trace 级别）
//!
//! 你会看到：
//! - `[explorer]` 行：Explorer 探索融合机会
//! - `[stream]` 行：StreamOptimizer 注册/停止
//! - `[plan]` 行：Policy 决策（cache hit / exploration completed）
//!
//! 四个操作（Clone, ScalarMul, ScalarAdd, Tanh）被融合为一个 elemwise_fuse kernel。

use burn::prelude::*;

fn main() {
    // 初始化环境——设置 BURN_FUSION_LOG=full 即可看到 fusion 内部日志
    env_logger::init();

    // Wgpu 默认启用 fusion，等价于 Wgpu = Fusion<CubeBackend<WgpuRuntime>>
    let device = Device::wgpu(DeviceKind::DefaultDevice);

    // 创建一个 2×2 的 tensor
    let tensor_1 = Tensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device);

    // 这三行操作在 Fusion 层全部入队，不触发 GPU 执行：
    //   Clone → ScalarMul(2.0) → ScalarAdd(1.0) → Tanh
    let y = tensor_1.clone() * 2.0 + 1.0; // 入队三条 OperationIr
    let z = y.tanh(); // 入队第四条

    // println! 触发 Display → into_data → drain → 融合 → 执行
    // 四条操作融合为 一个 elemwise_fuse kernel
    println!("{z}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_example_produces_expected_shape() {
        let device = Device::wgpu(DeviceKind::DefaultDevice);

        let tensor_1 = Tensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device);
        let y = tensor_1.clone() * 2.0 + 1.0;
        let z = y.tanh();

        // tanh((2*2+1), (3*2+1), (4*2+1), (5*2+1))
        // = tanh(5, 7, 9, 11)
        let data = z.into_data();
        let expected = [
            (2.0_f32 * 2.0 + 1.0).tanh(),
            (3.0_f32 * 2.0 + 1.0).tanh(),
            (4.0_f32 * 2.0 + 1.0).tanh(),
            (5.0_f32 * 2.0 + 1.0).tanh(),
        ];
        let result: Vec<f32> = data.to_vec().unwrap();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "{r} != {e}");
        }
    }
}
