//! Burn Autodiff 专题 · 主示例
//!
//! 三行操作（clone, *, +, tanh）在 Autodiff 层构建梯度图，
//! backward() 触发前向融合执行和反向传播。
//!
//! ## 运行
//!
//! ```bash
//! cd src/autodiff-test
//! BURN_FUSION_LOG=full cargo run --release
//! ```
//!
//! ## 预期输出
//!
//! 前向输出 z = tanh(2x+1)，梯度 ∂z/∂x = (1 - tanh²(2x+1)) × 2
//! 对 x = [[2, 3], [4, 5]]，手动计算:
//!   2x+1 = [[5, 7], [9, 11]]
//!   tanh(2x+1) ≈ [[0.9999, 1.0000], [1.0000, 1.0000]]
//!   ∂z/∂x = (1 - tanh²) × 2 ≈ [[0.0002, 0.0000], [0.0000, 0.0000]]

use burn::prelude::*;

fn main() {
    let device = Device::wgpu(DeviceKind::DefaultDevice);

    // 创建一个需要梯度的叶子张量
    let x = Tensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device).require_grad();

    // 前向计算（同时构建梯度图——MulBackward、AddBackward、TanhBackward 被注册）
    let z = (x.clone() * 2.0 + 1.0).tanh();

    // backward() 返回 Gradients 容器：
    //   1. 触前向 drain → 融合执行
    //   2. BFS 从 z 开始逆序执行反向步骤
    //   3. 梯度累积到 Gradients 容器中
    let grads = z.backward();

    let grad_x = x.grad(&grads).unwrap();
    println!("前向输出 z:\n{z}");
    println!("梯度 ∂z/∂x:\n{grad_x}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autodiff_gradient_matches_manual() {
        let device = Device::wgpu(DeviceKind::DefaultDevice);

        let x = Tensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device).require_grad();
        let z = (x.clone() * 2.0 + 1.0).tanh();
        let grads = z.backward();

        let grad = x.grad(&grads).unwrap().into_data();
        let grad_vec: Vec<f32> = grad.to_vec().unwrap();

        // 手动计算：∂z/∂x = (1 - tanh²(2x+1)) × 2
        for (i, &val) in [2.0_f32, 3.0, 4.0, 5.0].iter().enumerate() {
            let t = (val * 2.0 + 1.0).tanh();
            let expected = (1.0 - t * t) * 2.0;
            assert!(
                (grad_vec[i] - expected).abs() < 1e-5,
                "x={val}: grad={} expected={expected}", grad_vec[i]
            );
        }
    }
}
