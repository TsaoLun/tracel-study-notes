//! Burn Fusion 专题 · 第二章 · OperationQueue 练习
//!
//! 对应文档：docs/burn/fusion/2-operation-queue.md
//! 主题：OperationQueue —— 惰性执行与"推迟了什么"
//!
//! 主示例见 `src/main.rs`。本模块为「动手改」作业的自证测试。

#![cfg(test)]

use burn::prelude::*;

/// 基线：三步操作融合后数值正确。
/// 证明入队 + drain 路径完整可跑。
#[test]
fn fusion_example_correct() {
    let device = Device::wgpu(DeviceKind::DefaultDevice);
    let tensor_1 = Tensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device);
    let y = tensor_1.clone() * 2.0 + 1.0;
    let z = y.tanh();

    // tanh(5, 7, 9, 11)
    let data = z.into_data();
    let expected = [
        (2.0_f32 * 2.0 + 1.0).tanh(),
        (3.0_f32 * 2.0 + 1.0).tanh(),
        (4.0_f32 * 2.0 + 1.0).tanh(),
        (5.0_f32 * 2.0 + 1.0).tanh(),
    ];
    let result: Vec<f32> = data.to_vec().unwrap();
    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "{r} != {e}");
    }
}

/// 自证测试：作业 3 —— `into_data()`（不打印）同样触发 drain。
/// 弱验证：数值正确即证明 drain 发生（否则队列没跑、读不出值）。
#[test]
fn into_data_triggers_drain() {
    let device = Device::wgpu(DeviceKind::DefaultDevice);
    let tensor_1 = Tensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device);
    let y = tensor_1.clone() * 2.0 + 1.0;
    let z = y.tanh();

    // 只读不打印 —— into_data() 同样走 read_tensor_float → drain_stream
    let data = z.into_data();
    let result: Vec<f32> = data.to_vec().unwrap();
    let expected = [
        (2.0_f32 * 2.0 + 1.0).tanh(),
        (3.0_f32 * 2.0 + 1.0).tanh(),
        (4.0_f32 * 2.0 + 1.0).tanh(),
        (5.0_f32 * 2.0 + 1.0).tanh(),
    ];
    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "{r} != {e}");
    }
    println!("✓ into_data() 触发 drain，数值与 println! 路径一致");
}
