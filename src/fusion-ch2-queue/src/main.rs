//! Burn Fusion 专题 · 第二章 · OperationQueue 主示例
//!
//! 三步操作（clone, *, +, tanh）入队但不执行；println! 触发 drain 才跑。
//! 通过 BURN_FUSION_LOG 环境变量观察入队时序与执行时序。
//!
//! ## 运行
//!
//! ```bash
//! cd src/fusion-ch2-queue
//! BURN_FUSION_LOG=full cargo run --release
//! ```
//!
//! ## 预期
//!
//! fusion execution table 在程序末尾才出现（drain 由 println! 触发）。
//! 删掉 println! 再跑：无 execution table，op 留在队列里未执行。

use burn::prelude::*;

fn main() {
    // Wgpu 默认启用 fusion，等价于 Fusion<CubeBackend<WgpuRuntime>>
    let device = Device::wgpu(DeviceKind::DefaultDevice);

    let tensor_1 = Tensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device);

    // 这三行只入队 OperationIr，不触发 GPU 执行：
    //   Clone → ScalarMul(2.0) → ScalarAdd(1.0) → Tanh
    let y = tensor_1.clone() * 2.0 + 1.0;
    let z = y.tanh();

    // println! 触发 Display → into_data → read_tensor_float → drain_stream
    // 四条 op 融合为一个 elemwise_fuse kernel
    println!("{z}");
}
