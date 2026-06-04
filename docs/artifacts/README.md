# 编译器产物存档

本目录存放从 CubeCL/Burn 源码实际生成的编译器产物——宏展开代码、IR dump、着色器代码、日志输出。正文引用这些产物时使用相对路径（如 `../artifacts/gelu-expand.rs`）。

每个产物附带生成命令和源码版本，便于复验。

## 生成环境

| 项目 | 说明 |
|------|------|
| rustc | nightly-2026-06-03 (或标注实际版本) |
| cubecl | cubecl/ 参考仓库 git HEAD |
| burn | burn/ 参考仓库 git HEAD |
| cargo-expand | v1.0.122 |

## 文件索引

### CubeCL 宏展开

| 文件 | 内容 | 生成命令 |
|------|------|----------|
| [gelu-expand.rs](gelu-expand.rs) | `#[cube(launch_unchecked)] fn gelu_array` 的 `cargo expand` 输出 | `cargo expand --target-dir target_gelu` |
| [arith-expand.rs](arith-expand.rs) | `#[cube(launch)] fn arith_kernel` (a+b*c) 的 `cargo expand` 输出 | `cargo expand --target-dir target_arith` |

### CubeCL IR (Scope dump)

| 文件 | 内容 | 生成命令 |
|------|------|----------|
| [arith-ir.txt](arith-ir.txt) | `ArithKernel::define()` 返回的 `KernelDefinition.body` debug 打印 | `cargo test homework_2_ir_dump -- --nocapture` |

### CubeCL 生成着色器

| 文件 | 内容 | 生成命令 |
|------|------|----------|
| [gelu-wgsl.wgsl](gelu-wgsl.wgsl) | GELU kernel 经 `WgslCompiler` 编译后的 WGSL 着色器 | 从 `cubecl/cubecl-wgpu` debug 输出采集 |
| [elemwise-fuse-metal.metal](elemwise-fuse-metal.metal) | Burn Fusion 融合后的 elemwise_fuse 生成 Metal 着色器 | `RUST_LOG=wgpu_hal::metal=trace cargo run --release` |

### Burn Fusion 日志

| 文件 | 内容 | 生成命令 |
|------|------|----------|
| [fusion-trace.log](fusion-trace.log) | `RUST_LOG=burn_fusion=trace` 完整输出 | `cd src/burn-test && RUST_LOG=burn_fusion=trace cargo run --release 2>&1` |
| [fusion-trace-annotated.log](fusion-trace-annotated.log) | 同上，带行号注解 | 手工标注 |

## 更新说明

产物随参考仓库代码更新而变化。每个产物文件顶部标注：
- 生成日期
- 源码仓库 commit hash
- 生成命令

如产物与正文描述不一致，以正文引用的符号名和路径为准（产物仅作辅助验证）。
