# Tracel 学习笔记

> 深入 [Tracel](https://github.com/tracel-ai) 开源生态的源码级机制分析：Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。
>
> 不是 API 教程。是在 Rust 类型系统、编译器流水线、GPU JIT 的层面解释"为什么这么设计"和"怎么做到的"。

---

## 背景：Tracel 生态

| 项目 | 一句话 | 仓库 |
|------|--------|------|
| **Burn** | Rust 深度学习框架——用 trait 嵌套替代运行时 dispatch | [tracel-ai/burn](https://github.com/tracel-ai/burn) |
| **CubeCL** | 多平台 GPU 计算编译器——`#[cube]` 写 kernel，JIT 到 CUDA/HIP/WGPU/CPU | [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) |
| **CubeK** | 基于 CubeCL 的成品算子库（matmul、attention、convolution 等） | [tracel-ai/cubek](https://github.com/tracel-ai/cubek) |
| **Burn-ONNX** | ONNX→Rust AOT 编译器——构建期把模型翻译为可调试的 Rust 源码 | [tracel-ai/burn-onnx](https://github.com/tracel-ai/burn-onnx) |

四个项目形成一条完整链路：

```
用户模型（Rust struct）
    ↓
Burn（Autodiff + Fusion + Backend trait）
Burn-ONNX（ONNX → Rust AOT 编译）
burn-cubecl → CubeK 内核
    ↓
CubeCL（#[cube] + IR + JIT + autotune）
    ↓
CUDA / HIP / WGPU / CPU …
```

---

## 阅读指南

### 主线：Burn 底层机制（3 篇）

| # | 文档 | 主题 | 读完能解释 |
|:---:|------|------|------------|
| 地图 | [blog-burn-summary.md](blog-burn-summary.md) | 类型栈 + 融合流 + 8.2× 框架开销 | `Autodiff<Fusion<CubeBackend<CudaRuntime>>>` 每一层做什么；为什么 0.21.0 拿回 8.2× |
| ONNX | [blog-burn-onnx-summary.md](blog-burn-onnx-summary.md) | ONNX→Rust AOT 编译器 | 6 阶段流水线、注意力融合、SDXL 分区编译 |
| GPU | [blog-cubecl-summary.md](blog-cubecl-summary.md) | CubeCL 编译器框架地图 | `#[cube]` 宏展开、SSA 定点循环、autotune、13 种 TileKind |

**建议按地图 → ONNX → GPU 顺序阅读**，每篇约 20–40 分钟。

### 专题：CubeCL 编译器（跟练 + 写作计划）

| # | 文档 | 主题 | 适合 |
|:---:|------|------|------|
| 计划 | [blog-cubecl-plan.md](blog-cubecl-plan.md) | 8 章写作计划 + 入门引导 | 了解专题结构、GPU 新人入门指引 |
| 1 | [blog-cubecl-1.md](blog-cubecl-1.md) | GELU 走通一条 launch | 跑通 `cargo run --example gelu --features cpu`，理解 Host 与 kernel 两层世界 |
| 2–8 | 待写 | expand、trait、comptime、拓扑、JIT、autotune、CubeK/Burn | 见 [计划表](blog-cubecl-plan.md#章节目录) |

CubeCL 专题假设你会 Rust，不要求写过 CUDA/WGSL。每章有可运行的锚点示例、源码路径、作业。

---

## 仓库结构

```
tracel-study-notes/
├── README.md                      ← 你在这里
├── .gitignore                     ← 忽略参考源码仓库
│
├── blog-burn-summary.md           ← Burn 综合地图
├── blog-burn-onnx-summary.md      ← Burn-ONNX AOT 编译器
│
├── blog-cubecl-summary.md         ← CubeCL 编译器地图
├── blog-cubecl-plan.md            ← CubeCL 专题写作计划
├── blog-cubecl-1.md               ← CubeCL 专题 1：GELU launch
│
├── burn/          (gitignored)    ← tracel-ai/burn 参考源码
├── burn-onnx/     (gitignored)    ← tracel-ai/burn-onnx 参考源码
├── cubecl/        (gitignored)    ← tracel-ai/cubecl 参考源码
└── cubek/         (gitignored)    ← tracel-ai/cubek 参考源码
```

文档中所有源码引用使用相对于各仓库根的路径（如 `crates/burn-autodiff/src/backend.rs:27`），方便在本地对照阅读。如需跟练，请将对应仓库 clone 到本目录下（确保目录名与 `.gitignore` 中的名称一致）：

```bash
git clone https://github.com/tracel-ai/burn.git
git clone https://github.com/tracel-ai/burn-onnx.git
git clone https://github.com/tracel-ai/cubecl.git
git clone https://github.com/tracel-ai/cubek.git
```

---

## 写作约定

- **源码路径写全**：`crates/burn-fusion/src/stream/multi.rs:102`，可直接跳转
- **术语首次出现括号简注**，完整释义在各文档末尾的词汇说明表
- **系列导航**：每篇末尾有完整导航表，可跳转到任意相关文档
- **章节末尾有作业**（CubeCL 专题），用于验证理解

---

## 许可

文档内容以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可发布。所引用的 Tracel 项目源码各按其自有许可证（Apache 2.0 / MIT）。
