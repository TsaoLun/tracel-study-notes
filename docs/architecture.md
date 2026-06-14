# Tracel 的架构：类型栈、Trait 边界与分层组合

> Burn、CubeCL、CubeK 解决的是不同层次的问题。它们能像 `Autodiff<Fusion<CubeBackend<WgpuRuntime>>>` 一样叠加，是因为每个项目通过 trait 定义了一层清晰的边界——上层只知道下层"能做什么"（trait 方法签名），不知道"怎么做"（具体实现）。

## 本文是什么

一个跨项目架构图，解释四件事：
1. 每个项目在技术栈中的位置和它解决的系统问题
2. 层与层之间通过哪些 trait 交互——接口是什么，信息怎么传递
3. 为什么这些层可以独立演进——改 CubeK 的 tile 策略不需要改 Burn 的代码
4. 和 PyTorch/XLA 的架构差异——同样的问题，不同的分层选择

Trait 在这里的角色是定义层与层之间的合约。每层能做什么、不能做什么，由它暴露的 trait 方法签名决定。

---

## 技术栈分层

```
用户代码: model.forward(&input)
    ↓ Tensor API（matmul, relu, tanh, ...）
    ↓ BridgeTensor 按 Device 路由——用户侧不写 Backend 泛型

装饰器链（Device 类型在编译期展开，例如 Wgpu = Autodiff<Fusion<CubeBackend<...>>>）：

┌─────────────────────────────────────────────────────────┐
│ Autodiff<B>         装饰器。前向：记录梯度图后把 op 交给 │
│                      内层；.backward() 时 BFS 逆序执行   │
│   ↓                                                    │
│ Fusion<B>            装饰器。前向 op 入队；drain 时探索  │
│                      融合，缓存到 ExecutionPlanStore     │
│   ↓                                                    │
│ CubeBackend<R>       CubeCL 桥梁。注册 fuser，映射 op   │
│   ↓                                                    │
│ CubeCL Runtime       JIT #[cube] kernel → IR →          │
│                      WGSL/SPIR-V/MSL → GPU 执行          │
└─────────────────────────────────────────────────────────┘
    ↓
GPU（CUDA / Metal / Vulkan / WebGPU）

    CubeK（可选，非装饰器层）
    matmul 等 op 在 CubeBackend 内调用；Blueprint 纪律 + autotune 选策略
```

Autodiff 和 Fusion 都是装饰器——包裹内层后端，在内层操作之上附加梯度跟踪或入队融合。CubeBackend 连接 CubeCL Runtime。CubeK 不参与类型栈嵌套：CubeBackend 在特定 op 上可选调用 CubeK 实现，再经 CubeCL launch。

`Tensor` 的泛型参数是维度 `D` 和元素类型 `K`，不含 `Backend`（`burn/crates/burn-tensor/src/tensor/api/base.rs`）。Backend 组合体现在 `Device` 与内部 `BridgeTensor` 路由上——类型栈对框架内部成立，用户 API 保持无泛型传染。细节见 [全景篇 §2](burn/burn-systems-architecture.md)。

同样的 tensor 操作向下穿过这些层时，每一层只改变**执行方式**（延迟？融合？跟踪梯度？），不改变**计算结果**。

---

## 层间接口

每一层只通过 trait 看到下一层：

```
Autodiff<B>  ──Backend trait──→  Fusion<B>
    "给我做 float_matmul，我不关心你是直接执行还是排队"
    
Fusion<B>    ──FusionBackend trait──→  CubeBackend<R>
    "OperationIr 入队了，drain 时你来真正执行"
    
CubeBackend  ──FusionRuntime trait──→  CubeCL Runtime  
    "这里是四个 OperationFuser，帮我 JIT 编译和 launch"
    
CubeCL       ──ComputeServer trait──→  GPU Driver
    "编译好的 shader，帮我 dispatch"
```

关键的架构约束：**上层不能绕过 trait 直接调用下层的内部方法**。trait 表面由设计收窄；每层的 trait 方法签名定义对外能力，上层无法调用不在 trait 上的方法。以 `Autodiff<Fusion<CubeBackend<R>>>` 为例：

- **前向**：Autodiff 记录梯度图后，经 `Backend` trait 把 op 交给 `Fusion<B>` 入队；drain 时才真正执行。Autodiff 与 Fusion 在前向路径上串联。
- **反向**：Autodiff 的 `Backward` 步骤直接调内层后端的 op（`B::float_matmul` 等），不经 Fusion 的入队与融合引擎——反向 op 当前以独立 kernel 执行。两层在反向路径上解耦。
- Fusion 在前向路径上只处理 `OperationIr` 序列——不感知 autodiff 的 gradient 语义（`ad_enabled()` 对内层为 `false`，见 `burn/crates/burn-fusion/src/backend.rs`）。
- CubeCL 只接收 `KernelDefinition`（编译好的 IR），不知道 `OperationIr`，也不关心 kernel 来自融合还是单独 launch。

---

## 每层解决的问题

### Autodiff：梯度怎么算

**问题**：PyTorch 把 autograd 嵌入 tensor 运行时——每个 tensor 携带 `grad_fn` 指针，推理时也有开销。怎么让 autodiff 可选、编译期可排除？

**方案**：Autodiff 是装饰器，不是 tensor 的内置属性。`CubeBackend` 上没有 `.backward()`。`Autodiff<CubeBackend>` 上有。推理时用 `CubeBackend`，训练时用 `Autodiff<CubeBackend>`——编译期决定是否链接 autodiff crate。

**和 Fusion 的边界**：前向时 Autodiff 记录 gradient tape，再把 op 交给内层 `Fusion<B>` 入队融合。反向时各 `Backward` 步骤直接调内层后端 op，不经 Fusion 融合引擎——前向融合与反向执行因此在路径上分离。

[系统设计](burn/autodiff-system-design.md)

### Fusion：操作怎么省

**问题**：element-wise 操作密集时，kernel launch overhead 超过 compute time。但用户不会手动标注"这些 op 可以融合"。

**方案**：不立即执行——先排队，在必须拿到结果时（drain）才探索融合。OperationFuser 竞标操作序列，最优方案缓存到 ExecutionPlanStore。不需要用户标注，不需要静态图。

**和 CubeCL 的边界**：融合后产生 FuseTrace，通过 CubeCL 的 kernel launch 机制执行。Fusion 不关心 kernel 怎么编译成 GPU 指令。

[系统设计](burn/kernel-fusion-system-design.md)

### CubeCL：GPU 代码怎么生成

**问题**：一份 kernel 逻辑，要跑在 CUDA、Metal、Vulkan、WebGPU 上。平台之间 shader 语言不同，硬件指令集不同。

**方案**：`#[cube]` proc-macro 把 Rust 函数在编译期展开为 IR（嵌套 Scope 树），JIT 时翻译为 WGSL/SPIR-V/MSL。`#[comptime]` 参数在 JIT key 中哈希——不同的 comptime 值生成不同的编译产物。

**和 CubeK 的边界**：CubeCL 提供 autotune 框架（`AutotuneKey` + `TunableSet` + `TuneCache`）。CubeK 用它组织 matmul 的候选策略。CubeCL 不关心策略怎么枚举。

[系统设计](cubecl/jit-compilation-pipeline.md) | [Autotune](cubecl/autotune-system-design.md)

### CubeK：kernel 变体怎么管

**问题**：一个 matmul 有 30+ 种硬件实现策略（CMMA/Mma/Register/PlaneVec/Interleaved × 多种加载策略）。哪些进 JIT key？哪些留给 autotune？怎么防止 JIT 缓存爆炸？

**方案**：Blueprint 纪律——只把结构性选择放进 JIT key（`Hash + Eq`），Routine 在 key 生成前用离散化把连续空间映射到有限候选。Autotuner 在有限的 Strategy 枚举间选最快。

**和 CubeCL 的边界**：CubeK 的 Blueprint 哈希值进入 CubeCL 的 `KernelId`，最终触发 JIT 编译。CubeK 不接触编译管线。

[系统设计](cubek/blueprint-routine-autotune.md)

### Burn-ONNX：模型怎么导入

**问题**：ONNX Runtime 在运行时加载模型、解析 protobuf、按图执行。每一层都是运行时开销。

**方案**：`build.rs` 构建期解析 ONNX → 生成 Rust 源码 → 编译为二进制。生成的代码穿过 Burn 类型栈，享受与手写模型相同的融合和 autotune。

[详细分析](burn/onnx-summary.md)

---

## 各层决策时机

各项目用各自的术语描述「何时做选择」。下表列典型决策与触发时机，便于跨项目对照（非源码中的统一命名）：

| 层 | 典型决策 | 时机 |
|----|----------|------|
| Burn 类型栈 | 是否链接 autodiff crate、Device 对应哪种 Backend 组合 | `cargo build`（monomorphization） |
| Fusion | 连续 op 如何合并、何时 drain | 读张量 / sync 点（运行时） |
| CubeCL | `KernelId` 对应哪份 shader | 首次 launch JIT cache miss |
| CubeK | 同 Blueprint 下哪条 matmul 策略最快 | 新 `AutotuneKey` 首次 benchmark |
| Burn-ONNX | ONNX 图 → 哪份 Rust 源码 | `build.rs` 执行时 |

Fusion 的 drain 发生在运行时；Burn 的 Backend 选择在编译期——两者机制不同，不必强行归入同一时间轴。

---

## 与 PyTorch/XLA 的架构对比

| 维度 | PyTorch | XLA | Tracel |
|------|---------|-----|--------|
| **autograd 的位置** | 嵌入 tensor（`grad_fn` 指针） | 嵌入 HLO 图（反向是图变换） | 装饰器（`Autodiff<B>`），编译期可选 |
| **后端的切换** | 运行时 `tensor.to(device)` + Dispatch Key 查表 | 编译期（XLA 编译整个图） | `Device` 路由 + Backend 类型栈编译期单态化（用户侧无 Backend 泛型） |
| **算子融合** | Dynamo trace + Inductor 编译 | XLA HLO fusion pass（编译期规则） | 惰性入队 + drain 时探索 + ExecutionPlanStore 缓存 |
| **GPU 代码生成** | AOT（nvcc 预编译 CUDA kernel）+ Triton JIT | AOT（XLA → PTX/Metal） | JIT（首次 launch，`#[cube]` → IR → WGSL/SPIR-V/MSL） |
| **kernel 选择** | 手写 CUDA kernel + Triton autotune（Python 参数网格） | 后端固定实现 | autotune（策略枚举 + 优先级剪枝 + anchor 缓存） |
| **模型导入** | 运行时（ONNX Runtime / torch.onnx） | 运行时（TF Serving）+ AOT | AOT（`build.rs` 生成 Rust 源码） |

---

## 相关文档

### 系统设计文章
| 项目 | 文章 |
|------|------|
| 全栈 | [全景篇](burn/burn-systems-architecture.md) |
| Burn | [Fusion](burn/kernel-fusion-system-design.md)、[Autodiff](burn/autodiff-system-design.md) |
| CubeCL | [Autotune](cubecl/autotune-system-design.md)、[JIT 编译管线](cubecl/jit-compilation-pipeline.md) |
| CubeK | [Blueprint 纪律](cubek/blueprint-routine-autotune.md) |

### 导航与教程
| 项目 | 地图 | 专题计划 |
|------|------|----------|
| Burn | [summary.md](burn/summary.md) | [fusion/index.md](burn/fusion/index.md) |
| Burn ONNX | [onnx-summary.md](burn/onnx-summary.md) | [onnx/index.md](burn/onnx/index.md) |
| CubeCL | [summary.md](cubecl/summary.md) | [index.md](cubecl/index.md) |
| CubeK | [cubek/summary.md](cubek/summary.md) | — |

---

→ 下一篇：[全景篇](burn/burn-systems-architecture.md) — 一行代码穿行四层

[概念索引](concept-index.md) · [源码版本管理](SOURCE-VERSION.md)
