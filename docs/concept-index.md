# 概念索引

反向索引：从关键概念定位到对应的文章小节。

## 融合引擎

| 概念 | 文章 | 小节 |
|------|------|------|
| kernel launch 开销与融合收益 | [Fusion](burn/kernel-fusion-system-design.md) | §为什么需要 Kernel Fusion |
| 惰性队列 vs XLA vs Triton | [Fusion](burn/kernel-fusion-system-design.md) | §三种融合范式 |
| OperationQueue (global/relative) | [全景](burn/burn-systems-architecture.md) | §惰性执行 |
| `MultiStream::drain` | [Fusion](burn/kernel-fusion-system-design.md) | §触发点 |
| `OperationFuser` 竞标 | [全景](burn/burn-systems-architecture.md) | §融合引擎 |
| Block 划分 (tensor 依赖判断) | [全景](burn/burn-systems-architecture.md) | §Block 的划分 |
| `clone_dyn()` / `Box<dyn>` 设计原因 | [全景](burn/burn-systems-architecture.md) | §Block 与 Fuser 的关系 |
| `FuseTraceLauncher` + 四个 planner | [全景](burn/burn-systems-architecture.md) | §从融合方案到 GPU Launch |
| Page / Slice 内存模型 | [Fusion](burn/kernel-fusion-system-design.md) | §GPU 内存管理 |
| `WgpuMemManager` 三池 | [Fusion](burn/kernel-fusion-system-design.md) | §GPU 内存管理 |
| `ALLOC_AFTER_FREE` (5次) | [Fusion](burn/kernel-fusion-system-design.md) | §GPU 内存管理 |

## Autotune

| 概念 | 文章 | 小节 |
|------|------|------|
| 策略枚举 vs Triton 参数网格 | [Autotune](cubecl/autotune-system-design.md) | §CubeCL 的路 |
| matmul 30候选分类 | [Autotune](cubecl/autotune-system-design.md) | §CubeCL 的路 |
| `TuneGroup` 优先级提前终止 | [Autotune](cubecl/autotune-system-design.md) | §搜索策略 |
| 完整 walkthrough (A100, 8-15ms) | [Autotune](cubecl/autotune-system-design.md) | §搜索策略 |
| anchor 量化 (ceil to base^n) | [Autotune](cubecl/autotune-system-design.md) | §缓存密钥 |
| `AutotuneLevel` 四级 | [Autotune](cubecl/autotune-system-design.md) | §缓存密钥 |
| 评分函数 (min×0.8+median×0.2×CV) | [Autotune](cubecl/autotune-system-design.md) | §搜索策略 |
| `TuneCache` (内存+持久化 checksum) | [Autotune](cubecl/autotune-system-design.md) | §缓存架构 |
| `FusedMatmulAutotuneKey` | [Autotune](cubecl/autotune-system-design.md) | §Fusion 场景 |
| Fork context + HandleCollector | [Autotune](cubecl/autotune-system-design.md) | §Fusion 场景 |
| autotune 容错 (fallback + autotune-checks) | [Autotune](cubecl/autotune-system-design.md) | §容错 |

## JIT 编译管线

| 概念 | 文章 | 小节 |
|------|------|------|
| `#[cube]` 宏展开 | [JIT](cubecl/jit-compilation-pipeline.md) | §第一步 |
| IR Scope 树 vs CFG | [JIT](cubecl/jit-compilation-pipeline.md) | §第二步 |
| `Versioned { id, version }` SSA | [JIT](cubecl/jit-compilation-pipeline.md) | §第二步 |
| `ConstOperandSimplify` (Add(0,x)→x) | [JIT](cubecl/jit-compilation-pipeline.md) | §第三步 |
| `ConstEval` (num_traits::Float) | [JIT](cubecl/jit-compilation-pipeline.md) | §第三步 |
| `InlineAssignments` | [JIT](cubecl/jit-compilation-pipeline.md) | §第三步 |
| 优化 pass 收敛循环 | [JIT](cubecl/jit-compilation-pipeline.md) | §第三步 |
| `AutoCompiler` WGSL/SPIR-V/MSL | [JIT](cubecl/jit-compilation-pipeline.md) | §第四步 |
| WGSL 扩展 (powf/isNan/isInf) | [JIT](cubecl/jit-compilation-pipeline.md) | §第四步 |
| `KernelId` 哈希 (type+dim+comptime) | [JIT](cubecl/jit-compilation-pipeline.md) | §第五步 |
| `#[comptime]` 与 `is_const: true` | [JIT](cubecl/jit-compilation-pipeline.md) | §编译期特化 |
| `#[unroll]` 在宏层面 | [JIT](cubecl/jit-compilation-pipeline.md) | §循环展开 |
| Pipeline 缓存 vs Autotune 缓存 | [JIT](cubecl/jit-compilation-pipeline.md) | §SPIR-V 磁盘缓存 |

## Autodiff

| 概念 | 文章 | 小节 |
|------|------|------|
| `Autodiff<B, C>` 装饰器 | [Autodiff](burn/autodiff-system-design.md) | §Autodiff 在框架中的位置 |
| `Backward<B, N>` trait + `OpsPrep` | [Autodiff](burn/autodiff-system-design.md) | §图构建 |
| `Requirement::Grad / GradInBackward` | [Autodiff](burn/autodiff-system-design.md) | §图构建 |
| `ComputingProperty` (Compute/MemoryBound) | [Autodiff](burn/autodiff-system-design.md) | §检查点策略 |
| `RetroForward` + 重算 | [Autodiff](burn/autodiff-system-design.md) | §检查点策略 |
| BFS 分层 + 逆序 | [Autodiff](burn/autodiff-system-design.md) | §反向执行 |
| 分布式梯度同步 (on_register) | [Autodiff](burn/autodiff-system-design.md) | §分布式梯度同步 |
| `GraphMemoryManagement` | [Autodiff](burn/autodiff-system-design.md) | §内存管理 |

## CubeK

| 概念 | 文章 | 小节 |
|------|------|------|
| Blueprint/Routine/Autotuner 三层 | [CubeK](cubek/blueprint-routine-autotune.md) | §解决方案 |
| `Blueprint` trait (Hash+Eq) | [CubeK](cubek/blueprint-routine-autotune.md) | §第 1 层 |
| `Routine::expand_blueprint` 离散化 | [CubeK](cubek/blueprint-routine-autotune.md) | §第 2 层 |
| `Strategy` ~41变体 + Auto 级联回退 | [CubeK](cubek/blueprint-routine-autotune.md) | §第 3 层 |
| `TileMatmulKind` 五种硬件路径 | [CubeK](cubek/blueprint-routine-autotune.md) | §TileMatmulKind |
| kernel 组合爆炸 (vs CUTLASS) | [CubeK](cubek/blueprint-routine-autotune.md) | §与 CUTLASS 的对比 |
| `TilingScheme` 四层大小 | [CubeK](cubek/blueprint-routine-autotune.md) | §与 CUTLASS 的对比 |

## 架构主线

| 概念 | 文章 | 小节 |
|------|------|------|
| 类型栈 + Trait 边界 + 层间接口 | [architecture](architecture.md) | 层间接口、每层解决的问题 |
| Tracel vs PyTorch/XLA | [architecture](architecture.md) | 与 PyTorch/XLA 的架构对比 |
| 各层决策时机 | [architecture](architecture.md) | 各层决策时机 |

---

→ [阅读理解路径](../README.md#阅读路径) · [所有文章导航](../README.md)
