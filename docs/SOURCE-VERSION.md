# 源码基准与漂移管理

## 当前基准

| 仓库 | commit | 日期 | 说明 |
|------|--------|------|------|
| burn | `cfa867f13` | 2026-06-05 | 六篇文章的源码校验基准 |
| cubecl | `ba103c7f` | 2026-06-04 | JIT 管线、autotune 和 CubeK 的源码基准 |
| burn-onnx | main | — | ONNX AOT 分析基准 |
| cubek | main | — | CubeK 分析基准 |

## 更新参考仓库

```bash
cd /path/to/tracel-study-notes
cd burn   && git pull && cd ..
cd cubecl && git pull && cd ..
cd cubek  && git pull && cd ..
```

## 文章 API 依赖矩阵

每篇文章的哪些部分依赖可能随源码版本漂移的 API：

### burn-systems-architecture.md（全景篇）

| 依赖 API | 风险 | 建议复查 |
|---------|------|----------|
| `Tensor<const D, K>` | 低 | 泛型结构稳定 |
| `Tensor::from_data(data, &device)` | 低 | 签名可能增加 `options` 参数 |
| `FusionServer` / `MultiStream::drain()` | 中 | 内部重构时签名不变但行为变化 |
| `FuseTraceLauncher::launch()` | 高 | 四个 planner 的数量和顺序可能变化 |
| `AutodiffServer::backward()` | 中 | BFS 遍历逻辑可能优化 |
| `Gradients::register()` | 低 | 累加语义稳定 |

### kernel-fusion-system-design.md（Fusion）

| 依赖 API | 风险 | 建议复查 |
|---------|------|----------|
| `OperationQueue` 字段 | 高 | 字段名和类型可能随重构变化 |
| `OperationFuser` trait (fuse/finish/properties) | 中 | trait 方法可能增加或重命名 |
| `ElementWiseFuser` / `MatmulFuser` 等 | 高 | 新 fuser 可能被添加 |
| `PersistentPool` / `SlicedPool` / `ExclusiveMemoryPool` | 低 | 内存管理基础设施稳定 |
| `ALLOC_AFTER_FREE` 常量值 | 低 | 可能调整但语义不变 |

### autotune-system-design.md（Autotune）

| 依赖 API | 风险 | 建议复查 |
|---------|------|----------|
| `TuneGroup` / `TunePlan` | 中 | 优先级签名为 `i8` 可能不变 |
| anchor 量化算法 | 低 | 数学算法稳定 |
| `AutotuneLevel` 枚举 | 中 | 级别可能增加 |
| `TuneCache` 缓存路径格式 | 高 | 路径格式已变过一次 |
| `TunableSet` 构建模式 | 低 | UI 稳定 |
| matmul 候选数量 | 高 | 新候选可能被添加 |

### jit-compilation-pipeline.md（JIT）

| 依赖 API | 风险 | 建议复查 |
|---------|------|----------|
| `#[cube]` 属性宏 | 低 | 宏签名稳定 |
| `Operation` 枚举变体 | 中 | 新操作可能增加 |
| `ConstOperandSimplify` 简化规则 | 低 | 恒等变换稳定 |
| `AutoCompiler` 枚举 | 中 | 新后端可能添加 |
| `KernelId::info` 字段 | 高 | 类型擦除机制可能调整 |
| Pipeline 缓存路径 (SPIR-V) | 高 | 路径格式可能变化 |

### autodiff-system-design.md（Autodiff）

| 依赖 API | 风险 | 建议复查 |
|---------|------|----------|
| `Autodiff<B, C>` 泛型 | 低 | 装饰器模式稳定 |
| `Backward<B, N>` trait | 中 | N 的传递方式可能变化 |
| `OpsPrep` 类型状态 | 高 | 状态转换可能增加或合并 |
| `ComputingProperty` 枚举 | 低 | 分类逻辑稳定 |
| `CheckpointStrategy` trait | 低 | 策略模式稳定 |
| `Gradients::register()` hook | 中 | hook 机制可能调整 |

### blueprint-routine-autotune.md（CubeK）

| 依赖 API | 风险 | 建议复查 |
|---------|------|----------|
| `Blueprint` trait (Hash+Eq) | 低 | trait 边界稳定 |
| `Routine` trait | 中 | 方法可能增加 |
| `Strategy` 枚举变体数量 | 高 | 新策略持续增加 |
| `TileMatmulKind` 枚举 | 中 | 新变体可能添加 |
| `MatmulAutotuneKey` 字段 | 中 | stride_factor 等可能细调 |
| `TilingScheme` | 低 | 分层块模型稳定 |

## 漂移检查清单

每次 `git pull` 参考仓库后：

1. **运行练习验证编译**：`cargo check --workspace` 在 `src/` 下
2. **按风险等级优先复查**：
   - **高风险**（字段/变体/签名变化）：对应用 `grep` 在参考仓库中搜索相关符号名，确认仍存在且语义一致
   - **中风险**（trait 方法/Ops 枚举）：复查对应文章的正文描述和代码块
   - **低风险**（基础设施/数学算法）：可信任，定期抽查
3. **更新本文的基准 commit**
4. **记录变更在 MEMORY.md 或本文件中**
