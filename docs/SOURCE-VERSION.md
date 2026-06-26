# 源码基准与漂移管理

## 当前基准

| 仓库 | commit | 日期 | 说明 |
|------|--------|------|------|
| burn | `78f10aec1` | 2026-06-10 | 六篇文章的源码校验基准 |
| cubecl | `35b861d0` | 2026-06-12 | JIT 管线、autotune 和 CubeK 的源码基准 |
| burn-onnx | `846b2452` | 2026-06-11 | ONNX AOT 分析基准 |
| cubek | `4ccfc4f2` | 2026-06-16 | CubeK 分析基准 |

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
| `Strategy` 枚举变体数量 | 高 | 新策略持续增加（当前 ~41 变体） |
| `TileMatmulKind` 枚举 | 中 | 5 种变体稳定 |
| `MatmulAutotuneKey` 字段 | 中 | stride_factor 等可能细调 |
| `TilingScheme` | 低 | 分层块模型稳定 |
| `Tile` 结构体 (`cubek-tile`) | 中 | 内部字段可能重构（如 check 移到 MemData） |
| `TileKind` 枚举 (`cubek-std`) | 低 | 13 种计算变体稳定 |

## 漂移检查清单

每次 `git pull` 参考仓库后：

1. **运行练习验证编译**：`cargo check --workspace` 在 `src/` 下
2. **按风险等级优先复查**：
   - **高风险**（字段/变体/签名变化）：对应用 `grep` 在参考仓库中搜索相关符号名，确认仍存在且语义一致
   - **中风险**（trait 方法/Ops 枚举）：复查对应文章的正文描述和代码块
   - **低风险**（基础设施/数学算法）：可信任，定期抽查
3. **更新本文的基准 commit**
4. **记录变更在 MEMORY.md 或本文件中**

## 已知漂移

### 2026-06-12: cubecl `35b861d0` — `Variable` → `Value` 重构

CubeCL commit `35b861d0` (`refactor: Simplify Variable to align it with existing IRs`) 将 `Variable` 重命名为 `Value`，`VariableKind` 重命名为 `ValueKind`，后者从 15+ 变体简化为 2 个变体（`Value { id }` 和 `Constant`）。

**影响文章**：`jit-compilation-pipeline.md` 中讨论 `Variable` 和 `VariableKind` 的段落已更新命名（`Value`/`ValueKind`），`burn-systems-architecture.md` 中引用的 Scope 结构与 `ABSOLUTE_POS` 内建说明同步更新。概念性描述（SSA 版本控制、内建变量、常量）保持准确。

**状态**：✅ 已完成

### 2026-06-10: burn `78f10aec1` — Autodiff 改为 device 显式属性

Burn `78f10aec1` 起 autodiff 不再默认开启：`Device::wgpu(...)` 返回非 autodiff device，在其上创建的 tensor 调 `backward()` 会 panic "Requires autodiff tensor"。需显式 `Device::wgpu(...).autodiff()`（`burn-tensor/src/device.rs:428`）把 device 包成 `DispatchDevice::Autodiff`。

**影响**：`src/autodiff-test/` 的 `main.rs` 与测试均需在 device 构造处加 `.autodiff()`。已修。

**docs 旧心智模型**：多处文章仍按"编译期类型差异决定是否 autodiff"描述，与 `78f10aec1` 后的"device 运行时路由 + cargo feature 编译期链接"双层模型矛盾。已同步更新：`burn/burn-systems-architecture.md`（开篇示例补 `.autodiff()`、架构图区分推理/训练栈、对比表三行、全链路回顾代码补 device 构造与 `grads`）、`architecture.md`（开篇、L31 类型别名、§Autodiff、决策时机表、对比表）、`burn/autodiff-system-design.md`（开篇、位置段、动手 callout、谁该用哪个、小结）、`README.md`、`docs/primer.md`、`docs/concept-index.md`。

**状态**：✅ 已完成

### 2026-06-16: cubek `4ccfc4f2` — 模块重组、Strategy 扩展与 cfft 公开

CubeK `c6a0bf40` → `4ccfc4f2` 之间的 4 个提交包含了以下变化：

1. **模块重组** (`51fda58b`)：`cubek-matmul/src/launch/` 目录不再存在，`strategy.rs` 和 `tune_key.rs` 移入 `cubek-matmul/src/strategy/`。旧的 `launch/strategy.rs` 已改为 `strategy/strategy.rs`。`cubek-tile` 中 `Tile.check` 字段移入 `MemData.check`，`register.rs` 和 `schedule.rs` 从 `matmul/mod.rs` 中独立。
2. **Strategy 变体增加** (`51fda58b`)：从 ~34 个变体增加到 41 个。新增 `SimpleVecMat`/`DoubleVecMat`（VecMat inner product）、`GemvUnitPerpendicular`（GEMV）、`Gemm`/`CpuGemm`（CPU GEMM）等路径。
3. **cfft 公开** (`b0d8226c`)：`cubek-fft` 新增公开的 complex FFT（cfft）模块，原有的 rfft/irfft 现在与 cfft 并列。
4. **Reduce shared memory 限制** (`f44bd68d`)：`cubek-reduce` 的 Cube routine 现在对 ArgTopK/TopK 推断的 reduce width 做了 shared memory 上限钳制。

**影响文章**：
- `blueprint-routine-autotune.md`：Strategy 变体数量、源码路径（launch/ → strategy/）、autotune key 路径 —— **已更新**
- `summary.md`：架构一览中 FFT 描述 —— **已更新**

**状态**：✅ 已完成
