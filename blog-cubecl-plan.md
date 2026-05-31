# CubeCL 专题写作计划

> 本专题在 [CubeCL GPU 地图](blog-cubecl-summary.md)（鸟瞰：expand、SSA、autotune、CubeK）之后，按**可跟练、可对照源码**的顺序拆章。  
> **读计划前**：若你从未跑过 CubeCL，可先读 [summary 读前须知](blog-cubecl-summary.md#读前须知) 或下方「入门引导」，再打开 [第一章](blog-cubecl-1.md)。

---

## 入门引导（GPU / CubeCL 新人必读）

### 你不需要先会 CUDA

本专题假设你会 **Rust**，并愿意对照仓库读代码。不要求写过 `.cu` / WGSL。遇到「线程块、warp」等词，可先记：**CubeCL 用自家名字（Cube、Unit、Plane）统一各平台**，第四章会对照 CUDA/WebGPU。

### 本专题的「主示例」是什么？

全程用 CubeCL 仓库自带的 **`gelu` 示例**（`cubecl/examples/gelu/`）：

| 名字 | 是什么 | 在哪里 |
|------|--------|--------|
| **GELU** | 深度学习里常用的激活函数，对张量逐元素计算 \( \mathrm{GELU}(x) \) | 数学公式见下文；实现见 `lib.rs` 的 `gelu_scalar` |
| **`gelu_array`** | 带 `#[cube(launch_unchecked)]` 的 **kernel 函数名**：每个并行线程处理一个（或一组）元素 | `cubecl/examples/gelu/src/lib.rs` |
| **`gelu_scalar`** | 被 `gelu_array` 调用的子函数，算单个 `Vector` 上的 GELU | 同文件，`#[cube]` 无 launch |
| **`gelu::launch`** | **Host 代码**：分配 buffer、调用 `gelu_array::launch_unchecked`、读回结果 | 同文件 `pub fn launch` |
| **`examples/gelu.rs`** | 程序入口 `main`，按 Cargo feature 选 CPU/CUDA/WGPU runtime | `cubecl/examples/gelu/examples/gelu.rs` |

宏展开后还会出现 **`GeluArray` 结构体**（PascalCase），实现 `CubeKernel` trait；`launch_unchecked` 创建它并交给 runtime，**JIT 编译时在 `define()` 里才调用 `expand`**。第一章会画完整调用链。

### 建议阅读顺序

1. 可选：扫一眼 [summary 词汇表 · 核心概念](blog-cubecl-summary.md#核心概念本篇最重要)（5 分钟）。
2. **第一章**：跟跑 `cargo run --example gelu --features cpu`（无 GPU 也可）。
3. 并行参考：[cubecl-book · Installation + Simple Reduction](cubecl/cubecl-book/src/getting-started/summary.md)（练手写 reduction，与本专题互补）。
4. 需要全貌时再读 [blog-cubecl-summary.md](blog-cubecl-summary.md)。

### 三份材料如何分工

| 材料 | 角色 |
|------|------|
| [blog-cubecl-summary.md](blog-cubecl-summary.md) | **地图**：文首 [读前须知](blog-cubecl-summary.md#读前须知) + 机制全览 + 文末术语表 |
| [cubecl-book](cubecl/cubecl-book/src/SUMMARY.md) | **练手写 kernel**（reduction 渐进教程） |
| **本专题** | **对照源码走编译器路径**（launch → expand → opt → 后端） |

---

## 定位与读者

**目标读者**：会用 Rust；未必写过 GPU；未必读过 Burn 系列（有则更好）。  
**不覆盖**：Burn Fusion 调度细节（见 [blog-burn-summary.md](blog-burn-summary.md) 第五节）。

---

## 写作约定

1. **每章开头**：用 2–3 句话说明「本章锚点示例是什么、读完能干什么」，不假设读者已懂函数名。
2. **每章一个主示例**，源码路径写全（相对 `cubecl/` 仓库根）。
3. **正文先可运行、再钉源码**；编译器深度逐章加码。
4. **章末**：小结 + 作业 + 下章预告。
5. **完整术语表**仍以 [summary 文末](blog-cubecl-summary.md#词汇说明表) 为准；各章只引入本章最少新词。

---

## 章节目录

| 章 | 文件 | 标题 | 读完能做什么 | 核心源码锚点 |
|:---:|------|------|--------------|--------------|
| 1 | [blog-cubecl-1.md](blog-cubecl-1.md) | 用 GELU 走通一条 launch | Host launch、`GeluArray::define`、`KernelBuilder.scope` | `examples/gelu/`、`launch.rs`、`kernel.rs`（`define_body`）、`compute/builder.rs` |
| 2 | [blog-cubecl-2.md](blog-cubecl-2.md) | expand：`+` → `__expand_*_method` → IR | 理解 **NativeExpand 间接层**，而非「表达式直连 Operation」 | `generate/expression.rs`、`generate/kernel.rs` |
| 3 | blog-cubecl-3.md | trait / impl 与 `#[define]` | `Float` 泛型 kernel、`__expand_{method}`、CubeK 常见签名 | `parse/cube_trait.rs`、`parse/cube_impl.rs`、`generate/cube_trait.rs` |
| 4 | blog-cubecl-4.md | comptime 与 JIT 缓存键 | `#[comptime]`、`comptime!`、多份 JIT 产物 | `parse/kernel.rs`、`parse/statement.rs`、`generate/kernel.rs`（`KernelId`）；参考 book `core-features/comptime.md` |
| 5 | blog-cubecl-5.md | 拓扑与四轴 | `ABSOLUTE_POS`、`PLANE_DIM`、launch 与硬件映射 | `frontend/topology.rs`、`cubecl-cpp/.../kernel.rs` |
| 6 | blog-cubecl-6.md | JIT 管线：Scope → PTX/WGSL | SSA、**post-SSA 定点循环**、后端 | `cubecl-opt/src/lib.rs`（`apply_post_ssa_passes` ~605–615） |
| 7 | blog-cubecl-7.md | vectorization 与 autotune（两节） | 区分 launch 时 JIT 键 vs 首次执行 benchmark | **§7.1** `gelu` `vector_size`、`core-features/vectorization.md`；**§7.2** `cubecl-runtime/.../tune/`、CubeK `TileKind` |
| 8 | blog-cubecl-8.md | CubeK 纪律与 Burn 边界 | Blueprint 纪律 + Burn 如何落到 CubeCL | cubek `GUIDE.md`；`burn-cubecl`、`Backend` trait、Fusion 与 JIT 分界 |

> **相对原 7 章计划的变化**：新增 **第 3 章 trait/impl**（评估指出缺口）；原第 6 章拆为 **两节** 仍在一篇内；原第 7 章扩 checklist 并改为第 8 章。

---

## 各章要点（写作 checklist）

### 第一章（已写，需与源码对齐）

- [x] 章首说明 GELU / `gelu_array` / 示例路径（见 [blog-cubecl-1.md](blog-cubecl-1.md)）
- [x] 分别用 `cargo run --example gelu --features cpu|cuda|wgpu` 跑通（**三次编译**，非一次跑三后端）
- [x] `ABSOLUTE_POS` + `vector_size` / `CubeDim` 算术（`CubeDim::new_1d(1)` 当 len=4、vector_size=4）
- [x] `mod gelu_array` 含 **`GeluArray` + `CubeKernel::define`**
- [x] 调用链：`launch_unchecked` → `KernelLauncher` → `client.launch` → `compile` → `define()` → `expand` + `build`
- [x] 区分 `KernelDefinition`（`build`）与 **cubecl-opt + 后端 codegen**（真正「编译」）
- [x] 预告 `#[define]`（第二章/第三章）

### 第二章（已写）

- [x] `IntoExpand::into_expand` → `__expand_add_method` 等（`expression.rs` ~63）
- [x] 方法内部才向 `Scope` 注册 `Operation`（**两层，非直连**）
- [x] `if` / 短路 `&&` `||` 的 expand 路径各一例
- [x] `ArithKernel::define()` 只看 Scope（不 launch）

### 第三章（待写，新增）

- `#[cube]` on trait：`__expand_{method}` 命名（`cube_trait.rs`）
- `impl CubeType for Float` 一类模式为何能写 `F: Float`
- `#[define(Lhs, Rhs)]` 与 launch 泛型注册（对照 cubek `matmul_entry` 签名，不必全文展开）

### 第四章（待写）

- Rust `const` vs `#[comptime]` vs runtime `BufferArg`
- `sum_plane`：`plane: bool` → 两份 JIT 产物
- `KernelMetadata::id` / `KernelId` 字段来源（`generate/kernel.rs`）

### 第五章（待写）

- 四轴表 + 平台 builtins 对照
- `ABSOLUTE_POS` 后端合成跟读 `cubecl-cpp`

### 第六章（待写）

- `Function::run_opt()` 总流程 + 行号
- **post-SSA：`loop { 10 passes; break if AtomicCounter==0 }`**（`lib.rs` 605–615），强调 **iterate until quiescence**
- GVN / 强度削减后为何可能 **再跑一轮** post-SSA

### 第七章（待写，一章两节）

**§7.1 Vectorization**

- launch 参数 `vector_size` → `KernelSettings` → JIT 键
- gelu 改 vector_size 实验

**§7.2 Autotune**

- `Tuner`、`AutotuneKey`、anchor 分桶
- 与 §7.1 **对比表**（时机、缓存、失败形态）
- CubeK `TileKind` 仅点到为止

### 第八章（待写）

- kernel explosion 算术（3×3×3 × TileKind）
- Burn 调用链：`Backend` → `burn-cubecl` → `ComputeClient` → CubeK kernel launch
- **Fusion 调度 vs CubeCL JIT** 边界（引用 [blog-burn-summary.md](blog-burn-summary.md) 第五节）
- 何时直接用 CubeCL vs 用 Burn + CubeK

---

## 对外部评估的采纳说明

| 评估点 | 是否采纳 | 处理 |
|--------|----------|------|
| plan 缺少新人引导 | ✅ | 增加「入门引导」整节 |
| 第二章标题过度简化 | ✅ | 改为 `__expand_*_method` + NativeExpand |
| 第五章缺 post-SSA 定点循环 | ✅ | checklist + 第六章标题强调 |
| 缺 trait/impl 章 | ✅ | 新增第 3 章 |
| 第六章 vectorization/autotune 过重 | ✅ | 一章两节 + 对比表 |
| 第三章源码锚点不准 | ✅ | 改为 macros 源码路径 + book 作参考 |
| 第七章 checklist 不足 | ✅ | 第八章 checklist 扩写 |
| ch1 `CubeDim::new_1d(4)` 错误 | ✅ | 改正为 `new_1d(1)` |
| ch1 `new()` 内部调用 expand | ✅ | 改为 `define()` 在 compile 路径 |
| ch1 缺 `GeluArray` / `CubeKernel` | ✅ | 补全宏产物与调用链 |
| ch1 mermaid 过于直连 | ✅ | 改为间接触发 + 分阶段 |
| ch1 build vs compile 混淆 | ✅ | 五步时间线拆开 |
| checklist「三 feature 一次跑通」 | ✅ | 改为三次分别编译 |
| 缺 `#[define]` 预告 | ✅ | 第一章误区/下章预告 |

---

## 进度

| 状态 | 文档 |
|------|------|
| ✅ 已更新 | `blog-cubecl-plan.md`（本文件）、`blog-cubecl-1.md`（修订）、`blog-cubecl-2.md` |
| 📋 待写 | `blog-cubecl-3.md` … `blog-cubecl-8.md` |
| 📎 地图 | `blog-cubecl-summary.md` |

---

## 系列导航（专题内）

| 篇 | 文档 | 状态 |
|:---:|------|------|
| 地图 | [blog-cubecl-summary.md](blog-cubecl-summary.md) | 已发布 |
| 计划 | **本文** | 已更新 |
| 专题 1 | [blog-cubecl-1.md](blog-cubecl-1.md) | 已修订 |
| 专题 2 | [blog-cubecl-2.md](blog-cubecl-2.md) | 已发布 |
| 专题 3–8 | `blog-cubecl-3.md` … | 待写 |

*Burn 底层机制 · CubeCL 专题 · [系列索引](README.md)*
