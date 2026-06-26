# CubeCL 的 JIT 编译管线：宏展开、IR 设计与多平台代码生成

> `#[cube]` 过程宏将 Rust 函数展开为 IR 操作 → 嵌套 Scope 树做定点优化 → WGSL/SPIR-V/MSL 三后端生成代码。与 Triton 的运行时 JIT 不同，CubeCL 的 comptime 特化在 Rust crate 编译期完成，GPU 编译在首次 launch 时触发并缓存。

> **导读** · 难度：中等偏难 · 预计 ~90 分钟 + 两个练习 · [学习地图](../../README.md#学习地图) 阶段 4
>
> - **读前应知道**：GPU 执行模型；Rust proc-macro 不需要预先掌握，文章逐步解释 `#[cube]`
> - **AI infra 通用映射**：把 kernel 源码编译成多平台 GPU 二进制是通用问题，对比 Triton 的 JIT→PTX 与 XLA 的 HLO→PTX（基线见 [primer · Part B](../primer.md#part-b--对比基线速查)）。
> - **本篇回答**：(1) `#[cube]` 如何在编译期展开成 IR；(2) IR 如何做定点优化；(3) 如何翻译为多平台 shader 并 dispatch 到 GPU
> - **配套练习**：[src/ch1-gelu-variants](../../src/ch1-gelu-variants/)（先，写 kernel 变体）、[src/ch2-expand-study](../../src/ch2-expand-study/)（后，看宏展开）

## 问题：什么构成了一个 GPU 编译器

GPU 编程的传统方式：用 CUDA C++ 写 kernel → nvcc 编译 → 加载 PTX → 启动。但 CubeCL 的目标是将 **Rust 函数直接变成 GPU 可执行代码**，且同时支持 CUDA、Metal、Vulkan、WebGPU。这需要一套完整的编译管线：

1. 把 Rust 的 `#[cube]` 标注函数转成 IR
2. 对 IR 做优化（常量折叠、死代码消除、循环展开）
3. 将优化后的 IR 翻译为平台特定着色器语言（WGSL、SPIR-V、MSL）
4. 编译着色器、创建 compute pipeline、绑定资源、dispatch

这篇文章展开每一步的系统设计。

### 为什么分两段：comptime + launch-time

CubeCL 把编译拆成两段——comptime（crate 编译期）和 launch-time（首次 GPU 调用时）——这不是偶然的。和 Triton 的单段 JIT 对比能看清这个设计的约束来源：

**Triton 的路**：Python 函数 → Triton IR → LLVM IR → PTX。整个编译链路在运行时一次性走完。优点是用户在 Jupyter 里改一行 Python 就能重编译 kernel，迭代极快。代价是首次调用延迟——复杂 kernel 的编译加 autotune 可达 100-500ms。

**CubeCL 的路**：`#[cube]` 宏展开（comptime，`cargo build` 时）→ `expand()` 生成 cubecl_ir（comptime）→ IR 优化（comptime）→ GPU 代码生成（launch-time，首次调用时，~10-50ms）→ GPU 驱动编译 + 缓存（launch-time）。

两段分离的后果：

- **comptime 阶段在 Rust 编译器眼皮底下完成**。宏展开错误、类型不匹配、IR 生成错误——这些在 `cargo check` 时就能发现，不需要等到 GPU 运行时。Triton 的 Python 宏在运行时才展开，kernel 语法错误只能在首次 launch 时暴露。
- **launch-time 只做平台相关的代码生成和 GPU 编译**。这部分无法在 comptime 完成——不知道目标 GPU 是 A100 还是 M2，不知道驱动版本。但工作量已大幅缩减：IR 已优化完毕，只需做目标代码翻译。
- **首次调用延迟比 Triton 低一个数量级**（~10-50ms vs ~100-500ms），因为 comptime 承担了大部分编译工作。这对生产部署有意义——用户不会因为冷启动等半秒。
- **代价是开发迭代比 Triton 慢**。改一行 kernel 代码需要 `cargo build`（秒级），而不是 Jupyter cell 重跑（毫秒级）。在探索阶段，Triton 的体验更流畅。

选择两段分离不是因为"这样更好"——是因为 CubeCL 的目标场景（生产部署、多平台 AOT）和 Triton 的目标场景（研究探索、单平台 JIT）有根本不同的约束。

---

## 第一步：`#[cube]` 过程宏 —— 从 Rust 到 IR

### 宏的输入和输出

一个标记了 `#[cube]` 的 Rust 函数：

```rust
#[cube(launch_unchecked)]
fn add_kernel(lhs: &Tensor<f32>, rhs: &Tensor<f32>, out: &mut Tensor<f32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
    }
}
```

`#[cube]` 的属性宏（`cubecl/crates/cubecl-macros/src/lib.rs:56`）做三件事：

1. **保留原函数不变**（作些 AST 变换：移除辅助函数、替换 `define!` 宏）
2. **生成一个子模块** `<function_name>`，内含：
   - `expand()` 函数——原函数的"IR 化"版本，每次调用 `register()` 生成一条 `cubecl_ir::Instruction`
   - `KernelName` 结构体——存储 kernel 元数据（buffer 布局、标量参数、cube_dim）
   - `launch()` / `launch_unchecked()` 包装函数——连接 `KernelBuilder` → `KernelLauncher` → `ComputeClient::launch`

### 表达式到 IR 的映射

`#[cube]` 宏的核心工作是将 Rust 表达式转换为 `cubecl_ir::Operation`（`cubecl/crates/cubecl-macros/src/parse/expression.rs`）：

```
Rust:  out[pos] = lhs[pos] + rhs[pos]
  ↓ 宏解析每个表达式
IR:    Index(lhs, pos)     →  Value #1  (Memory::Index)
       Index(rhs, pos)     →  Value #2
       Add(#1, #2)         →  Value #3  (Arithmetic::Add)
       Assign(#3, out@pos) →  Memory::Store
```

Rust 的操作符（`+`、`*`、`/`、`%`、`>`、`==` 等）被一对一映射为 IR 的操作变体。`ABSOLUTE_POS` 成为一个内建值（`Builtin::AbsolutePos`，`Builtin` 是独立枚举）。`out.len()` 成为一次 `Metadata::BufferLength` 调用。`if` 成为 `Branch::If`，携带一个嵌套的 `Scope`。

这个映射的目的是产生一个**可跨平台编译的中间表示**——不同于 LLVM IR 或 SPIR-V，CubeCL 的 IR 专为 GPU compute 设计，包含共享内存、workgroup 同步、协同矩阵乘累加（CMMA）、张量内存加速器（TMA）等 GPU 原语。

---

## 第二步：IR 设计 —— 嵌套 Scope 树

### 为什么不是基本块 CFG

传统编译器 IR（LLVM IR、SPIR-V）使用基本块 + 控制流图（CFG）。CubeCL 的 IR 使用**嵌套 Scope 树**：

```rust
// cubecl/crates/cubecl-ir/src/scope.rs:34
pub struct Scope {
    pub instructions: RefCell<Vec<Instruction>>,
    pub return_value: Option<Value>,
    pub locals: RefCell<Vec<Value>>,
    pub global_state: GlobalState,
}
```

每个 `Branch`（`if`、`loop`、`for`）携带自己的子 `Scope`。这形成一棵树而非图。选择树形表示的代价是**没有 phi 节点，无法表达复杂控制流合并**；收益是**编译简单**——每个后端编译器递归遍历 Scope 树即可生成正确的嵌套着色器代码，无需处理 CFG 的支配树和 SSA 重建。

### 值系统：Versioned SSA

cubecl `35b861d0`（"Simplify Variable to align it with existing IRs"）把旧的 15+ 变体的 `VariableKind` 简化为两个变体：

```rust
// cubecl/crates/cubecl-ir/src/variable.rs:65
pub enum ValueKind {
    Value { id: Id },              // 一个带 id 的值（统一表达 SSA 值、局部变量、全局缓冲等）
    Constant(ConstantValue),       // 编译期常量（#[comptime] 值在 IR 中的体现）
}

pub struct Value {
    pub kind: ValueKind,
    pub ty: Type,                  // 类型携带语义：指针/标量/向量、地址空间等
}
```

这次重构把原先散在 `VariableKind` 各变体里的信息收拢：缓冲区/局部/共享/版本化等区分移到 `Type` 与 `id` 空间，内建变量（`UnitPos`、`CubeDim`、`AbsolutePos` 等）独立成 `Builtin` 枚举，由 `Type` 标记内建。概念不变——每次赋值产生新 `id` 实现 SSA 式版本控制，旧版本不可修改；从未重新绑定的变量对应常量或不可变值；`Constant` 仍是 `#[comptime]` 值在 IR 中的体现。命名从 `Variable`/`VariableKind` 改为 `Value`/`ValueKind` 与主流 IR（LLVM `Value`、MLIR `Value`）对齐。

### 操作全集

```rust
// cubecl/crates/cubecl-ir/src/operation.rs
pub enum Operation {
    Copy(Value),
    Memory(Memory),          // Load/Store/Index
    Arithmetic(Arithmetic),  // Add/Mul/Fma/Sin/Cos/Exp/Log/...
    Comparison(Comparison),  // Eq/Lt/Gt/IsNan/IsInf
    Bitwise(Bitwise),        // And/Or/Xor/Shift/Not
    Operator(Operator),      // Cast/Select/Swizzle/Reinterpret
    Branch(Branch),          // If/IfElse/Loop/RangeLoop/Switch/Return/Break
    Synchronization(Synchronization), // sync_cube/sync_plane
    Plane(Plane),            // subgroup broadcast/shuffle/sum/min/max
    CoopMma(CoopMma),        // 协同矩阵乘累加（tensor core 抽象）
    Tma(TmaOps),             // 张量内存加速器（H100 特性）
    Barrier(BarrierOps),     // 内存屏障
    Metadata(Metadata),      // Stride/Shape/BufferLen
    // ...
}
```

这个操作集合的覆盖率是关键的设计权衡。过于丰富（每个平台特性都是独立操作）→ 编译器复杂度爆炸。过于贫乏（只有通用操作）→ 无法利用特定硬件加速（Tensor Core、TMA、subgroup shuffle）。CubeCL 的选择是**比 SPIR-V 更贴近 GPU 语义，比 WGSL 更丰富**——每一类硬件特性都有一组操作，但不是每个平台变体都有独立操作。

---

## 第三步：IR 优化

在生成目标代码之前，CubeCL 对 Scope 树应用一系列优化 pass（`cubecl/crates/cubecl-core/src/post_processing/mod.rs:27`）：

1. **`ConstOperandSimplify`**（`post_processing/constant_prop.rs:24`）：半常量化简，处理如 `Add(0, x)` → `x`、`Mul(x, 1)` → `x`、`Mul(x, 0)` → `0`、`Div(x, 1)` → `x`，以及布尔短路（`true || x` → `true`）。这不是简单的"两个常量相加"——它消除的是**一边为常量的无用计算**，在融合 kernel 中尤为重要（融合的标量乘法 `x * 1.0` 会被直接移除）。

2. **`ConstEval`**（`post_processing/constant_prop.rs:131`）：真正的常量求值。`Add(Constant(1.0), Constant(2.0))` → `Constant(3.0)`。支持三角函数、指数、对数——所有求值在编译器的 Rust 代码中用 `num_traits::Float` 完成，不引入 GPU 指令——求值在 host 侧完成。

3. **`InlineAssignments`**（`post_processing/expression_merge.rs:13`）：建立替换表。当看到 `Copy(input)` 且输入和输出的类型匹配时，记录 `{out → input}`，后续所有使用 `out` 处替换为 `input`。`x = y; z = x + 1` 变为 `z = y + 1`。

4. **死代码消除**：前几步产生的不再被引用的变量被移除。

5. 以上四步在 `optimize_scope()` 的 `loop` 中反复运行直到收敛——常量折叠可能打开内联机会，内联又可能打开新的常量折叠。

> ▶ **动手**：`cd src/ch1-gelu-variants && cargo test -- --nocapture`
> 写 GELU kernel 的三种变体（标量、vec2、vec4）。每种变体对应不同的 `CubeDim` 和向量化参数，在 GPU 上跑通后再回来看 IR 如何生成——先建立"我能写一个 kernel"的直觉。

还有后端特定的 pass。WGSL 编译器（`cubecl/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs:123`）在生成代码前运行：

- `CheckedIoVisitor` —— 为矢量化访问插入边界检查
- `DisaggregateVisitor` —— 将胖指针（`Tensor` 参数包含 data + shape + stride）拆分为基本分量，使后端能处理
- `UnrollVisitor` —— 再次展开循环，限制最大向量大小

---

## 第四步：多平台代码生成

### AutoCompiler 运行时选择

```
WgpuAdapter::initialize()
  ↓ backend = wgpu::Adapter::get_info().backend
  ↓ 根据 backend 选择：
  │   Vulkan + spirv feature + 设备支持 → Spirv(SpirvCompiler)
  │   Metal + msl feature              → Msl(MslCompiler)
  │   否则                              → Wgsl(WgslCompiler)
```

`AutoCompiler` 枚举（`cubecl/crates/cubecl-wgpu/src/compiler/base.rs:35`）在 `WgpuServer` 初始化时解析，后续所有 kernel 编译都通过同一个编译器实例。

### WGSL 编译器：一对一翻译

WGSL 编译器（`cubecl/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs`）的工作是**一对一地将 IR 操作翻译为 WGSL 代码**：

```
IR:              WGSL:
Add(#1, #2)  →  (var_1 + var_2)
Index(buf, i)→  buf.elements[i]
sync_cube()  →  workgroupBarrier()
sin(x)       →  sin(x)  // WGSL 原生支持
tanh(x)      →  tanh_extension(x)  // WGSL 无原生 tanh，插入扩展函数
```

WGSL 原生不支持的数学函数（`tanh`、`powf`、`isNan` 等）在 `register_extensions` 中作为完整的 WGSL 函数注入（`compiler.rs:1216`）。注入的函数参数化为元素类型和向量大小——`safe_tanh_1_f32`、`safe_tanh_4_f32` 等按需生成。

例如 `powf` 的 WGSL 扩展（`compiler/wgsl/extension.rs:241`）处理了 `pow()` 对负底数未定义行为的问题：

```wgsl
fn powf_primitive_f32(lhs: f32, rhs: f32) -> f32 {
    if rhs == 0.0 { return 1.0; }           // 指数 0
    let even = rhs % 2.0 == 0.0;
    if even { return pow(abs(lhs), rhs); }   // 偶指数：取绝对值
    return -pow(-lhs, rhs);                  // 奇指数：取负绝对值
}
```

`isNan` 和 `isInf` 通过 IEEE 754 位操作实现——WGSL 没有原生的 NaN/Inf 检测。这些扩展函数在 `ComputeShader` 格式化时追加在 `fn main` 之后（`shader.rs:213`），WGSL 语法允许这样的后置定义。

### SPIR-V 和 MSL：完整的后端编译，绕过 wgpu 的 WGSL 验证

一个常见的误解是 SPIR-V 路径"绕过编译"。实际上 `cubecl-spirv` crate（`cubecl/crates/cubecl-spirv/src/compiler.rs:144`）实现了一个**完整的 `Compiler` trait 后端**——它将 CubeCL IR 翻译为 SPIR-V 二进制，运行与 WGSL 相同的优化 pass（`CheckedIoVisitor`、`DisaggregateVisitor`、`UnrollVisitor`），外加 SPIR-V 专用的变换（`ErfTransform`、`BitwiseTransform`）。

区别在于 wgpu 的使用方式：编译后的 SPIR-V 二进制通过 `create_shader_module_passthrough`（`cubecl/crates/cubecl-wgpu/src/backend/base.rs:79`）直接传给 Vulkan 驱动，跳过 wgpu 内部的 WGSL 编译和验证。编译后的二进制还被缓存在磁盘上，key 为 `(properties_hash, kernel_id.stable_hash())`，驱动更新通过 `properties_hash` 自动触发缓存失效。

对于 MSL 路径（Metal），`cubecl-cpp` crate 提供类似的能力——将 IR 编译为 Metal Shading Language 源代码，通过 passthrough 提交。

这解释了为什么 WGSL 是"始终可用的兜底"——它只需要 wgpu 运行时，不依赖额外的编译 crate（`cubecl-spirv`、`cubecl-cpp` 都是 optional feature）。SPIR-V/MSL 路径提供更快的编译（跳过 wgpu 内部 WGSL 层），但需要额外的编译依赖。

---

## 第五步：从编译到 Launch

### Pipeline 创建和缓存

`WgpuServer::pipeline()`（`cubecl/crates/cubecl-wgpu/src/compute/server.rs:165`）：

```
1. 生成 `KernelId`（`type_id`、`address_type`、`cube_dim`、`mode`、`info: Option<Info>`，其中 `info` 是 `#[comptime]` 参数的类型擦除包装，`cubecl/crates/cubecl-runtime/src/id.rs:53`）
2. 检查 self.pipelines: HashMap<KernelId, (ComputePipeline, CompilerInfo)>
   → 命中则跳过编译，直接返回
3. 编译：compiler.compile_kernel(kernel, mode) → (CompiledKernel, CompilerInfo)
4. 创建 ShaderModule：device.create_shader_module(source)
5. 创建 ComputePipeline：device.create_compute_pipeline(module, layout)
6. 缓存到 self.pipelines
```

**KernelId 是编译缓存的关键**——它包含 `type_id`（Rust 类型标识，区分不同的 `#[cube]` 函数）、`address_type`（缓冲区寻址是 32 位还是 64 位）、`cube_dim`（workgroup 大小）、`mode`（是否 unchecked）以及 `info`（`#[comptime]` 参数的哈希值）。

最后一项 `info` 是整个融合机制的基础：`#[comptime]` 操作序列（如 `[Assign, Mul, Add, Tanh]`）被哈希并内嵌在 `KernelId` 中。不同的操作序列产生不同的 `KernelId` → 不同的缓存条目 → 不同的编译产物。但编译过程是一样的——都是展开同一个泛型 `elemwise_fuse` 模板。

### BindGroup 和 Dispatch

GPU 实际执行在 `WgpuStream::register_pipeline`（`cubecl/crates/cubecl-wgpu/src/compute/stream.rs:587`）：

```
1. 从 WgpuResource 构建 BindGroupEntry 列表（buffer offset/size 等）
2. 打开 ComputePass：encoder.begin_compute_pass()
3. pass.set_pipeline(&pipeline)
4. 创建 BindGroup：device.create_bind_group(layout, entries)
5. pass.set_bind_group(0, &bind_group, &[])
6. pass.dispatch_workgroups(x, y, z)   // 或 dispatch_workgroups_indirect()
```

对于 SPIR-V 有特化：通过 `pass.set_immediates()` 传入内联常量（SPIR-V specialization constants）。对于 MSL：过渡自定义 Metal 资源。

> ▶ **动手**：`cd src/ch2-expand-study && cargo test -- --nocapture`
> 现在你已经写过一个 kernel。回来看内部：Rust `+` 如何变成 `__expand_add_method(scope, rhs)`，与本文的表达式→IR→优化→代码生成流程对照。

---

## 编译期特化（#[comptime]）—— 融合的核心

### 机制

CubeCL 的关键创新不在于"把 Rust 编译成 GPU 代码"（这是所有 GPU 编译器都做的），而在于 **`#[comptime]` 参数使得一个 `#[cube]` 函数可以为任意操作序列生成专用代码**。

```rust
#[cube(launch_unchecked, address_type = "dynamic")]
fn elemwise_fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: &FuseBlockConfig,  // ← 编译期参数
) { ... }
```

`#[comptime]` 参数在宏展开时被标记为 `is_const: true`。在生成的 `expand()` 函数中，这类参数保持 Rust 值的身份——**不会**被转换为 IR 变量。在 `launch()` 包装函数中，它们的哈希值进入 `KernelId::info`。

这意味着：
- `elemwise_fuse::launch(config_a)` 和 `elemwise_fuse::launch(config_b)` 产生不同的 `KernelId`
- 不同的 `KernelId` 触发不同的编译或不同的缓存命中
- 编译器为每个 `config` 生成 it 专用的着色器代码（循环展开 `config.ops`，match 分发到具体的 `FuseOp`）

### 循环展开：发生在宏层面，不是 IR 优化

`#[unroll]` 标记的 `for` 循环不是在 IR 层面展开的——展开发生在**宏代码生成阶段**（`cubecl/crates/cubecl-macros/src/generate/expression.rs:259`）。宏计算出循环边界（必须是 `#[comptime]` 常量），然后在 Rust 的 `for i in start..end` 中为每次迭代调用一次 body 闭包。每次调用直接向 IR scope 注册指令——循环体被物理复制 N 次，不存在 `Branch::RangeLoop` 的 IR 节点。

注意 `UnrollVisitor`（`cubecl-core/src/post_processing/unroll.rs`）做的是**向量拆解**（vector unrolling）而非循环展开——它将宽向量（如 `vec16<f32>`）分解为多个标量/窄向量操作，以满足后端对向量宽度的限制。

### 和 Triton 的 JIT 对比

**通用问题**：把 kernel 源码编译成 GPU 二进制——编译分几段、错误何时暴露、冷启动预算多大。任何 GPU JIT 都要选：运行时一次性编译（探索友好、冷启动重），还是把能提前的提前到编译期（冷启动轻、迭代慢）。

| 维度 | CubeCL | Triton |
|------|--------|--------|
| 编译发生时机 | Rust crate 编译期（proc macro） | Python 运行时（`@triton.jit` 触发） |
| 特化机制 | `#[comptime]` 泛型参数 + Rust monomorphization | Python AST → Triton IR → 编译 |
| 融合方式 | `#[comptime]` 操作序列驱动模板展开 | 用户在 Python 层面写融合逻辑 |
| 编译器语言 | Rust proc macro + CubeCL IR → 后端编译器 | Python → Triton IR → Triton GPU IR → LLVM IR → PTX |
| 首次执行延迟 | 低（只是 comptime 值不同时的重新编译） | 高（完整 JIT 编译链） |
| 缓存粒度 | 每个 (kernel_fn, comptime_hash) 一个缓存条目 | 每个 (kernel_fn, input_signature) 一个缓存条目 |

**谁该用哪个**：

- **研究探索、单平台、要快迭代** → Triton：Jupyter 里改一行 Python 就重编译，毫秒级循环；代价是首次调用 100-500ms（完整 JIT + autotune），且 kernel 语法错误只能首次 launch 时暴露。
- **生产部署、多平台 AOT、冷启动敏感** → CubeCL：宏展开/类型/IR 错误在 `cargo check` 时暴露，首次 launch 只剩平台代码生成 ~10-50ms；代价是改一行 kernel 需 `cargo build`（秒级），探索阶段迭代比 Triton 慢。

一句话：CubeCL 用 Rust 编译期能力换冷启动轻、错误前置；Triton 用运行时编译换迭代流畅——选择取决于"你在探索还是在部署、冷启动预算多少、要不要多平台"。

> **延伸 · GPU 硬件层**：本篇停在"IR 如何生成与优化、何时编译"。要深入到 kernel 内部如何利用 Tensor Core / TMA / Tensor Memory 等硬件特性，见 [modern-gpu-programming-for-mlsys](../../modern-gpu-programming-for-mlsys/) 的 `chapter_tensor_cores`、`chapter_tma`、`chapter_tmem`。

---

## SPIR-V 的磁盘缓存

对于 Vulkan 后端，编译后的 SPIR-V 二进制缓存在磁盘上（`cubecl/crates/cubecl-wgpu/src/compute/server.rs:125`）：

```
Cache key: (properties_hash, kernel_id.stable_hash())
Cache path: {root}/spirv_{vendor}_{device}/{version}/{key_hash}
```

`properties_hash` 是设备属性的哈希（驱动版本、支持的 feature 等）——因此驱动更新自动触发缓存失效。这解决了 GPU 编译中常见的"驱动更新后运行错误"问题。

---

## 限制

1. **IR 不是图，是树**：Scope 树表示不支持 phi 节点。复杂的控制流（loop 内有多种 break 路径并合并值）需要额外的变量复写，生成的代码可能不如 CFG+SSA 优化充分。

2. **SPIR-V/MSL 的预编译要求**：非 WGSL 路径依赖预先编译好的二进制。kernel 作者需要维护每个平台的编译 pipeline。这也是为什么 `AutoCompiler` 可以 fallback 到 WGSL——WGSL 路径是始终可用的兜底。

3. **编译缓存维度爆炸**：即使是"同一个"融合 kernel，不同的 `#[comptime]` 操作序列产生不同的缓存条目。如果一个模型中有 20 种不同的融合模式，就有 20 个缓存条目。这在编译期（首次遇到每种模式时）累积延迟。

4. **没有链接时优化**：CubeCL 的编译是 per-kernel 的。多个 kernel 之间无法共享常量池或做跨 kernel 的内联。在 kernel 数量多的场景下（如大的 fusion 图，见 [Burn Kernel Fusion 系统设计](../burn/kernel-fusion-system-design.md)），每个 kernel 独立编译增加了缓存压力和首次执行延迟。编译缓存和 [Autotune 缓存](autotune-system-design.md)正交——前者缓存编译后的 shader，后者缓存最快 kernel 索引。

---

## 关键源码入口

- 宏入口：`cubecl/crates/cubecl-macros/src/lib.rs`
- 宏代码生成（launch/expand）：`cubecl/crates/cubecl-macros/src/generate/`
- IR 作用域与操作：`cubecl/crates/cubecl-ir/src/scope.rs`、`operation.rs`、`branch.rs`、`variable.rs`
- IR 优化：`cubecl/crates/cubecl-core/src/post_processing/mod.rs`
- 编译器 trait：`cubecl/crates/cubecl-runtime/src/compiler.rs`
- WGSL 编译器：`cubecl/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs`
- AutoCompiler 分派：`cubecl/crates/cubecl-wgpu/src/compiler/base.rs`
- Pipeline 创建与缓存：`cubecl/crates/cubecl-wgpu/src/compute/server.rs`
- GPU dispatch：`cubecl/crates/cubecl-wgpu/src/compute/stream.rs`
- KernelId：`cubecl/crates/cubecl-runtime/src/id.rs`

---

## 本篇小结

读完你现在能回答：

- `#[cube]` 宏在 Rust crate 编译期做了什么，生成的 IR 是嵌套 Scope 树而非平铺基本块
- IR 优化做哪些定点变换，GPU 编译为什么推迟到首次 launch 并按 `KernelId` 缓存
- 同一份 kernel 逻辑如何分别落到 WGSL / SPIR-V / MSL

> ✓ **完成自检**：能解释 `a + b` 在 `#[cube]` 函数中经历了什么——从 Rust 表达式到 IR 操作到 GPU 指令。

---

你已经理解了 kernel 如何编译和启动。下一步回答的问题是：**同一个 matmul 在 1024×4096 和 4096×1024 的最优 tile 大小不同——如何在首次执行时选出最快者，并缓存结果。** 这是 Autotune 系统设计的起点。

← [Fusion 系统设计](../burn/kernel-fusion-system-design.md) | → 下一篇：[Autotune 系统设计](autotune-system-design.md)

动手：[src/ch1-gelu-variants/](../../src/ch1-gelu-variants/) — GELU kernel 变体练习 · [src/ch2-expand-study/](../../src/ch2-expand-study/) — 宏展开观察
