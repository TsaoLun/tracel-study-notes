# ONNX 模型编译：Burn ONNX 的构建时代码生成

## 读前须知

- **Burn ONNX 是什么**：在 `build.rs` 里运行的 AOT 编译器——把 ONNX 图翻译为可调试的 Rust 源码（`model.rs`）与权重（`model.bpk`），运行时二进制不依赖 ONNX Runtime。
- **本文覆盖**：IR 流水线（6 阶段）、注意力融合（hero pass）、分区编译、三层测试体系。Burn 类型栈与融合流见 [综合地图](blog-burn-summary.md)，GPU JIT 见 [CubeCL 篇](blog-cubecl-summary.md)。
- **统计基准**：测试数字来自 `burn-onnx` 仓库 `crates/onnx-official-tests/expectations.toml`（ONNX v1.19.0，1615 条），复验命令见 [README](README.md#源码版本与数字校验)。

系列分工与导航见 [README](README.md)。

---

## 架构一览

```
model.onnx (protobuf)
    ↓ build.rs
Phase 1: Protobuf → RawNode（初始化器 → Constant 节点）
Phase 2: 属性类型化 + Gemm→Linear 早期合并
Phase 2b: RNN 权重链折叠
Phase 3: 迭代类型推断（ScalarTensor vs ScalarNative 区分）
Phase 4: Identity 消除 → 8 个 simplify pass 定点循环
    ├── coalesce_attention（SDPA 分解 → 单一 Attention 节点）
    ├── constant folding / CSE + DCE / permute-reshape / …
    └── 定点循环 ≤10 轮
Phase 5–6: finalization → Node 枚举 → codegen（169 种 Node 变体 dispatch）
    ↓
model.rs + model.bpk → 普通 Burn 代码 → Autodiff<Fusion<CubeBackend>>
```

---

## 核心结论

> Burn ONNX 在构建期做运行时 loader 做不到的事：模式匹配（SDPA → `Attention`）、常量折叠、分区——输出是普通 Rust，运行时无 ORT、无 protobuf。生成代码与手写 Burn 模型走同一条融合流与 CubeCL JIT 路径。

---

## 注意力融合：AOT 编译器与 loader 的核心差异

PyTorch 的 `F.scaled_dot_product_attention(Q, K, V)` 在导出 ONNX（尤其是旧版 opset）时不生成 `Attention` 算子，而是拆解为：`MatMul(Q, K^T) → [Div/Mul(scale)] → [Add(mask)] → Softmax → MatMul(scores, V)`。

ONNX Runtime 的路径：5 次 kernel launch，5 次显存读写中间结果——运行时 loader 按图执行。

Burn ONNX 的路径：编译器在生成代码前识别 SDPA 分解模式，融合为单一 `Attention` 节点——生成代码调用 Burn 原生注意力，经融合流与 CubeK/CubeCL 内核执行。这种模式匹配是 AOT 编译器特有的能力。

`crates/onnx-ir/src/simplify/coalesce_attention.rs`（约 1367 行）的算法：

1. 找所有 `Softmax` 节点（注意力模式的锚点）
2. 从 Softmax 向后追踪：`Softmax ← [Add(mask)] ← [Div/Mul(scale)] ← MatMul`
3. 从 MatMul 向后追踪：两个输入——一个来自 Q，一个来自 `K → Transpose(-2, -1)`
4. 从 Softmax 向前追踪：输出被 `MatMul(scores, V)` 消费
5. 全部匹配后，用一个 `Attention { Q, K, V, scale, mask }` 节点替换

难的是不同模型导出不同模式：

| 变体 | Q 的变换 | K 的变换 | 出现在 |
|------|----------|----------|--------|
| 标准 | 无 | Transpose(-2, -1) | 大多数模型 |
| QK 预缩放 | Transpose([0,2,1,3]) → Mul(scale) | Transpose([0,2,3,1]) → Mul(scale) | RF-DETR |
| 仅 Q 预缩放 | Transpose → Mul(scale) | Transpose(-2, -1) | DINOv2 |
| 对称预缩放 | Mul(scale) | Transpose → Mul(scale) | DepthPro |

RF-DETR 的情况：K 的 transpose 把 head-split 和 key-transpose 合并为一次 perm `[0,2,3,1]`（对比 Q 的 `[0,2,1,3]`）。匹配成功后编译器插入修正的 K Transpose 恢复标准语义。

注意力融合使 62 个 `attention_*_expanded` 测试中 27 个达到 `pass`（`expectations.toml` 统计）。一个 pass 覆盖一整类模式——生成代码质量的改善来自模式匹配，而非针对每个测试的逐一修复。

---

## 支撑这场胜利的流水线

注意力融合是 8 个 simplify pass 中的一个（定点循环最多 10 轮，通常 3–5 轮收敛）。它们运行在 IR 流水线的 Phase 4b。总览：

| 阶段 | 模块 | 做什么 |
|:---:|------|--------|
| 1 | `initialization` | Protobuf → `RawNode`，常量初始化，mmap 外部权重 |
| 2 | `node_conversion` | 类型化节点 + `Gemm→Linear` 等早期合并 |
| 2b | 早期 constant fold | RNN 权重的 Slice→Concat 链折叠（最多 10 轮） |
| 3 | `type_inference` | 迭代形状/类型推断，`ScalarTensor` vs `ScalarNative` 区分 |
| 4 | `post_processing` | Identity 消除 |
| 4b | `simplify` | **8 个 pass** 定点简化（含注意力融合） |
| 5 | `finalization` | 清理、图输出整理 |
| 6 | `convert_to_graph` | `RawNode` → `Node` 枚举 |
| — | `burn-onnx` codegen | `ModelGen` / `impl_node_codegen_dispatch!` → `model.rs`（169 种 Node 变体） |

### 各阶段的动机

**Phase 1**：`GraphProto` 是未类型化的 protobuf 对象，解析为 `RawNode` 初胚。外部数据引用（>2GB 模型把权重放在 sidecar 文件中）用 memory-mapped I/O 处理。

**Phase 2**：`ProcessorRegistry` 把原始属性（如 `{ name: "kernel_shape", ints: [3, 3] }`）转为类型化配置（`Conv2dConfig`）。同时做早期模式合并（`Gemm → Linear`，`MatMul+Add → Linear`）。

**Phase 2b**：PyTorch 导出 RNN 权重时常 `Slice`→`Concat`→`Unsqueeze` 链重组——不折叠为常量的话 Phase 3 的类型推断会被 "Dynamic" 阻塞。定点循环（最多 10 轮）处理连锁折叠。

**Phase 3**：迭代式类型推断——收集输入偏好 → 同步已知类型 → 推断输出 → 检查收敛。`ScalarTensor` vs `ScalarNative` 区分解决了"ONNX 里到处是 0 维张量，但 Rust 代码需要真的 `i64` 值"的问题。

**Phase 4–4b**：消除 Identity 节点后，运行 8 个 simplify pass 的定点循环。除注意力融合外还有：permute-reshape、constant shape folding（裸 `Shape(x)` 故意保留以支持动态输入）、constant folding、idempotent elimination（`Relu(Relu(x))`→`Relu(x)`）、CSE + DCE。Pass 间有级联——CSE 合并 → DCE 清除 → 下一轮 constant folding 可折叠被释放的常量。

**Codegen**：Phase 5–6 在 `onnx-ir` 内完成 finalization 与 `RawNode`→`Node` 转换。Rust 源码生成在 `burn-onnx` 的 `ModelGen`：`impl_node_codegen_dispatch!` 对 169 种 `Node` 变体生成 match dispatch（`burn-onnx/src/burn/node_codegen.rs`）。ONNX 算子表见 `SUPPORTED-ONNX-OPS.md`（当前约 168/209 import 支持）。

---

## 分区编译：当图大到编译器吃不下

SDXL 的 UNet 有上万个节点，全部塞进一个 `forward()` 会导致 Rust 编译不可接受。

`crates/burn-onnx/src/burn/partition.rs`（437 行）的算法：

1. **O(n) 扫描切分成本**：前缀和差分数组（`delta[producer+1] += 1, delta[consumer] -= 1`）计算每个切分点有多少张量跨越边界。
2. **贪心选切点**：将图切为 64–256 节点的子模块，每个窗口内选切分成本最低的位置。
3. **常量重排**：把常量节点移到它们首个消费者的前面——避免权重在开头定义、500 个节点后才用，导致被迫跨越多分区。

---

## 怎么保证正确

三层测试，从锁死输出到验证数值：

1. **625 个快照测试**（`insta::assert_snapshot`）：锁死 codegen 输出——改逻辑后 `cargo insta review`
2. **178 个集成测试**：Python `onnx.reference.ReferenceEvaluator` 作 ground truth，Rust 逐元素对比
3. **1615 个上游测试**（ONNX v1.19.0，`expectations.toml`）：

| status | 含义 | 数量（当前） |
|--------|------|-------------|
| `pass` | codegen + compile + 数值匹配 | 722 |
| `fail-compare` | 编译运行但数值偏差 | 179 |
| `skip-compile` | codegen 成功但 Rust 不编译 | 230 |
| `skip-codegen` | codegen 失败/拒绝 | 484 |

在能 codegen 且能编译的集合（722+179=901）中，约 80.1% 数值通过（722/901）。

`expectations.toml`（6653 行）声明每个测试的期望状态。代码改动让 `skip-codegen` 测试突然开始生成代码时，CI 告警——期望文件需更新以反映新状态，防止无声回归。

---

## 代价和收益

代价：IR 流水线 + 8 pass 简化 + 169 个 Node codegen 变体 + 3 层测试。168/209 ONNX 算子行 import 支持（约 80%），上游 pass 非 100%。

收益不在于覆盖率百分比。在于能跑的模型运行路径与手写 Burn 代码完全相同：
- 无运行时依赖（无 ORT 共享库，无 protobuf 解析器）
- 可调试（`forward()` 是普通 Rust 函数，IDE 可跳转，调试器可设断点）
- 可修改（生成的代码是源码，`Relu` 换成 `Gelu` 后重编译）
- 可嵌入（`LoadStrategy::Embedded` 把权重编译进二进制，`no_std` 也能跑）

---

## 词汇说明表

| 术语 | 简要说明 |
|------|----------|
| **AOT 编译器** | Ahead-of-Time：构建期把 ONNX 译为 Rust，非运行时解释 protobuf |
| **coalesce_attention** | 约 1367 行的 simplify pass：SDPA 分解模式 → 单一 `Attention` 节点 |
| **8 个 simplify pass** | 每轮迭代按固定顺序执行；定点循环最多 10 轮 |
| **expectations.toml** | 1615 条上游测试的声明式 status |
| **分区编译** | 大图切为 64–256 节点子模块（`partition.rs`） |
| **IR 流水线** | Protobuf → 类型推断 → 8 个 simplify pass 定点迭代 → finalization → codegen |

*Burn 底层机制系列 · ONNX AOT 编译 · 导航见 [README](README.md)*
