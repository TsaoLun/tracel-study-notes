> **归档**：旧架构的章节计划。Burn-ONNX 的系统设计分析尚未完成。当前阅读路径见 [README](../../../README.md)。

# Burn ONNX 专题写作计划（已归档）
> **读计划前**：若你尚未读过 ONNX 地图，可先扫一眼 [架构一览](../onnx-summary.md#架构一览)（5 分钟），再回到这里。

---

## 入门引导

### 你不需要先会 ONNX 规范

本专题假设你会 Rust，并了解 Burn 的基本使用。不要求读过 ONNX protobuf 规范——每章会解释涉及的 ONNX 算子语义。

### 本专题的「主示例」是什么？

全程用一个可跟踪的 ONNX 模型示例（在 `burn-onnx/crates/model-checks/` 中选一个最小模型，如 `all-minilm-l6-v2`）：

| 名字 | 是什么 | 在哪里 |
|------|--------|--------|
| **`onnx-ir`** | IR 流水线的核心 crate：protobuf 解析、类型推断、simplify pass | `burn-onnx/crates/onnx-ir/` |
| **`onnx-ir-derive`** | proc-macro：`#[node]` 注解自动生成 Node 枚举变体 | `burn-onnx/crates/onnx-ir-derive/` |
| **`ModelGen`** | 构建期入口：读取 ONNX → 经过 IR 流水线 → 生成 `model.rs` | `burn-onnx/crates/burn-onnx-gen/` |
| **`expectations.toml`** | ONNX 官方测试的预期结果（1615 条），统计复验命令见 [README](../../../README.md) |

**跟跑方式**：在 clone 的 `burn-onnx/` 仓库中运行模型测试：

```bash
cd burn-onnx
cargo test -p onnx-ir --test
```

---

## 定位与读者

**目标读者**：读过 [Burn ONNX 地图](../onnx-summary.md)；想知道"ONNX 图如何变成 Rust 源码"的 Rust 开发者。
**不覆盖**：Burn 类型栈与融合流（见 [Burn 地图](../summary.md)）、CubeCL JIT 编译器（见 [CubeCL 专题](../../cubecl/index.md)）。

---

## 写作约定

1. **每章开头**：2–3 句话说明锚点示例与读完能解释什么。
2. **每章一个主示例**，源码路径写全（相对 `burn-onnx/` 仓库根）。
3. **正文先可运行、再钉源码**；IR 转换深度逐章加码。
4. **章末**：小结 + 作业 + 下章预告。
5. **完整术语表**以 [ONNX 地图文末](../onnx-summary.md) 为准。

---

## 章节目录

| 章 | 文件 | 标题 | 读完能解释 | 核心源码锚点 |
|:---:|------|------|------------|--------------|
| 1 | 1-protobuf-to-ir.md | Protobuf → IR：解析、初始化器、Constant 折叠 | ONNX protobuf 如何转换为 `RawNode`；初始化器如何转为 `Constant` 节点；external data 的 mmap 加载 | `crates/onnx-ir/src/proto_conversion.rs`、`crates/onnx-ir/src/initialization.rs`（Phase 1–2） |
| 2 | 2-type-inference.md | 类型推断：`ScalarTensor` vs `ScalarNative` | 迭代类型推断的算法；为什么需要区分 tensor 标量和原生标量；推断失败时的错误信息 | `crates/onnx-ir/src/type_inference.rs`（Phase 3） |
| 3 | 3-attention-fusion.md | 注意力融合：SDPA 分解 → 单一 `Attention` 节点 | `coalesce_attention` 的模式匹配算法（5 步匹配、4 种变体的适配）；为什么 AOT 编译器能做运行时 loader 做不到的事 | `crates/onnx-ir/src/simplify/coalesce_attention.rs`（~1367 行） |
| 4 | 4-simplify-passes.md | simplify 定点循环：8 个 pass 的协作 | 8 个 simplify pass 的职责；定点循环的收敛条件（≤10 轮）；pass 的执行顺序设计 | `crates/onnx-ir/src/simplify/` 目录全部 8 个 pass |
| 5 | 5-finalization-codegen.md | finalization → codegen：Node 枚举 dispatch | `finalization` 阶段做什么；`#[node]` 宏如何生成 Node 枚举变体；169 种 Node 变体的 dispatch 逻辑 | `crates/onnx-ir/src/finalization.rs`、`crates/onnx-ir-derive/`（Phase 5–6） |
| 6 | 6-testing-modelgen.md | 测试体系与 ModelGen：从 1615 条测试看质量保证 | 三层测试体系（单元/集成/官方 ONNX 测试）；`expectations.toml` 的状态语义；`ModelGen` 如何生成可调试的 Rust 源码 | `crates/onnx-official-tests/expectations.toml`、`crates/burn-onnx-gen/` |

---

## 各章要点（写作 checklist）

### 第一章（待写）：Protobuf → IR

- [ ] ONNX protobuf 格式概览（graph、node、initializer、value_info）
- [ ] `proto_conversion.rs`：Protobuf → `RawNode` 的转换
- [ ] 初始化器处理：权重数据如何转为 Constant 节点（Phase 1）
- [ ] external data：大权重文件的 mmap 加载策略
- [ ] Phase 2：属性类型化 + `Gemm→Linear` 等早期合并
- [ ] 跟练：选一个最小的 ONNX 模型，在 `onnx-ir/src/pipeline.rs` 中设断点观察 Phase 1-2 的输出

### 第二章（待写）：类型推断

- [ ] 为什么 ONNX 需要类型推断：protobuf 中类型信息不完整
- [ ] `ScalarTensor` vs `ScalarNative` 的语义区分
- [ ] 迭代推断算法：多轮扫描直到收敛
- [ ] 推断失败的诊断信息
- [ ] 跟练：给模型注入一个类型错误，观察推断失败时的 panic 信息

### 第三章（待写）：注意力融合

- [ ] 从 ONNX 地图 §注意力融合 展开：5 步匹配算法的完整源码追踪
- [ ] Softmax 锚点发现 → 后向追踪（mask、scale、MatMul）→ 前向验证 → 节点替换
- [ ] 4 种模型变体的适配（标准、QK 预缩放、仅 Q 预缩放、对称预缩放）
- [ ] RF-DETR 的特殊处理：K transpose 合并 + 编译器插入修正 Transpose
- [ ] `expectations.toml` 中 `attention_*_expanded` 测试：27/62 pass 的语义
- [ ] 跟练：在 `coalesce_attention.rs` 中找到 RF-DETR 的特殊匹配臂

### 第四章（待写）：simplify 定点循环

- [ ] 8 个 pass 的完整清单与职责：
  - `coalesce_attention`：SDPA 模式识别
  - `constant_fold`：常量折叠
  - `dead_nodes`：死节点消除
  - `idempotent`：幂等操作消除
  - `identity_element`：恒等元素消除
  - `permute_reshape`：permute + reshape 合并
  - `redundant_nodes`：冗余节点消除
  - `constant_shape`：常量形状传播
- [ ] 定点循环：`loop { run all passes; break if no changes || iterations >= 10 }`
- [ ] pass 的顺序设计：为什么 `constant_fold` 要在 `dead_nodes` 之前
- [ ] 与 CubeCL cubecl-opt 定点循环的对比：都是 iterate until quiescence，但简化的内容不同（图拓扑 vs SSA 指令）
- [ ] 跟练：在 loop 中加 counter，观察不同模型收敛所需的轮数

### 第五章（待写）：finalization → codegen

- [ ] `finalization` 阶段：IR → `Node` 枚举的最终转换
- [ ] `#[node]` proc-macro（`onnx-ir-derive`）：自动生成 enum 变体与 dispatch 代码
- [ ] 169 种 Node 变体的组织方式（按算子类别分组）
- [ ] codegen：`Node` 枚举 → `model.rs` 的 Rust 源码生成
- [ ] 生成代码的质量：为什么生成的 `model.rs` 可调试（不像 PyTorch 的 TorchScript）
- [ ] 跟练：生成一个最小模型，阅读 `model.rs` 中的 forward 函数

### 第六章（待写）：测试体系与 ModelGen

- [ ] ONNX 官方测试套件（ONNX v1.19.0，1615 条）
- [ ] `expectations.toml` 的状态：`pass` / `skip` / `fail` 的语义与更新流程
- [ ] 三层测试体系：
  - 单元测试（各 pass 的独立测试）
  - 集成测试（模型检查：albert、yolo、silero-vad 等）
  - 官方 ONNX backend 测试
- [ ] `ModelGen`：从构建期到运行时的完整流程
- [ ] 权重格式 `model.bpk`（BurnPack）的设计
- [ ] 跟练：在 `expectations.toml` 中找一条 `fail` 项，分析失败原因（是缺少算子支持还是精度不达标）

---

## 与其他专题的关系

| 维度 | ONNX 专题（本专题） | [Burn Fusion 专题](../fusion/index.md) | [CubeCL 专题](../../cubecl/index.md) |
|------|---------------------|----------------------------------------|--------------------------------------|
| **推迟什么** | 模型导入——从运行期推迟到构建期 | 连续 op 的合并时机 | GPU 指令的选择与优化 |
| **决策时机** | `cargo build`（L1） | 运行期（读张量前 drain） | 首次 launch JIT miss（L2） |
| **交汇点** | 生成的 `model.rs` 是普通 Burn 代码，穿过 Fusion 流 | 融合后的 FuseTrace 交给 CubeCL launch | CubeCL runtime 编译 + 执行 |

---

## 进度

| 状态 | 文档 |
|------|------|
| 📋 待写 | `1-protobuf-to-ir.md` … `6-testing-modelgen.md` |
| 📎 地图 | `../onnx-summary.md` |
| 📎 本计划 | `index.md` |

---

## 系列导航（Burn 底层机制系列内）

| 篇 | 文档 | 状态 |
|:---:|------|------|
| 地图 | [../summary.md](../summary.md) | 已发布 |
| ONNX 地图 | [../onnx-summary.md](../onnx-summary.md) | 已发布 |
| Fusion 计划 | [../fusion/index.md](../fusion/index.md) | 已发布 |
| **ONNX 计划** | **本文** | 新发布 |
| Autodiff 地图 | [../autodiff/summary.md](../autodiff/summary.md) | 已发布 |

*Burn 底层机制 · ONNX 专题 · [系列索引](../../../README.md)*
