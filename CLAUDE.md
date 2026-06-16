# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 性质

本仓库是 [Tracel](https://github.com/tracel-ai) 开源生态的系统设计分析文档，覆盖 Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。文档以系统设计分析为主（从代码中提取设计决策、权衡和机制链路），章节教程为辅（逐步 walkthrough 源码机制）。

核心产出是五篇系统设计文章，覆盖 Burn/CubeCL 技术栈的四个核心系统：Kernel Fusion、Autotune、JIT 编译管线、Autodiff，以及一篇全链路全景文章。

## 架构

```
docs/                          ← 分析文档（Markdown，不可执行）
  architecture.md              ← 类型栈、Trait 边界与分层组合
  concept-index.md             ← 概念反向索引：~70 个关键概念 → 文章位置
  SOURCE-VERSION.md            ← 源码基准、API 漂移矩阵、更新检查清单
  <project>/                   ← burn/ | cubecl/ | cubek/
    <system>-system-design.md  ← （系统设计）从设计决策到源码实现的完整分析
    summary.md                 ← （导航）指向系统设计文章 + 章节教程的索引
    index.md                   ← （计划）章节写作计划 + 入门引导
    N-title.md                 ← （章节）跟练教程、逐机制展开源码
  appendix/                    ← 附录（翻译、归档）

src/                           ← 示例与作业（Cargo workspace，可执行）
  Cargo.toml                   ← workspace 根，7 个 member crate
  burn-test/                   ← Fusion 融合日志示例 + 测试
  autodiff-test/               ← Autodiff 梯度验证示例 + 测试
  ch1-gelu-variants/           ← CubeCL GELU kernel 变体作业
  ch2-expand-study/            ← CubeCL 宏展开观察作业
  ch3-trait-study/             ← （骨架）trait 机制作业
  fusion-ch2-queue/            ← （骨架）队列机制作业
  fusion-ch3-drain/            ← （骨架）drain 机制作业

burn/          (gitignored)    ← tracel-ai/burn 参考源码
cubecl/        (gitignored)    ← tracel-ai/cubecl 参考源码
cubek/         (gitignored)    ← tracel-ai/cubek 参考源码
burn-onnx/     (gitignored)    ← tracel-ai/burn-onnx 参考源码
```

**双用途仓库**：`docs/` 中的文档和 `src/` 中的练习 crate 互补。系统设计文章独立成篇，不需要对应 crate；章节教程与练习 crate 编号对应。

**添新内容时**：
- 新系统设计文章：放入对应 `<project>/` 目录，以 `<system>-system-design.md` 命名
- 新章节教程：遵循 `N-title.md` 命名，对应 `src/<chapter-crate>/` 练习

## 文档类型与导航

| 类型 | 命名 | 导航要求 |
|------|------|----------|
| 系统设计 | `*-system-design.md` | 末尾 `← 上一篇 \| → 下一篇` 链入相邻文章 |
| 导航页 | `summary.md` | 末尾 `→ 推荐入口` 指向系统设计文章或 README |
| 章节计划 | `index.md` | 末尾链接系列索引 |
| 章节教程 | `N-title.md` | 末尾 `← 系统设计文章 \| 下一章 →` |
| 附录 | `appendix/*.md` | 末尾注明来源和更新日期 |

阅读路径：`architecture → 全景 → Fusion → JIT → Autotune → CubeK → Autodiff`。Inline exercise callouts（`▶ 动手` / `▶ 跟练`）在文章中穿插，读者在概念点上停下来验证。添加新文章或练习时：1) 按路径顺序插入 2) 更新上下游导航 3) 在合适的文章中加 inline callout。路径见 [README.md](README.md)。

## 命令

### 设置（一次性）

参考仓库（已 gitignore）需要在本地 clone 才能解析 `src/` 中的 crate 依赖：

```bash
git clone https://github.com/tracel-ai/burn.git
git clone https://github.com/tracel-ai/burn-onnx.git
git clone https://github.com/tracel-ai/cubecl.git
git clone https://github.com/tracel-ai/cubek.git
```

### 运行练习（来自 `src/`）

CubeCL 章节的测试：

```bash
cd src/ch1-gelu-variants && cargo test -- --nocapture
cd src/ch2-expand-study && cargo test -- --nocapture
```

含日志输出的 Fusion 示例：

```bash
cd src/burn-test && RUST_LOG=burn_fusion=trace cargo run --release
cd src/fusion-ch2-queue && RUST_LOG=burn_fusion=trace cargo run --release
```

### 验证参考仓库状态

```bash
# 统计 burn-onnx 测试覆盖率
grep -r 'insta::assert_snapshot' burn-onnx --include '*.rs' | wc -l
grep '^status' burn-onnx/crates/onnx-official-tests/expectations.toml | sort | uniq -c
```

### Bootstrapping 一个新章节 crate

从已有骨架复制（如 `ch3-trait-study`），调整 `Cargo.toml` 中的依赖路径使其指向正确的参考仓库，并将 crate 添加到 `src/Cargo.toml` 的 `members` 列表中。

## 技术写作规范

说服力来自读者可自行验证的事实（源码路径、测试数字、复验命令），而非替读者决定什么重要。

### 禁用的话术模式

| # | 模式 | 违例 | 替代 |
|---|------|------|------|
| 1 | 替读者预设心理 | "比你想象"、"你没法"、"你只能" | 直接陈述事实 |
| 2 | 强制二元对立 | "不是 X，是 Y"、"核心不是 A 而是 B" | 正面描述；多个属性共存 |
| 3 | 绝对化宣称 | "零开销"、"永远不会"、"真正的" | 具体描述机制，加限定词 |
| 4 | 关闭解释 | "这就是为什么"、"关键在于" | "这解释了"、"以 X 为例" |
| 5 | 替读者过滤 | "这不是学究气的分类。这才是……" | 直接说它是什么 |
| 6 | 轻视已知 | "（谁都会写）" | "（是成熟的已知模式）" |
| 7 | 排他性焦虑 | "门票"、"你能不能进入" | "实际需求"/"硬约束" |
| 8 | 夸大后果 | "后果比你想象的严重" | 说清具体影响，不预设严重程度 |

### 写作自检

每段写完问自己：
1. 能用源码路径/测试数字/复验命令支撑吗？
2. 在替读者决定什么吗？
3. "不是 X，是 Y" 去掉对立、正面说 Y，更准确吗？

### 价值写作方法论

系统设计文章的价值不在"这个设计好"的评价，而在让读者自己得出价值判断。五种技巧：

**1. 用可测量的成本替代形容词。**
不说"swizzle 对性能很重要"，说"不做 swizzle 时，同一 warp 内的 32 个线程访问 shared memory 的不同 bank 会串行化——最坏情况下 32 次访问而非 1 次"。把价值锚定到可被第三方验证的量上。

**2. 用对比框架替代评价框架。**
不说"策略枚举比参数网格好"，说 Triton 做了什么（5×4×3×3=180 候选）、CubeCL 做了什么（6-35 个候选）、各自的设计代价。数字自己会说话——读者不需要被告知哪个更好。对比暴露了 trade-off 的维度，价值判断留给读者的使用场景。

**3. 展示替代方案的不可接受性。**
不说"autotune 很重要"，展示如果不做 autotune 会怎样：用 gemm 的 tile size 跑 matvec 浪费 90% 的 compute unit。价值来自"如果不这样，会出现什么具体后果"——读者自己推导出必要性。

**4. 把价值附着在"谁需要这个"上。**
不说"这个设计很精妙"，说"什么场景下你需要关心这个设计：你的模型在不同 batch size 下跑，部署到不同 GPU，用了 kernel fusion"。读者自己对号入座。同时点明谁不需要——反面场景让正面价值更可信。

**5. 用一个极限案例锚定问题边界。**
极端案例不是评价技术的好坏，而是暴露设计的硬约束。例如"180 个候选 × 13 次 launch × 10μs ≈ 23ms autotune 延迟——但一个 BERT 模型有 ~30 个不同的 matmul shape，加上 fusion 组合爆炸，autotune 时间秒级。而用户的第一个推理请求通常在 100ms 超时预算内。"这解释了策略枚举的硬约束，不评价它。

**价值写作的自检**：如果删掉所有暗示"这个设计好"的措辞，只保留事实（数字、对比、极端案例、场景归属），读者仍然能自己形成价值判断——那就对了。如果删掉评价后段落失去信息量，说明段落在替读者思考。

### 源码引用格式

- 路径写全：`crates/…/file.rs` + 符号名（如 `pub struct MultiStream`）
- 行号仅作近似参考，随版本漂移
- 数字附带来源（仓库、文件、可用脚本复验）

## 相关文档

系列导航见 [README.md](README.md)（知识图谱 + 阅读路径）。写作规范副本在 `.cursor/rules/writing-style.mdc`（Cursor 环境自动加载）。
