# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 性质

本仓库是 [Tracel](https://github.com/tracel-ai) 开源生态的源码级机制分析文档，覆盖 Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。每篇文档以源码路径和可复验命令为论据，解释设计动机和实现链路。
每篇文档覆盖一个主题的完整机制链路，读者读完应能解释"为什么这么设计"和"怎么做到的"。

## 架构

```
docs/                          ← 机制分析文档（Markdown，不可执行）
  architecture.md              ← 跨项目架构主线
  <project>/                   ← burn/ | cubecl/ | cubek/
    summary.md                 ← （地图）心智模型 + 架构连接
    index.md                   ← （计划）章节写作计划 + 入门引导
    N-title.md                 ← （章节）跟练教程、逐机制展开源码

src/                           ← 示例与作业（Cargo workspace，可执行）
  Cargo.toml                   ← workspace 根，members 列出每个章节对应的 crate
  <chapter-crate>/             ← 一个章节对应一个 crate（如 ch1-gelu-variants、burn-test）
    Cargo.toml                 ← 依赖路径指向 ../../burn 或 ../../cubecl（gitignored 参考仓库）
    src/main.rs | src/lib.rs

burn/          (gitignored)    ← tracel-ai/burn 参考源码
cubecl/        (gitignored)    ← tracel-ai/cubecl 参考源码
cubek/         (gitignored)    ← tracel-ai/cubek 参考源码
burn-onnx/     (gitignored)    ← tracel-ai/burn-onnx 参考源码
```

**双用途仓库**：`docs/` 中的文档和 `src/` 中的练习 crate 通过章节编号一一对应。一篇新章节的典型添加流程是先写如 `docs/cubecl/3-trait-impl.md` 的文档，然后在 `src/ch3-trait-study/` 中创建练习骨架，并将该 crate 添加到 workspace 的 `members` 列表中。

文档按**主题（project）→ 类型（summary/index/chapter）**组织。summary 先建立心智模型，index 规划章节序列，chapter 逐步展开机制。读者通过优先阅读 summary 来入门。

**添新内容时**：遵循此结构。一个新主题需要一个目录，其中包含 `summary.md`，可选包含 `index.md` 和按顺序编号的章节。如果一个章节有练习，需在 `src/` 中创建对应的 crate。

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

### 源码引用格式

- 路径写全：`crates/…/file.rs` + 符号名（如 `pub struct MultiStream`）
- 行号仅作近似参考，随版本漂移
- 数字附带来源（仓库、文件、可用脚本复验）

## 相关文档

系列导航见 [README.md](README.md)。
