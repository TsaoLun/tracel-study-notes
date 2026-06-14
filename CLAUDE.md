# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 性质

本仓库是 [Tracel](https://github.com/tracel-ai) 开源生态的系统设计分析文档，覆盖 Burn（Rust DL 框架）、CubeCL（多平台 GPU 编译器）、CubeK（高性能算子库）、Burn-ONNX（AOT 模型导入）。文档以系统设计分析为主（从代码中提取设计决策、权衡和机制链路），章节教程为辅（逐步 walkthrough 源码机制）。

核心产出是五篇系统设计文章，覆盖 Burn/CubeCL 技术栈的四个核心系统：Kernel Fusion、Autotune、JIT 编译管线、Autodiff，以及一篇全链路全景文章。

## 架构

```
docs/                          ← 分析文档（Markdown，不可执行）
  architecture.md              ← 跨项目架构主线："决策推迟"
  <project>/                   ← burn/ | cubecl/ | cubek/
    <system>-system-design.md  ← （系统设计）从设计决策到源码实现的完整分析
    summary.md                 ← （导航）指向系统设计文章 + 章节教程的索引
    index.md                   ← （计划）章节写作计划 + 入门引导
    N-title.md                 ← （章节）跟练教程、逐机制展开源码

src/                           ← 示例与作业（Cargo workspace，可执行）
  Cargo.toml                   ← workspace 根，members 列出每个章节对应的 crate
  <chapter-crate>/             ← 一个章节对应一个 crate
    Cargo.toml                 ← 依赖路径指向 ../../burn 或 ../../cubecl
    src/main.rs | src/lib.rs

appendix/                      ← 附录（翻译、归档）
  automatic-kernel-fusion.md   ← 旧博客中文翻译（含烧 2026.05 源码更新）

burn/          (gitignored)    ← tracel-ai/burn 参考源码
cubecl/        (gitignored)    ← tracel-ai/cubecl 参考源码
cubek/         (gitignored)    ← tracel-ai/cubek 参考源码
burn-onnx/     (gitignored)    ← tracel-ai/burn-onnx 参考源码
```

**双用途仓库**：`docs/` 中的文档和 `src/` 中的练习 crate 互补。系统设计文章独立成篇，不需要对应 crate；章节教程与练习 crate 编号对应。

**添新内容时**：
- 新系统设计文章：放入对应 `<project>/` 目录，以 `<system>-system-design.md` 命名
- 新章节教程：遵循 `N-title.md` 命名，对应 `src/<chapter-crate>/` 练习

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
