# burn-test: 观察 Fusion 引擎

验证 [Fusion 系统设计](../../docs/burn/kernel-fusion-system-design.md) 中描述的惰性排队和融合机制。

> **对应的 NN 概念**：`clone → *2 → +1 → tanh` 是一串 element-wise 算子，正是 NN 里"又多又碎"的那类——所以融合有收益。背景见 [primer · 为什么 element-wise 又多又碎](../../docs/primer.md#part-a--领域最小集)。

## 运行

```bash
cd src/burn-test
BURN_FUSION_LOG=full cargo run --release
```

## 日志逐行解读

`BURN_FUSION_LOG=full` 输出展示融合引擎完整工作流程，分为三个阶段。

### 阶段一：tensor 创建（Init）

```
[fuser] closed on Init (unsupported op variant); had 0 ops
[fuser] closed on Init (unsupported op variant); had 0 ops
[fuser] closed on Init (unsupported op variant); had 0 ops
[fuser] closed on Init (unsupported op variant); had 0 ops
[stream] op 0 Init::InitOperationIr → new block (total: 1)
```

四条 `[fuser] closed` 对应四个 `OperationFuser`（ElementWise、Matmul、Reduce、ReduceBroadcasted）——全都拒绝 `Init` 操作（"unsupported op variant"）。`Init` 只是记录"tensor 已分配"，不可融合。

`[stream]` 表示 StreamOptimizer 为它创建了新 Block。随后 `exploration completed: lazy, 1/1 ops optimized`——仅一条 op，生成的 fusion trace 标注 `un-fused (1 op)`。

### 阶段二：读到 z 时触发 drain——融合发生

```diff
[stream] op 0 NumericFloat::MulScalar → new block (total: 1)
[stream] op 1 NumericFloat::AddScalar → accepted in 1/1 block(s)
[stream] op 2 Float::Tanh → accepted in 1/1 block(s)
```

三个 element-wise op 先后到达。第一个 `MulScalar` 创建新 Block，后两个 `AddScalar` 和 `Tanh` 因为与已有 Block 共享 tensor 依赖（`t1`、`t2`），被接受——`Block::register()` 中的 `ids.contains()` 检查命中。

```
[fuser] closed on BaseFloat (base fuse rejected); had 3 ops, est_bindings=3/7
（×4，四个 fuser 各一行）
```

四个 fuser 都接受了三个 op（`had 3 ops`），但在遇到后续的 `BaseFloat::Slice` 时关闭——Slice 不属于 element-wise 融合。`est_bindings=3/7` 是估算的 binding 计数（已用 3，最大 7）。

```
[explorer] still_optimizing → false after op BaseFloat::Slice (explored 4 ops)
selected single strategy
[plan] exploration completed: lazy, 3/4 ops optimized
```

4 个操作（3 可融合 + 1 Slice），最终 3 个融合。"selected single strategy"——只有一个 Block，ElementWiseFuser 胜出。

```
fusion trace (3 ops, 1 block)
idx  block                                     op                       inputs       outputs    
0    ▸ fused ElementWise (score=420, 3 ops)    NumericFloat::MulScalar  t0:f32[2,2]  t1:f32[2,2]
1                                              NumericFloat::AddScalar  t1:f32[2,2]  t2:f32[2,2]
2                                              Float::Tanh              t2:f32[2,2]  t3:f32[2,2]
```

fusion trace 表显示完整 tensor flow：`t0 → MulScalar → t1 → AddScalar → t2 → Tanh → t3`。三个操作融合为一个 `elemwise_fuse` kernel。

**为什么是 ElementWiseFuser 胜出？** 日志中四个 fuser 都 `had 3 ops`——看起来都参与了竞争。但 `find_best_optimization_index` 的判定是 `ready=true && score > best_score`。`MatmulFuser`、`ReduceFuser`、`ReduceBroadcastedFuser` 对纯 element-wise 序列的 `ready` 都是 `false`——它们根本没参赛。详细解释见 [Fusion 系统设计：OperationFuser 的竞争制](../../docs/burn/kernel-fusion-system-design.md#融合引擎operationfuser-的竞争探索)。

### 阶段三：slice kernel（缓存命中）

```
[plan] cache hit: execute plan #2 (lazy, segment has 1 ops)
fusion trace (1 op, 1 block)  ← Slice kernel，未融合
[plan] cache hit: execute plan #2 (lazy, segment has 1 ops)  ← 第二次 slice
```

`println!("{}", z)` 的内部实现需要多次 slice 来格式化输出——每次直接命中已缓存的 plan #2，不再重新探索。

## 观察要点

1. **冷启动 vs 热路径**：首次 `cargo run` 看到 `[plan] exploration completed`；第二次运行同一 binary 看到 `[plan] cache hit`——`ExecutionPlanStore` 在进程生命期内跨 stream 缓存方案。

2. **四个 fuser 的并行决策**：每条 op 触发四条 `[fuser]` 日志。`had N ops` = 该 fuser 累积接受了多少操作。`closed on X` = 该 fuser 因为遇到 X 类操作而停止接受。

3. **`still_optimizing`**：当所有 fuser 返回 `Closed`，探索终止——`still_optimizing → false`。

4. **对比简洁模式**：`BURN_FUSION_LOG=basic` 只显示 fusion trace 表（最终执行策略），无 `[fuser]`/`[explorer]` 细节。

## 验证点

- 看到 `[plan] exploration completed`（首次）或 `[plan] cache hit`（重复执行）即说明融合引擎正常工作。
- fusion trace 表里出现 `▸ fused ElementWise (... 3 ops)`，三个 element-wise op 合并为一个 `elemwise_fuse` kernel。
- 数值正确性由测试保证：`cargo test` 跑 `fusion_example_produces_expected_shape`，对照 `tanh([[5,7],[9,11]])` 逐元素验证（容差 1e-6）。

