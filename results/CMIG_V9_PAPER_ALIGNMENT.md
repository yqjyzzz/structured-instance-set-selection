# CMIG v9 论文对齐说明

更新时间：2026-09-12

本文件记录公开复现仓库与论文 `rewrite_v9_anti_defensive` 的对齐关系。它不新增实验结果，也不替代冻结运行回执。

## 当前论文版本

- 标题：Topology-Aware Query Selection for Surgical Instrument Instance Segmentation
- 论文入口：[arXiv:2608.11607](https://arxiv.org/abs/2608.11607)
- 本地修订：`rewrite_v9_anti_defensive`
- 论文改动类型：局部叙事与段落组织修订
- 代码、数据划分、阈值、seed、估计量和结果：保持不变

## 论文与仓库的对应关系

| 论文内容 | 仓库位置 | 说明 |
|---|---|---|
| 候选图构造、节点和边特征 | `src/structured_instance_selection/rid_qgraph_core.py` | 对应方法中的候选图表示 |
| 图路径与节点特征匹配路径 | `rid_qgraph_core.py` | 对应正式比较的两个完整路径 |
| 消息传递、预测头和集合基数 | `rid_qgraph_core.py` | 展示实现结构，不单独证明组件因果作用 |
| MILP 精确子集选择 | `rid_qgraph_core.py` | 对应最终实例集合构造 |
| 冻结病例划分与评估合同 | `configs/`、`protocol/` | 保持病例级统计和预设判定规则 |
| 主张—证据—估计量 | `results/CLAIM_EVIDENCE_ESTIMAND_SPINE.md` | 对应论文的 claim–evidence spine |
| 结果与边界 | `docs/05_结果与边界.md` | 对应论文 Results、Discussion 和 Supplement 的边界 |

## 结果边界

- 密封源域测试和 ROBUST-MIPS 分别支持指定条件下的完整路径效果。
- Endoscapes 仅一个随机种子满足完整判定条件，不写成稳定或普遍迁移。
- 正式比较不能分离关系特征、消息传递、基数预测、选择器或额外容量的独立贡献。
- 机制诊断是探索性和补充性证据。
- `SAFETY_PRESERVATION_NOT_ESTABLISHED` 表示保持条件未获支持，不等于 `SAFETY_DEGRADATION_SUPPORTED`。

## 发布说明

本次同步更新的是论文入口和仓库文档。代码没有因 v9 改写而改变。arXiv 的线上版本不会因本地文件更新而自动变化；如需发布新版本，应由作者在 arXiv 页面提交并重新确认元数据。
