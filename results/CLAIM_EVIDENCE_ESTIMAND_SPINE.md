# 主张—证据—估计量主干

本文件保存论文主张与证据边界的简表。正式名称和工作流标签保持原样，其余说明使用中文。

| 主张 | 目标估计量 | 证据 | 允许结论 |
|---|---|---|---|
| 完整路径改善最终实例集构建 | `complete relational path` 减去 `node-feature-matched path` | 密封源域测试、ROBUST-MIPS | 指定条件下的完整路径效果 |
| 外部表现依赖数据域 | 每个数据域单独应用冻结判定条件 | ROBUST-MIPS、Endoscapes | 各域的直接迁移结果 |
| 机制解释 | 探索性诊断对照 | 补充材料与诊断记录 | 结果与基数通路贡献一致，但不证明单一机制 |
| 安全行为 | 非补偿安全标签 | 病例级核查与冻结门槛 | `SAFETY_PRESERVATION_NOT_ESTABLISHED` 不等于 `SAFETY_DEGRADATION_SUPPORTED` |

## 工作流标签

以下标签只用于项目追踪，不直接写入论文正文：

- `PASS_SOURCE_TEST_GRAPH_CONFIRMATION`
- `REVISE_DOMAIN_TRANSFER_UNSTABLE`
- `NO_SUPPORT_MULTI_DOMAIN_HELDOUT_ROBUSTNESS`
- `SAFETY_PRESERVATION_NOT_ESTABLISHED`
- `SAFETY_DEGRADATION_SUPPORTED`

