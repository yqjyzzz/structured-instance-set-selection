"""输出冻结协议摘要。"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "protocol/QGRAPH_A2_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = protocol["frozen_design"]
    inference = protocol["final_inference"]
    print(f"协议编号：{protocol['protocol_id']}")
    print(f"数据集：{protocol['data_contract']['dataset']}")
    print(f"数据划分：{protocol['data_contract']['split']}")
    print(f"完整候选图：{frozen['full_candidate_graph']}")
    print(f"优化器：{frozen['optimizer']}")
    print(f"bootstrap 次数：{inference['bootstrap_replicates']}")
    print(f"bootstrap 随机种子：{inference['bootstrap_seed']}")
    print("外部结果：各数据域单独报告，不进行合并")


if __name__ == "__main__":
    main()
