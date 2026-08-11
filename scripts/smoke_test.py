"""运行不依赖论文数据的端到端冒烟测试。"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structured_instance_selection.demo import run_demo  # noqa: E402


def main() -> None:
    summary = run_demo()
    assert summary.candidate_count == 3
    assert summary.eligible_count == 2
    assert summary.selected_count == 1
    assert summary.selector_status == "PASS_EXACT"
    print("通过：候选图构造")
    print("通过：图路径与节点路径前向传播")
    print("通过：MILP 精确选择")


if __name__ == "__main__":
    main()
