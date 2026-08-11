"""依次运行仓库校验、端到端示例和测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    ("仓库结构校验", [sys.executable, "scripts/verify_release.py"]),
    ("端到端冒烟测试", [sys.executable, "scripts/smoke_test.py"]),
    ("自动化测试", [sys.executable, "-m", "pytest", "-q"]),
)


def main() -> int:
    for label, command in COMMANDS:
        print(f"\n[{label}]", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            print(f"失败：{label}", flush=True)
            return completed.returncode
    print("\n全部检查通过", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
