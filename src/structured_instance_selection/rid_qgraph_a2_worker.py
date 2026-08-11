#!/usr/bin/env python3
"""Sequential immutable queue worker for RID-QGRAPH A2 atomic groups."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--discovery-seed", type=int, required=True)
    parser.add_argument("--graph-seed", type=int, required=True)
    parser.add_argument(
        "--task", action="append", required=True, help="outer_fold:arm"
    )
    parser.add_argument("--solver-workers", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    runner = root / "tools/rid_qgraph_a2_train_group.py"
    folds = root / "assets/RID_PHASE1_CASE_FOLDS.csv"
    core = root / "tools/rid_qgraph_core.py"
    preflight = root / "protocol/QGRAPH_A2_PREFLIGHT_RECEIPT.json"
    freeze = root / "protocol/QGRAPH_A2_PROTOCOL_FREEZE_RECEIPT.json"
    completed = []
    for spec in args.task:
        fold_text, arm = spec.split(":", 1)
        fold = int(fold_text)
        group = f"seed{args.discovery_seed}_fold{fold}_g{args.graph_seed}"
        final = args.output_root / arm / group
        partial = final.with_name(final.name + ".partial")
        if final.is_dir():
            completed.append({"task": spec, "status": "SKIP_EXISTING_FINAL"})
            continue
        if partial.exists():
            raise FileExistsError(f"partial requires audit before recovery: {partial}")
        command = [
            sys.executable,
            str(runner),
            "--arm",
            arm,
            "--cache",
            str(args.cache),
            "--folds",
            str(folds),
            "--core",
            str(core),
            "--preflight",
            str(preflight),
            "--protocol-freeze",
            str(freeze),
            "--output-root",
            str(args.output_root),
            "--discovery-seed",
            str(args.discovery_seed),
            "--outer-fold",
            str(fold),
            "--graph-seed",
            str(args.graph_seed),
            "--solver-workers",
            str(args.solver_workers),
        ]
        print(json.dumps({"event": "START", "task": spec, "command": command}), flush=True)
        subprocess.run(command, check=True)
        completed.append({"task": spec, "status": "PASS_FINAL"})
    receipt = {
        "schema_version": "rid_qgraph_a2_worker_v1",
        "status": "PASS_WORKER_QUEUE",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "discovery_seed": args.discovery_seed,
        "graph_seed": args.graph_seed,
        "tasks": completed,
    }
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
