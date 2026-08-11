#!/usr/bin/env python3
"""Checkpoint-free deterministic lane benchmark for A2 cloud release."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rid_qgraph_a2_train_group import explicit_arm_loss, state_sha256
from rid_qgraph_core import TopologyAwareCandidateGraph, set_deterministic_seed
from rid_qgraph_train_group import fit_scaler, load_rows, model_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--arm", default="full_control")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    receipt = json.loads(
        (args.cache / "QGRAPH_CACHE_RECEIPT.json").read_text(encoding="utf-8")
    )
    case = sorted(receipt["case_cache_sha256"])[0].split(".")[0]
    rows = load_rows(args.cache, {case})[: args.frames]
    scaler = fit_scaler(rows)
    device = torch.device(args.device)
    set_deterministic_seed(101)
    model = TopologyAwareCandidateGraph().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    started = time.perf_counter()
    for step in range(args.steps):
        row = rows[step % len(rows)]
        output = model(**model_inputs(row, scaler, device))
        loss, _ = explicit_arm_loss(output, row, args.arm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "PASS_A2_LANE_BENCHMARK",
                "arm": args.arm,
                "steps": args.steps,
                "frames": len(rows),
                "seconds": time.perf_counter() - started,
                "final_state_sha256": state_sha256(model.state_dict()),
                "formal_training_started": False,
                "checkpoint_written": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
