#!/usr/bin/env python3
"""Train one frozen RID-QGRAPH-ABL-A2-SUPERVISION atomic arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rid_qgraph_core import (
    EDGE_FEATURES,
    FREEZE_ID,
    TopologyAwareCandidateGraph,
    load_and_validate_folds,
    multitask_loss,
    set_deterministic_seed,
)
from rid_qgraph_train_group import (
    add_final_metrics,
    case_balanced_f1,
    fit_scaler,
    labels,
    load_rows,
    model_inputs,
    solve_task,
    write_csv,
)


PROTOCOL_ID = "RID-QGRAPH-ABL-A2-SUPERVISION"
ARMS = ("full_control", "no_pair_supervision", "no_cardinality_supervision")
POLICY_PAIR = "full_pair_common"
POLICY_CARD = "full_cardinality_common"
PAIR_FEATURE_INDICES = tuple(
    EDGE_FEATURES.index(name)
    for name in (
        "pair_iou",
        "left_contained_by_right",
        "right_contained_by_left",
        "overlap_coefficient",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def explicit_arm_loss(
    output: dict[str, torch.Tensor], row: dict, arm: str
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build each arm loss explicitly; subtraction from full loss is forbidden."""
    _, components = multitask_loss(output, labels(row))
    if arm == "full_control":
        loss = (
            components["node_validity"]
            + components["topology"]
            + components["pair_competition"]
            + components["cardinality"]
        )
    elif arm == "no_pair_supervision":
        loss = (
            components["node_validity"]
            + components["topology"]
            + components["cardinality"]
        )
    elif arm == "no_cardinality_supervision":
        loss = (
            components["node_validity"]
            + components["topology"]
            + components["pair_competition"]
        )
    else:
        raise ValueError(arm)
    return loss, components


def geometry_pair(row: dict) -> np.ndarray:
    edge = row["edge_features"].detach().cpu().numpy()
    if len(edge) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.clip(edge[:, PAIR_FEATURE_INDICES].max(axis=1), 0.0, 1.0)


def policy_names(arm: str) -> tuple[str, ...]:
    if arm == "full_control":
        return (POLICY_PAIR, POLICY_CARD)
    if arm == "no_pair_supervision":
        return (POLICY_PAIR,)
    if arm == "no_cardinality_supervision":
        return (POLICY_CARD,)
    raise ValueError(arm)


@torch.no_grad()
def predict_policy(
    model: torch.nn.Module,
    rows: list[dict],
    scaler: dict[str, torch.Tensor],
    device: torch.device,
    executor: ProcessPoolExecutor,
    policy: str,
    fit_only_cardinality: int,
) -> list[dict]:
    model.eval()
    tasks = []
    for row in rows:
        output = model(**model_inputs(row, scaler, device))
        eligible_count = int(row["eligible"].sum())
        if policy == POLICY_PAIR:
            pair_probabilities = geometry_pair(row)
            requested = min(
                int(output["cardinality_logits"].argmax().item()), eligible_count
            )
        elif policy == POLICY_CARD:
            pair_probabilities = (
                output["pair_competition_logits"].sigmoid().cpu().numpy()
            )
            requested = min(int(fit_only_cardinality), eligible_count)
        else:
            raise ValueError(policy)
        tasks.append(
            {
                "case_id": row["case_id"],
                "frame_id": row["frame_id"],
                "candidate_ids": row["candidate_ids"],
                "node_logits": output["node_validity_logits"].cpu().numpy(),
                "edge_index": row["edge_index"].numpy(),
                "pair_probabilities": pair_probabilities,
                "eligible": row["eligible"].numpy(),
                "cardinality": requested,
                "gt_iou": row["gt_iou"].numpy(),
                "gt_count": int(row["gt_count"]),
                "topology": row["topology"].numpy(),
            }
        )
    return list(executor.map(solve_task, tasks, chunksize=1))


def train_arm(
    arm: str,
    fit_rows: list[dict],
    calibration_rows: list[dict],
    scaler: dict[str, torch.Tensor],
    graph_seed: int,
    device: torch.device,
    executor: ProcessPoolExecutor,
    epoch_log: Path,
    fit_only_cardinality: int,
) -> tuple[
    str,
    dict[str, dict[str, torch.Tensor]],
    dict[str, int],
    dict[str, float],
]:
    set_deterministic_seed(graph_seed)
    model = TopologyAwareCandidateGraph().to(device)
    initial_hash = state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    policies = policy_names(arm)
    best_epochs = {policy: 0 for policy in policies}
    best_objectives = {policy: -math.inf for policy in policies}
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    with epoch_log.open("a", encoding="utf-8") as handle:
        for epoch in range(1, 31):
            model.train()
            order = list(range(len(fit_rows)))
            random.Random(graph_seed * 10_000 + epoch).shuffle(order)
            loss_sum = 0.0
            started = time.perf_counter()
            for index in order:
                row = fit_rows[index]
                output = model(**model_inputs(row, scaler, device))
                loss, _ = explicit_arm_loss(output, row, arm)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
            objectives = {}
            flags = {}
            for policy in policies:
                calibration = predict_policy(
                    model,
                    calibration_rows,
                    scaler,
                    device,
                    executor,
                    policy,
                    fit_only_cardinality,
                )
                objective = case_balanced_f1(calibration)
                objectives[policy] = objective
                flags[policy] = sum(not row["exact"] for row in calibration)
                # Strictly greater preserves the earliest best checkpoint.
                if objective > best_objectives[policy]:
                    best_objectives[policy] = objective
                    best_epochs[policy] = epoch
                    best_states[policy] = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            event = {
                "event": "a2_epoch_complete",
                "protocol_id": PROTOCOL_ID,
                "arm": arm,
                "epoch": epoch,
                "fit_loss_mean": loss_sum / len(fit_rows),
                "calibration_case_balanced_instance_f1": objectives,
                "calibration_flag_count": flags,
                "best_epochs": best_epochs,
                "seconds": time.perf_counter() - started,
            }
            handle.write(json.dumps(event) + "\n")
            handle.flush()
            print(json.dumps(event), flush=True)
    if set(best_states) != set(policies):
        raise RuntimeError("checkpoint selection failed")
    return initial_hash, best_states, best_epochs, best_objectives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--discovery-seed", type=int, choices=(17, 23, 41), required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--graph-seed", type=int, choices=(101, 103, 107), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solver-workers", type=int, default=16)
    args = parser.parse_args()

    cache_receipt_path = args.cache / "QGRAPH_CACHE_RECEIPT.json"
    cache_receipt = json.loads(cache_receipt_path.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    frozen = json.loads(args.protocol_freeze.read_text(encoding="utf-8"))
    if (
        cache_receipt["status"] != "PASS_ATOMIC_CACHE"
        or cache_receipt["discovery_seed"] != args.discovery_seed
        or cache_receipt["input_sha256"]["core"] != sha256(args.core)
    ):
        raise ValueError("cache/core/discovery-seed mismatch")
    if preflight["status"] != "PASS_A2_SUPERVISION_PREFLIGHT":
        raise ValueError("A2 preflight not passed")
    if frozen["status"] != "PASS_FROZEN_BEFORE_A2_RESULTS":
        raise ValueError("A2 protocol not frozen")
    if frozen["hashes"]["runner_sha256"] != sha256(Path(__file__)):
        raise ValueError("A2 runner hash mismatch")

    folds = load_and_validate_folds(args.folds)
    roles = folds[args.outer_fold]
    group = f"seed{args.discovery_seed}_fold{args.outer_fold}_g{args.graph_seed}"
    final = args.output_root / args.arm / group
    partial = final.with_name(final.name + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError(f"atomic target exists: {final}")
    partial.mkdir(parents=True)

    fit_rows = load_rows(args.cache, roles["fit"])
    calibration_rows = load_rows(args.cache, roles["calibration"])
    held_rows = load_rows(args.cache, roles["held_out"])
    scaler = fit_scaler(fit_rows)
    torch.save(scaler, partial / "FIT_ROLE_STANDARDIZER.pt")
    fit_only_cardinality = int(
        round(statistics.median(int(row["gt_count"]) for row in fit_rows))
    )
    device = torch.device(args.device)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.solver_workers, mp_context=context
    ) as executor:
        initial_hash, states, epochs, objectives = train_arm(
            args.arm,
            fit_rows,
            calibration_rows,
            scaler,
            args.graph_seed,
            device,
            executor,
            partial / "EPOCH_LOG.jsonl",
            fit_only_cardinality,
        )
        outputs = {}
        for policy, state in states.items():
            checkpoint = partial / f"{policy}_best.pt"
            torch.save(
                {
                    "freeze_id": FREEZE_ID,
                    "protocol_id": PROTOCOL_ID,
                    "arm": args.arm,
                    "policy": policy,
                    "discovery_seed": args.discovery_seed,
                    "outer_fold": args.outer_fold,
                    "graph_seed": args.graph_seed,
                    "best_epoch": epochs[policy],
                    "calibration_objective": objectives[policy],
                    "fit_only_cardinality": fit_only_cardinality,
                    "initial_state_sha256": initial_hash,
                    "state_dict": state,
                },
                checkpoint,
            )
            model = TopologyAwareCandidateGraph()
            model.load_state_dict(state, strict=True)
            model.to(device)
            predictions = predict_policy(
                model,
                held_rows,
                scaler,
                device,
                executor,
                policy,
                fit_only_cardinality,
            )
            predictions = add_final_metrics(
                predictions, args.cache, roles["held_out"]
            )
            prediction_path = partial / f"{policy}_held_out.csv"
            write_csv(prediction_path, predictions)
            outputs[policy] = {
                "best_epoch": epochs[policy],
                "calibration_objective": objectives[policy],
                "checkpoint_sha256": sha256(checkpoint),
                "held_out_sha256": sha256(prediction_path),
                "held_out_frames": len(predictions),
                "solver_flags": sum(not row["exact"] for row in predictions),
            }

    receipt = {
        "schema_version": "rid_qgraph_a2_fit_group_v1",
        "status": "PASS_ATOMIC_A2_GROUP",
        "protocol_id": PROTOCOL_ID,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "group": group,
        "discovery_seed": args.discovery_seed,
        "outer_fold": args.outer_fold,
        "graph_seed": args.graph_seed,
        "initial_state_sha256": initial_hash,
        "roles": {name: sorted(values) for name, values in roles.items()},
        "fit_frames": len(fit_rows),
        "calibration_frames": len(calibration_rows),
        "held_out_frames": len(held_rows),
        "fit_only_cardinality": fit_only_cardinality,
        "policies": outputs,
        "loss_components": {
            "full_control": ["node_validity", "topology", "pair_competition", "cardinality"],
            "no_pair_supervision": ["node_validity", "topology", "cardinality"],
            "no_cardinality_supervision": ["node_validity", "topology", "pair_competition"],
        }[args.arm],
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clipping": 1.0,
            "epochs": 30,
        },
        "hashes": {
            "cache_receipt": sha256(cache_receipt_path),
            "folds": sha256(args.folds),
            "core": sha256(args.core),
            "runner": sha256(Path(__file__)),
            "preflight": sha256(args.preflight),
            "protocol_freeze": sha256(args.protocol_freeze),
        },
        "source_development_only": True,
        "source_test_access": False,
        "external_access": False,
    }
    receipt_path = partial / "QGRAPH_A2_FIT_GROUP_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, final)
    print(
        json.dumps(
            {"status": receipt["status"], "arm": args.arm, "group": group}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
