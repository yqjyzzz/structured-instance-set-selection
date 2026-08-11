#!/usr/bin/env python3
"""Run one frozen RID-QGRAPH-M1 fold/seed group atomically."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from rid_qgraph_core import (
    FREEZE_ID,
    NativeLabels,
    NodeOnlyMLP,
    TopologyAwareCandidateGraph,
    exact_structured_select,
    load_and_validate_folds,
    multitask_loss,
    set_deterministic_seed,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_rows(cache: Path, cases: set[str], suffix: str = "train") -> list[dict]:
    rows = []
    for case_id in sorted(cases):
        rows.extend(
            torch.load(
                cache / f"{case_id}.{suffix}.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
    return rows


def fit_scaler(rows: list[dict]) -> dict[str, torch.Tensor]:
    node_sum = torch.zeros(9, dtype=torch.float64)
    node_square = torch.zeros(9, dtype=torch.float64)
    node_count = 0
    edge_sum = torch.zeros(8, dtype=torch.float64)
    edge_square = torch.zeros(8, dtype=torch.float64)
    edge_count = 0
    for row in rows:
        nodes = row["node_features"].to(torch.float64)
        edges = row["edge_features"].to(torch.float64)
        node_sum += nodes.sum(0)
        node_square += (nodes * nodes).sum(0)
        node_count += len(nodes)
        if len(edges):
            edge_sum += edges.sum(0)
            edge_square += (edges * edges).sum(0)
            edge_count += len(edges)
    node_mean = node_sum / node_count
    node_var = (node_square / node_count - node_mean.square()).clamp_min(0)
    edge_mean = edge_sum / edge_count
    edge_var = (edge_square / edge_count - edge_mean.square()).clamp_min(0)
    node_std = node_var.sqrt()
    edge_std = edge_var.sqrt()
    node_std[node_std == 0] = 1
    edge_std[edge_std == 0] = 1
    return {
        "node_mean": node_mean.to(torch.float32),
        "node_std": node_std.to(torch.float32),
        "edge_mean": edge_mean.to(torch.float32),
        "edge_std": edge_std.to(torch.float32),
        "node_count": torch.tensor(node_count),
        "edge_count": torch.tensor(edge_count),
    }


def model_inputs(
    row: dict, scaler: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "node_features": (
            (row["node_features"] - scaler["node_mean"]) / scaler["node_std"]
        ).to(device, non_blocking=True),
        "edge_index": row["edge_index"].to(device, non_blocking=True),
        "edge_features": (
            (row["edge_features"] - scaler["edge_mean"]) / scaler["edge_std"]
        ).to(device, non_blocking=True),
        "eligible": row["eligible"].to(device, non_blocking=True),
    }


def labels(row: dict) -> NativeLabels:
    return NativeLabels(
        node_validity=row["node_validity"].numpy(),
        topology=row["topology"].numpy(),
        pair_competition=row["pair_competition"].numpy(),
        cardinality_class=int(row["cardinality_class"]),
    )


def native_counts(
    selected: tuple[int, ...], gt_iou: np.ndarray, gt_count: int
) -> tuple[int, int, int, float]:
    if selected and gt_count:
        subset = gt_iou[np.asarray(selected, dtype=int)]
        pred_indices, target_indices = linear_sum_assignment(-subset)
        tp = int((subset[pred_indices, target_indices] >= 0.5).sum())
    else:
        tp = 0
    fp = len(selected) - tp
    fn = gt_count - tp
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 1.0
    return tp, fp, fn, f1


def solve_task(task: dict) -> dict:
    result = exact_structured_select(
        task["candidate_ids"],
        task["node_logits"],
        task["edge_index"],
        task["pair_probabilities"],
        task["eligible"],
        task["cardinality"],
    )
    if result.exact:
        tp, fp, fn, f1 = native_counts(
            result.selected_indices, task["gt_iou"], task["gt_count"]
        )
    else:
        tp, fp, fn, f1 = 0, 0, task["gt_count"], 0.0
    return {
        "case_id": task["case_id"],
        "frame_id": task["frame_id"],
        "status": result.status,
        "exact": result.exact,
        "selected_indices": result.selected_indices,
        "selected_ids": result.selected_candidate_ids,
        "predicted_cardinality": task["cardinality"],
        "gt_count": task["gt_count"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "instance_f1": f1,
        "topology": task["topology"],
    }


@torch.no_grad()
def predict_and_select(
    model: torch.nn.Module,
    rows: list[dict],
    scaler: dict[str, torch.Tensor],
    device: torch.device,
    executor: ProcessPoolExecutor,
) -> list[dict]:
    model.eval()
    tasks = []
    for row in rows:
        output = model(**model_inputs(row, scaler, device))
        requested = int(output["cardinality_logits"].argmax().item())
        requested = min(requested, int(row["eligible"].sum()))
        tasks.append(
            {
                "case_id": row["case_id"],
                "frame_id": row["frame_id"],
                "candidate_ids": row["candidate_ids"],
                "node_logits": output["node_validity_logits"].cpu().numpy(),
                "edge_index": row["edge_index"].numpy(),
                "pair_probabilities": output[
                    "pair_competition_logits"
                ].sigmoid().cpu().numpy(),
                "eligible": row["eligible"].numpy(),
                "cardinality": requested,
                "gt_iou": row["gt_iou"].numpy(),
                "gt_count": int(row["gt_count"]),
                "topology": row["topology"].numpy(),
            }
        )
    return list(executor.map(solve_task, tasks, chunksize=1))


def case_balanced_f1(results: list[dict]) -> float:
    by_case: dict[str, list[float]] = {}
    for row in results:
        by_case.setdefault(row["case_id"], []).append(float(row["instance_f1"]))
    return float(np.mean([np.mean(values) for values in by_case.values()]))


def train_path(
    model_name: str,
    fit_rows: list[dict],
    calibration_rows: list[dict],
    scaler: dict[str, torch.Tensor],
    graph_seed: int,
    device: torch.device,
    executor: ProcessPoolExecutor,
    epoch_log: Path,
) -> tuple[dict[str, torch.Tensor], int, float]:
    set_deterministic_seed(graph_seed)
    if model_name == "graph":
        model = TopologyAwareCandidateGraph()
    elif model_name == "node_only":
        model = NodeOnlyMLP()
    else:
        raise ValueError(model_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_epoch = 0
    best_objective = -math.inf
    best_state = None
    with epoch_log.open("a", encoding="utf-8") as log_handle:
        for epoch in range(1, 31):
            model.train()
            order = list(range(len(fit_rows)))
            random.Random(graph_seed * 10_000 + epoch).shuffle(order)
            loss_sum = 0.0
            started = time.perf_counter()
            for index in order:
                row = fit_rows[index]
                output = model(**model_inputs(row, scaler, device))
                loss, _ = multitask_loss(output, labels(row))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
            calibration = predict_and_select(
                model, calibration_rows, scaler, device, executor
            )
            objective = case_balanced_f1(calibration)
            if objective > best_objective:
                best_objective = objective
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            event = {
                "event": "epoch_complete",
                "model": model_name,
                "epoch": epoch,
                "fit_loss_mean": loss_sum / len(fit_rows),
                "calibration_case_balanced_instance_f1": objective,
                "calibration_flag_count": sum(not row["exact"] for row in calibration),
                "best_epoch": best_epoch,
                "seconds": time.perf_counter() - started,
            }
            log_handle.write(json.dumps(event) + "\n")
            log_handle.flush()
            print(json.dumps(event), flush=True)
    if best_state is None:
        raise RuntimeError("checkpoint selection failed")
    return best_state, best_epoch, best_objective


def unpack_masks(values: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    flat = np.unpackbits(values.numpy(), axis=1, bitorder="little")
    return flat[:, : shape[0] * shape[1]].reshape((-1, *shape)).astype(bool)


def add_final_metrics(
    results: list[dict], cache: Path, held_cases: set[str]
) -> list[dict]:
    mask_lookup = {}
    for row in load_rows(cache, held_cases, suffix="masks"):
        mask_lookup[(row["case_id"], row["frame_id"])] = row
    enriched = []
    for result in results:
        mask_row = mask_lookup[(result["case_id"], result["frame_id"])]
        shape = tuple(mask_row["mask_shape"])
        candidates = unpack_masks(mask_row["packed_masks"], shape)
        targets = unpack_masks(mask_row["packed_gt_masks"], shape)
        if result["exact"] and result["selected_indices"]:
            prediction_union = candidates[list(result["selected_indices"])].any(0)
        else:
            prediction_union = np.zeros(shape, dtype=bool)
        target_union = targets.any(0) if len(targets) else np.zeros(shape, dtype=bool)
        intersection = int((prediction_union & target_union).sum())
        denominator = int(prediction_union.sum() + target_union.sum())
        union_dice = 2 * intersection / denominator if denominator else 1.0
        selected = list(result["selected_indices"]) if result["exact"] else []
        topology = result.pop("topology")
        result.update(
            {
                "union_dice": union_dice,
                "positive_frame": result["gt_count"] > 0,
                "set_failure": bool(
                    result["gt_count"] > 0
                    and not (
                        result["tp"] == result["gt_count"]
                        and result["fp"] == 0
                        and result["fn"] == 0
                    )
                ),
                "empty_frame_false_activation": bool(
                    result["gt_count"] == 0 and len(selected) > 0
                ),
                "duplicate_selected": int(topology[selected, 0].sum())
                if selected
                else 0,
                "merge_selected": int(topology[selected, 1].sum())
                if selected
                else 0,
                "split_selected": int(topology[selected, 2].sum())
                if selected
                else 0,
            }
        )
        enriched.append(result)
    return enriched


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "case_id",
        "frame_id",
        "status",
        "exact",
        "selected_ids",
        "predicted_cardinality",
        "gt_count",
        "tp",
        "fp",
        "fn",
        "instance_f1",
        "union_dice",
        "positive_frame",
        "set_failure",
        "empty_frame_false_activation",
        "duplicate_selected",
        "merge_selected",
        "split_selected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["selected_ids"] = json.dumps(row["selected_ids"])
            encoded.pop("selected_indices", None)
            writer.writerow({field: encoded[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--discovery-seed", type=int, choices=(17, 23, 41), required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--graph-seed", type=int, choices=(101, 103, 107), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solver-workers", type=int, default=24)
    args = parser.parse_args()

    cache_receipt_path = args.cache / "QGRAPH_CACHE_RECEIPT.json"
    cache_receipt = json.loads(cache_receipt_path.read_text(encoding="utf-8"))
    if (
        cache_receipt["status"] != "PASS_ATOMIC_CACHE"
        or cache_receipt["discovery_seed"] != args.discovery_seed
        or cache_receipt["input_sha256"]["core"] != sha256(args.core)
    ):
        raise ValueError("cache/core/discovery-seed contract mismatch")
    frozen = json.loads(args.execution_freeze.read_text(encoding="utf-8"))
    if frozen["status"] != "FROZEN_BEFORE_FIRST_FORMAL_FIT":
        raise ValueError("execution clarification not frozen")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if (
        preflight["status"] != "PASS_PREFLIGHT"
        or not preflight["formal_training_authorized_after_preflight"]
        or preflight["assets"]["core_tool_sha256"] != sha256(args.core)
    ):
        raise ValueError("controlling preflight/core authorization mismatch")
    folds = load_and_validate_folds(args.folds)
    roles = folds[args.outer_fold]
    group = (
        f"seed{args.discovery_seed}_fold{args.outer_fold}_g{args.graph_seed}"
    )
    final = args.output_root / group
    partial = args.output_root / f"{group}.partial"
    if final.exists() or partial.exists():
        raise FileExistsError("atomic formal group target already exists")
    partial.mkdir(parents=True)

    fit_rows = load_rows(args.cache, roles["fit"])
    calibration_rows = load_rows(args.cache, roles["calibration"])
    held_rows = load_rows(args.cache, roles["held_out"])
    scaler = fit_scaler(fit_rows)
    torch.save(scaler, partial / "FIT_ROLE_STANDARDIZER.pt")
    epoch_log = partial / "EPOCH_LOG.jsonl"
    device = torch.device(args.device)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.solver_workers, mp_context=context
    ) as executor:
        selected = {}
        for model_name in ("graph", "node_only"):
            state, epoch, objective = train_path(
                model_name,
                fit_rows,
                calibration_rows,
                scaler,
                args.graph_seed,
                device,
                executor,
                epoch_log,
            )
            checkpoint = partial / f"{model_name}_best.pt"
            torch.save(
                {
                    "freeze_id": FREEZE_ID,
                    "discovery_seed": args.discovery_seed,
                    "outer_fold": args.outer_fold,
                    "graph_seed": args.graph_seed,
                    "model": model_name,
                    "best_epoch": epoch,
                    "calibration_objective": objective,
                    "state_dict": state,
                },
                checkpoint,
            )
            model = (
                TopologyAwareCandidateGraph()
                if model_name == "graph"
                else NodeOnlyMLP()
            )
            model.load_state_dict(state)
            model.to(device)
            predictions = predict_and_select(
                model, held_rows, scaler, device, executor
            )
            predictions = add_final_metrics(predictions, args.cache, roles["held_out"])
            prediction_path = partial / f"{model_name}_held_out.csv"
            write_csv(prediction_path, predictions)
            selected[model_name] = {
                "best_epoch": epoch,
                "calibration_objective": objective,
                "checkpoint_sha256": sha256(checkpoint),
                "held_out_sha256": sha256(prediction_path),
                "held_out_frames": len(predictions),
                "held_out_cases": sorted(roles["held_out"]),
                "solver_flags": sum(not row["exact"] for row in predictions),
            }
    receipt = {
        "schema_version": "rid_qgraph_fit_group_v1",
        "status": "PASS_ATOMIC_FORMAL_GROUP",
        "freeze_id": FREEZE_ID,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "group": group,
        "discovery_seed": args.discovery_seed,
        "outer_fold": args.outer_fold,
        "graph_seed": args.graph_seed,
        "roles": {name: sorted(values) for name, values in roles.items()},
        "fit_frames": len(fit_rows),
        "calibration_frames": len(calibration_rows),
        "held_out_frames": len(held_rows),
        "models": selected,
        "input_sha256": {
            "cache_receipt": sha256(cache_receipt_path),
            "folds": sha256(args.folds),
            "core": sha256(args.core),
            "training_runner": sha256(Path(__file__)),
            "preflight": sha256(args.preflight),
            "execution_freeze": sha256(args.execution_freeze),
        },
        "source_development_only": True,
        "source_test_access": False,
        "external_access": False,
    }
    receipt_path = partial / "QGRAPH_FIT_GROUP_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.replace(partial, final)
    print(json.dumps({"status": receipt["status"], "group": group}), flush=True)


if __name__ == "__main__":
    main()
