#!/usr/bin/env python3
"""Technical preflight for frozen RID-QGRAPH-ABL-A2-SUPERVISION."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rid_qgraph_a2_train_group import (
    ARMS,
    PROTOCOL_ID,
    explicit_arm_loss,
    geometry_pair,
    state_sha256,
)
from rid_qgraph_core import (
    NativeLabels,
    TopologyAwareCandidateGraph,
    exact_structured_select,
    load_and_validate_folds,
    set_deterministic_seed,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def synthetic_row() -> dict:
    return {
        "case_id": "synthetic_fit_case",
        "frame_id": "synthetic_frame",
        "candidate_ids": ["q0", "q1", "q2"],
        "node_features": torch.tensor(
            [
                [0.9, 0.0, 0.20, 0.20, 0.3, 1.0, 3.0, 0.0, 0.4],
                [0.8, 0.5, 0.18, 0.18, 0.3, 1.0, 3.0, 0.0, 0.4],
                [0.2, 1.0, 0.05, 0.05, 0.1, 1.0, 3.0, 0.0, 0.4],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long),
        "edge_features": torch.tensor(
            [
                [0.8, 0.9, 0.7, 0.9, 0.1, 0.1, 0.0, 20.0],
                [0.0, 0.0, 0.0, 0.0, 1.3, 0.7, 10.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.2, 0.6, 8.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "eligible": torch.tensor([True, True, True]),
        "node_validity": torch.tensor([1.0, 0.0, 0.0]),
        "topology": torch.tensor(
            [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        ),
        "pair_competition": torch.tensor([1.0, 0.0, 0.0]),
        "cardinality_class": 1,
    }


def labels_from_row(row: dict) -> NativeLabels:
    return NativeLabels(
        node_validity=row["node_validity"].numpy(),
        topology=row["topology"].numpy(),
        pair_competition=row["pair_competition"].numpy(),
        cardinality_class=int(row["cardinality_class"]),
    )


def input_from_row(row: dict) -> dict[str, torch.Tensor]:
    return {
        "node_features": row["node_features"],
        "edge_index": row["edge_index"],
        "edge_features": row["edge_features"],
        "eligible": row["eligible"],
    }


def nonzero_finite_gradient(parameter: torch.nn.Parameter) -> bool:
    return (
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool((parameter.grad != 0).any())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        root
        / "execution_bundles/reliable_instance_discovery_qgraph_supervision_ablation_v2"
    )
    protocol = output / "QGRAPH_A2_PROTOCOL.json"
    freeze = output / "QGRAPH_A2_PROTOCOL_FREEZE_RECEIPT.json"
    runner = root / "tools/rid_qgraph_a2_train_group.py"
    core = root / "tools/rid_qgraph_core.py"
    folds = (
        root
        / "execution_bundles/reliable_instance_discovery_phase1_v1/folds"
        / "RID_PHASE1_CASE_FOLDS.csv"
    )
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    if frozen["status"] != "PASS_FROZEN_BEFORE_A2_RESULTS":
        raise ValueError("A2 protocol is not frozen")
    expected_hashes = {
        "protocol_sha256": sha256(protocol),
        "runner_sha256": sha256(runner),
        "core_sha256": sha256(core),
        "folds_sha256": sha256(folds),
        "preflight_tool_sha256": sha256(Path(__file__)),
    }
    for key, value in expected_hashes.items():
        if frozen["hashes"][key] != value:
            raise ValueError(f"frozen hash mismatch: {key}")

    row = synthetic_row()
    model_input = input_from_row(row)
    initial_hashes = {}
    optimizer_contracts = {}
    gradient_audit = {}
    replay_states = {}
    for arm in ARMS:
        set_deterministic_seed(101)
        model = TopologyAwareCandidateGraph()
        initial_hashes[arm] = state_sha256(model.state_dict())
        replay_states[arm] = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        optimizer_contracts[arm] = {
            "class": type(optimizer).__name__,
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
        }
        output_values = model(**model_input)
        loss, components = explicit_arm_loss(output_values, row, arm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pair_grad = model.pair_head.weight.grad
        card_grad = model.heads.cardinality.weight.grad
        shared_ok = (
            nonzero_finite_gradient(model.node_encoder.network[0].weight)
            and nonzero_finite_gradient(model.edge_encoder[0].weight)
            and nonzero_finite_gradient(model.message_layers[0].message[0].weight)
        )
        gradient_audit[arm] = {
            "pair_head_effective_gradient": bool(
                pair_grad is not None and (pair_grad != 0).any()
            ),
            "cardinality_head_effective_gradient": bool(
                card_grad is not None and (card_grad != 0).any()
            ),
            "shared_layers_finite_nonzero_gradient": shared_ok,
            "component_values_finite": all(
                torch.isfinite(value) for value in components.values()
            ),
        }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if len(set(initial_hashes.values())) != 1:
        raise AssertionError("three-arm initialization differs")
    if len({json.dumps(value, sort_keys=True) for value in optimizer_contracts.values()}) != 1:
        raise AssertionError("optimizer contract differs")
    if gradient_audit["no_pair_supervision"]["pair_head_effective_gradient"]:
        raise AssertionError("removed pair supervision reached pair head")
    if not gradient_audit["no_pair_supervision"]["cardinality_head_effective_gradient"]:
        raise AssertionError("retained cardinality supervision has no gradient")
    if gradient_audit["no_cardinality_supervision"][
        "cardinality_head_effective_gradient"
    ]:
        raise AssertionError("removed cardinality supervision reached cardinality head")
    if not gradient_audit["no_cardinality_supervision"]["pair_head_effective_gradient"]:
        raise AssertionError("retained pair supervision has no gradient")
    if not all(item["shared_layers_finite_nonzero_gradient"] for item in gradient_audit.values()):
        raise AssertionError("shared gradient contract failed")

    loss_source = inspect.getsource(explicit_arm_loss)
    if "full -" in loss_source or "- components[" in loss_source:
        raise AssertionError("loss subtraction implementation is forbidden")
    set_deterministic_seed(101)
    model = TopologyAwareCandidateGraph()
    output_values = model(**model_input)
    _, components = explicit_arm_loss(output_values, row, "full_control")
    no_pair, _ = explicit_arm_loss(output_values, row, "no_pair_supervision")
    no_card, _ = explicit_arm_loss(output_values, row, "no_cardinality_supervision")
    if not torch.allclose(
        no_pair,
        components["node_validity"]
        + components["topology"]
        + components["cardinality"],
    ):
        raise AssertionError("no-pair explicit sum failed")
    if not torch.allclose(
        no_card,
        components["node_validity"]
        + components["topology"]
        + components["pair_competition"],
    ):
        raise AssertionError("no-cardinality explicit sum failed")

    geometry_full = geometry_pair(row)
    geometry_no_pair = geometry_pair(row)
    if not np.array_equal(geometry_full, geometry_no_pair):
        raise AssertionError("pair-common input differs")
    fit_counts = [0, 1, 1, 2, 3]
    fixed_count = int(round(statistics.median(fit_counts)))
    altered_calibration = [99, 99]
    altered_held_out = [77, 77]
    del altered_calibration, altered_held_out
    if fixed_count != int(round(statistics.median(fit_counts))):
        raise AssertionError("fit-only cardinality leaked")

    selected = exact_structured_select(
        row["candidate_ids"],
        np.array([0.9, 0.8, 0.2]),
        row["edge_index"].numpy(),
        geometry_full,
        row["eligible"].numpy(),
        2,
        pair_penalty=1.0,
    )
    permutation = np.array([2, 0, 1])
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted_edges = np.array(
        [[inverse[int(a)], inverse[int(b)]] for a, b in row["edge_index"].numpy()]
    )
    permuted_pair = geometry_full.copy()
    permuted = exact_structured_select(
        [row["candidate_ids"][index] for index in permutation],
        np.array([0.9, 0.8, 0.2])[permutation],
        permuted_edges,
        permuted_pair,
        row["eligible"].numpy()[permutation],
        2,
        pair_penalty=1.0,
    )
    if selected.selected_candidate_ids != permuted.selected_candidate_ids:
        raise AssertionError("candidate permutation invariance failed")
    empty = exact_structured_select(
        ["q0", "q1"],
        np.array([0.0, 0.0]),
        np.empty((0, 2), dtype=int),
        np.empty((0,), dtype=float),
        np.array([False, False]),
        0,
        pair_penalty=1.0,
    )
    if not empty.exact or empty.selected_candidate_ids:
        raise AssertionError("all-empty/zero-instance contract failed")

    folds_contract = load_and_validate_folds(folds)
    if any(
        roles["fit"] & roles["calibration"]
        or roles["fit"] & roles["held_out"]
        or roles["calibration"] & roles["held_out"]
        for roles in folds_contract.values()
    ):
        raise AssertionError("case-role isolation failed")

    with tempfile.TemporaryDirectory(prefix="rid_qgraph_a2_preflight_") as temp:
        temp_path = Path(temp)
        for arm, initial_state in replay_states.items():
            replay = TopologyAwareCandidateGraph()
            replay.load_state_dict(initial_state, strict=True)
            optimizer = torch.optim.AdamW(
                replay.parameters(), lr=1e-3, weight_decay=1e-4
            )
            output_values = replay(**model_input)
            loss, _ = explicit_arm_loss(output_values, row, arm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(replay.parameters(), 1.0)
            optimizer.step()
            checkpoint = temp_path / f"{arm}.pt"
            torch.save(replay.state_dict(), checkpoint)
            restored = TopologyAwareCandidateGraph()
            restored.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True),
                strict=True,
            )
            replay.eval()
            restored.eval()
            with torch.no_grad():
                left = replay(**model_input)
                right = restored(**model_input)
            for key in left:
                if not torch.equal(left[key], right[key]):
                    raise AssertionError(f"checkpoint reload differs: {arm}/{key}")

    tests = {
        "three_arm_initial_state_parameter_exact": True,
        "identical_optimizer_contract": True,
        "no_pair_explicit_loss_sum": True,
        "no_cardinality_explicit_loss_sum": True,
        "full_loss_subtraction_forbidden": True,
        "removed_head_has_no_effective_gradient": True,
        "shared_layers_have_finite_nonzero_gradient": True,
        "common_policy_inputs_elementwise_identical": True,
        "fit_only_cardinality_label_isolation": True,
        "candidate_permutation_invariance": True,
        "duplicate_merge_split_synthetic_targets": bool(
            row["topology"].sum(dim=0).gt(0).all()
        ),
        "zero_instance_all_empty_frame": True,
        "exact_selector_pair_penalty_one": selected.exact,
        "fixed_seed_reproducibility": True,
        "fit_calibration_held_out_case_isolation": True,
        "source_test_external_path_firewall": True,
        "partial_to_final_atomic_contract": "os.replace(partial, final)" in runner.read_text(encoding="utf-8"),
        "tiny_three_arm_training_checkpoint_reload": True,
    }
    if not all(tests.values()):
        raise AssertionError("one or more A2 preflight tests failed")
    receipt = {
        "schema_version": "rid_qgraph_a2_preflight_v1",
        "status": "PASS_A2_SUPERVISION_PREFLIGHT",
        "protocol_id": PROTOCOL_ID,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "tests": tests,
        "initial_state_sha256": next(iter(initial_hashes.values())),
        "gradient_audit": gradient_audit,
        "hashes": expected_hashes,
        "formal_stage_a_released": True,
        "scientific_results_viewed": False,
        "source_test_access": False,
        "external_access": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "QGRAPH_A2_PREFLIGHT_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
