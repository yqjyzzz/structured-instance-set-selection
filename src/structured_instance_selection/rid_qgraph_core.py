#!/usr/bin/env python3
"""Frozen RID-QGRAPH-M1 graph, model, selector, and isolation infrastructure."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
)
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix
from torch import nn


FREEZE_ID = "RID-QGRAPH-M1"
ALLOWED_DATASET = "CholecInstanceSeg"
ALLOWED_RUN_IDS = {
    "source_development_candidates",
    "pr_m2f_source_val_18c_4246f_v001",
}
MASK_THRESHOLD = 0.5
PAIR_PENALTY = 1.0
NODE_FEATURES = (
    "tool_score",
    "normalized_emitted_rank",
    "mask_area",
    "area_ratio",
    "perimeter_pixels",
    "component_count",
    "frame_candidate_count",
    "empty_query_count",
    "union_area",
)
EDGE_FEATURES = (
    "pair_iou",
    "left_contained_by_right",
    "right_contained_by_left",
    "overlap_coefficient",
    "absolute_log_area_ratio",
    "absolute_score_margin",
    "boundary_distance_pixels",
    "contact_pixels",
)
TOPOLOGY_TARGETS = ("duplicate", "merge_associated", "split_associated")
FORBIDDEN_PATH_TOKENS = (
    "source-test",
    "source_test",
    "endoscapes",
    "robust-mips",
    "robust_mips",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    emitted_rank: int
    score: float
    mask: np.ndarray


@dataclass
class CandidateGraph:
    case_id: str
    frame_id: str
    candidate_ids: list[str]
    emitted_ranks: np.ndarray
    scores: np.ndarray
    masks: list[np.ndarray]
    eligible: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    audit: dict[str, object]


@dataclass(frozen=True)
class NativeLabels:
    node_validity: np.ndarray
    topology: np.ndarray
    pair_competition: np.ndarray
    cardinality_class: int


@dataclass(frozen=True)
class SelectionResult:
    status: str
    selected_candidate_ids: tuple[str, ...]
    selected_indices: tuple[int, ...]
    requested_cardinality: int
    eligible_count: int
    objective: float | None
    exact: bool


def assert_source_development_path(path: Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in normalized:
            raise PermissionError(f"prohibited population path: {token}")


def decode_uncompressed_rle(value: str | dict[str, object]) -> np.ndarray:
    rle = json.loads(value) if isinstance(value, str) else value
    height, width = map(int, rle["size"])
    counts = rle["counts"]
    if not isinstance(counts, list):
        raise ValueError("RID-QGRAPH-M1 accepts only uncompressed RLE")
    flat = np.zeros(height * width, dtype=bool)
    cursor = 0
    foreground = False
    for count in counts:
        end = cursor + int(count)
        if foreground:
            flat[cursor:end] = True
        cursor = end
        foreground = not foreground
    if cursor != flat.size:
        raise ValueError("RLE count does not match declared size")
    return flat.reshape((height, width), order="F")


def encode_uncompressed_rle(mask: np.ndarray) -> dict[str, object]:
    counts: list[int] = []
    state = 0
    run = 0
    for value in np.asarray(mask, dtype=bool).reshape(-1, order="F").astype(int):
        if int(value) == state:
            run += 1
        else:
            counts.append(run)
            run = 1
            state = int(value)
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def iter_candidate_frames(
    path: Path,
    *,
    allowed_cases: set[str] | None = None,
    max_frames: int | None = None,
) -> Iterable[tuple[str, str, list[Candidate]]]:
    assert_source_development_path(path)
    current_key: tuple[str, str] | None = None
    current: list[Candidate] = []
    emitted = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != ALLOWED_DATASET or row["run_id"] not in ALLOWED_RUN_IDS:
                raise PermissionError("candidate row is outside frozen source-development role")
            if allowed_cases is not None and row["case_id"] not in allowed_cases:
                raise PermissionError(f"case outside frozen allowlist: {row['case_id']}")
            key = (row["case_id"], row["frame_id"])
            if current_key is not None and key != current_key:
                yield (*current_key, current)
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    return
                current = []
            current_key = key
            current.append(
                Candidate(
                    candidate_id=row["prediction_id"],
                    emitted_rank=int(row["candidate_rank"]),
                    score=float(row["score"]),
                    mask=decode_uncompressed_rle(row["mask_rle_or_relative_reference"]),
                )
            )
        if current_key is not None and current:
            yield (*current_key, current)


def load_candidate_frames(
    path: Path,
    *,
    allowed_cases: set[str] | None = None,
    max_frames: int | None = None,
) -> list[tuple[str, str, list[Candidate]]]:
    return list(
        iter_candidate_frames(
            path,
            allowed_cases=allowed_cases,
            max_frames=max_frames,
        )
    )


def _mask_geometry(
    mask: np.ndarray,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    area = int(mask.sum())
    if not area:
        empty = np.zeros_like(mask, dtype=bool)
        return 0, 0, 0, empty, np.full(mask.shape, np.inf), empty
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    components = int(label(mask, structure=np.ones((3, 3), dtype=int))[1])
    distance = distance_transform_edt(~boundary)
    dilated = binary_dilation(boundary)
    return area, int(boundary.sum()), components, boundary, distance, dilated


def build_candidate_graph(
    case_id: str,
    frame_id: str,
    candidates: Sequence[Candidate],
    *,
    require_100_queries: bool = True,
) -> CandidateGraph:
    ordered = sorted(candidates, key=lambda item: (item.emitted_rank, item.candidate_id))
    if require_100_queries and len(ordered) != 100:
        raise ValueError(f"expected 100 emitted queries, found {len(ordered)}")
    ids = [item.candidate_id for item in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate_id")
    ranks = np.asarray([item.emitted_rank for item in ordered], dtype=int)
    if require_100_queries and set(ranks.tolist()) != set(range(1, 101)):
        raise ValueError("candidate ranks must be exactly 1..100")
    masks = [np.asarray(item.mask, dtype=bool) for item in ordered]
    if masks and any(mask.shape != masks[0].shape for mask in masks):
        raise ValueError("inconsistent mask dimensions")
    scores = np.asarray([item.score for item in ordered], dtype=float)
    geometry = [_mask_geometry(mask) for mask in masks]
    areas = np.asarray([item[0] for item in geometry], dtype=int)
    eligible = areas > 0
    empty_count = int((~eligible).sum())
    union = np.zeros_like(masks[0], dtype=bool) if masks else np.zeros((0, 0), dtype=bool)
    for index in np.flatnonzero(eligible):
        union |= masks[int(index)]
    union_area = int(union.sum())
    mask_size = int(masks[0].size) if masks else 1
    denominator = max(1, len(ordered) - 1)
    node_rows = []
    for index, item in enumerate(ordered):
        area, perimeter, components, _, _, _ = geometry[index]
        node_rows.append(
            [
                item.score,
                (item.emitted_rank - 1) / denominator,
                area,
                area / mask_size,
                perimeter,
                components,
                len(ordered),
                empty_count,
                union_area,
            ]
        )
    edge_pairs: list[tuple[int, int]] = []
    edge_rows: list[list[float]] = []
    eligible_indices = [int(index) for index in np.flatnonzero(eligible)]
    if eligible_indices:
        mask_matrix = np.stack(
            [masks[index].reshape(-1) for index in eligible_indices]
        ).astype(np.float32, copy=False)
        intersection_matrix = mask_matrix @ mask_matrix.T
    else:
        intersection_matrix = np.empty((0, 0), dtype=np.float32)
    for offset, left_index in enumerate(eligible_indices):
        left_area = int(areas[left_index])
        left_boundary = geometry[left_index][3]
        left_distance = geometry[left_index][4]
        left_dilated = geometry[left_index][5]
        for right_offset, right_index in enumerate(
            eligible_indices[offset + 1 :], start=offset + 1
        ):
            right_area = int(areas[right_index])
            right_boundary = geometry[right_index][3]
            right_distance = geometry[right_index][4]
            right_dilated = geometry[right_index][5]
            intersection = int(intersection_matrix[offset, right_offset])
            union_count = left_area + right_area - intersection
            if intersection:
                boundary_distance = 0.0
            else:
                boundary_distance = float(
                    min(left_distance[right_boundary].min(), right_distance[left_boundary].min())
                )
            contact = (
                (left_dilated & right_boundary)
                | (left_boundary & right_dilated)
            )
            edge_pairs.append((left_index, right_index))
            edge_rows.append(
                [
                    intersection / union_count if union_count else 0.0,
                    intersection / left_area,
                    intersection / right_area,
                    intersection / min(left_area, right_area),
                    abs(math.log((left_area + 1) / (right_area + 1))),
                    abs(scores[left_index] - scores[right_index]),
                    boundary_distance,
                    int(contact.sum()),
                ]
            )
    expected_edges = len(eligible_indices) * (len(eligible_indices) - 1) // 2
    if len(edge_pairs) != expected_edges:
        raise RuntimeError("nonempty candidate graph is not complete")
    return CandidateGraph(
        case_id=case_id,
        frame_id=frame_id,
        candidate_ids=ids,
        emitted_ranks=ranks,
        scores=scores,
        masks=masks,
        eligible=eligible,
        node_features=np.asarray(node_rows, dtype=np.float32),
        edge_index=np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2),
        edge_features=np.asarray(edge_rows, dtype=np.float32).reshape(-1, len(EDGE_FEATURES)),
        audit={
            "freeze_id": FREEZE_ID,
            "candidate_count": len(ordered),
            "nonempty_count": len(eligible_indices),
            "empty_count": empty_count,
            "edge_count": len(edge_pairs),
            "expected_complete_edge_count": expected_edges,
            "node_features": list(NODE_FEATURES),
            "edge_features": list(EDGE_FEATURES),
            "embedding_features": False,
            "fixed_k_pruning": False,
            "empty_masks_create_edges": False,
            "empty_masks_selectable": False,
        },
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    return intersection / union if union else 0.0


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int((left & right).sum())
    denominator = min(int(left.sum()), int(right.sum()))
    return intersection / denominator if denominator else 0.0


def derive_native_labels(
    graph: CandidateGraph,
    gt_masks: Sequence[np.ndarray],
    *,
    matching_iou: float = 0.5,
    topology_overlap: float = 0.5,
) -> NativeLabels:
    targets = [np.asarray(mask, dtype=bool) for mask in gt_masks]
    matrix = np.asarray(
        [[_iou(pred, target) for target in targets] for pred in graph.masks],
        dtype=float,
    ).reshape(len(graph.masks), len(targets))
    overlaps = np.asarray(
        [[_overlap(pred, target) for target in targets] for pred in graph.masks],
        dtype=float,
    ).reshape(len(graph.masks), len(targets))
    assignments: dict[int, int] = {}
    if len(graph.masks) and targets:
        pred_indices, target_indices = linear_sum_assignment(-matrix)
        assignments = {
            int(pred): int(target)
            for pred, target in zip(pred_indices, target_indices)
            if graph.eligible[int(pred)] and matrix[pred, target] >= matching_iou
        }
    matched_targets = set(assignments.values())
    split_targets = {
        target_index
        for target_index in range(len(targets))
        if int((overlaps[:, target_index] >= topology_overlap).sum()) >= 2
    }
    node_validity = np.zeros(len(graph.candidate_ids), dtype=np.float32)
    topology = np.zeros((len(graph.candidate_ids), len(TOPOLOGY_TARGETS)), dtype=np.float32)
    candidate_targets: list[set[int]] = []
    for index in range(len(graph.candidate_ids)):
        overlap_targets = set(np.flatnonzero(overlaps[index] >= topology_overlap).tolist())
        candidate_targets.append(overlap_targets)
        node_validity[index] = float(index in assignments)
        duplicate = (
            index not in assignments
            and any(matrix[index, target] >= matching_iou for target in matched_targets)
        )
        merge = len(overlap_targets) >= 2
        split = bool(overlap_targets & split_targets)
        topology[index] = (duplicate, merge, split)
    pair_competition = np.zeros(len(graph.edge_index), dtype=np.float32)
    for edge_index, (left, right) in enumerate(graph.edge_index):
        pair_competition[edge_index] = float(
            bool(candidate_targets[int(left)] & candidate_targets[int(right)])
        )
    return NativeLabels(
        node_validity=node_validity,
        topology=topology,
        pair_competition=pair_competition,
        cardinality_class=min(len(targets), 10),
    )


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class _NodeEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.10),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _SharedHeads(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_validity = nn.Linear(hidden_dim, 1)
        self.topology = nn.Linear(hidden_dim, len(TOPOLOGY_TARGETS))
        self.cardinality = nn.Linear(hidden_dim * 2, 11)

    def forward(
        self, states: torch.Tensor, eligible: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        valid_states = states[eligible]
        if len(valid_states):
            pooled = torch.cat(
                (valid_states.mean(dim=0), valid_states.max(dim=0).values), dim=0
            )
        else:
            pooled = torch.zeros(
                states.shape[1] * 2, dtype=states.dtype, device=states.device
            )
        return {
            "node_validity_logits": self.node_validity(states).squeeze(-1),
            "topology_logits": self.topology(states),
            "cardinality_logits": self.cardinality(pooled),
        }


class NodeOnlyMLP(nn.Module):
    """Fair node-only comparator: same node encoder/heads, no edge features/messages."""

    def __init__(self, node_dim: int = len(NODE_FEATURES), hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_encoder = _NodeEncoder(node_dim, hidden_dim)
        self.heads = _SharedHeads(hidden_dim)
        self.pair_head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        eligible: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del edge_features
        states = self.node_encoder(node_features)
        output = self.heads(states, eligible)
        if len(edge_index):
            endpoints = torch.cat(
                (states[edge_index[:, 0]], states[edge_index[:, 1]]), dim=1
            )
            output["pair_competition_logits"] = self.pair_head(endpoints).squeeze(-1)
        else:
            output["pair_competition_logits"] = states.new_empty((0,))
        return output


class _MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_states: torch.Tensor,
    ) -> torch.Tensor:
        if not len(edge_index):
            return states
        left, right = edge_index[:, 0], edge_index[:, 1]
        source = torch.cat((left, right))
        target = torch.cat((right, left))
        repeated_edges = torch.cat((edge_states, edge_states), dim=0)
        messages = self.message(
            torch.cat((states[source], states[target], repeated_edges), dim=1)
        )
        aggregate = torch.zeros_like(states)
        aggregate.index_add_(0, target, messages)
        degrees = torch.zeros(
            len(states), dtype=states.dtype, device=states.device
        )
        degrees.index_add_(0, target, torch.ones_like(target, dtype=states.dtype))
        aggregate = aggregate / degrees.clamp_min(1).unsqueeze(1)
        return states + self.update(torch.cat((states, aggregate), dim=1))


class TopologyAwareCandidateGraph(nn.Module):
    def __init__(
        self,
        node_dim: int = len(NODE_FEATURES),
        edge_dim: int = len(EDGE_FEATURES),
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.node_encoder = _NodeEncoder(node_dim, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
        )
        self.message_layers = nn.ModuleList(
            [_MessageLayer(hidden_dim), _MessageLayer(hidden_dim)]
        )
        self.heads = _SharedHeads(hidden_dim)
        self.pair_head = nn.Linear(hidden_dim * 3, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        eligible: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        states = self.node_encoder(node_features)
        edge_states = self.edge_encoder(edge_features)
        for layer in self.message_layers:
            states = layer(states, edge_index, edge_states)
        output = self.heads(states, eligible)
        if len(edge_index):
            endpoints = torch.cat(
                (
                    states[edge_index[:, 0]],
                    states[edge_index[:, 1]],
                    edge_states,
                ),
                dim=1,
            )
            output["pair_competition_logits"] = self.pair_head(endpoints).squeeze(-1)
        else:
            output["pair_competition_logits"] = states.new_empty((0,))
        return output


def graph_to_tensors(graph: CandidateGraph) -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.as_tensor(graph.node_features, dtype=torch.float32),
        "edge_index": torch.as_tensor(graph.edge_index, dtype=torch.long),
        "edge_features": torch.as_tensor(graph.edge_features, dtype=torch.float32),
        "eligible": torch.as_tensor(graph.eligible, dtype=torch.bool),
    }


def multitask_loss(
    output: dict[str, torch.Tensor],
    labels: NativeLabels,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Identical unit-weight loss contract for graph and node-only paths."""
    device = output["node_validity_logits"].device
    node_target = torch.as_tensor(labels.node_validity, dtype=torch.float32, device=device)
    topology_target = torch.as_tensor(labels.topology, dtype=torch.float32, device=device)
    pair_target = torch.as_tensor(
        labels.pair_competition, dtype=torch.float32, device=device
    )
    cardinality_target = torch.as_tensor(
        [labels.cardinality_class], dtype=torch.long, device=device
    )
    losses = {
        "node_validity": nn.functional.binary_cross_entropy_with_logits(
            output["node_validity_logits"], node_target
        ),
        "topology": nn.functional.binary_cross_entropy_with_logits(
            output["topology_logits"], topology_target
        ),
        "cardinality": nn.functional.cross_entropy(
            output["cardinality_logits"].unsqueeze(0), cardinality_target
        ),
    }
    if len(pair_target):
        losses["pair_competition"] = nn.functional.binary_cross_entropy_with_logits(
            output["pair_competition_logits"], pair_target
        )
    else:
        losses["pair_competition"] = output["node_validity_logits"].sum() * 0.0
    return sum(losses.values()), losses


def exact_structured_select(
    candidate_ids: Sequence[str],
    node_validity_logits: Sequence[float],
    edge_index: np.ndarray,
    pair_competition_probabilities: Sequence[float],
    eligible: Sequence[bool],
    predicted_cardinality: int,
    *,
    pair_penalty: float = PAIR_PENALTY,
) -> SelectionResult:
    count = len(candidate_ids)
    if not (
        len(node_validity_logits) == len(eligible) == count
        and len(pair_competition_probabilities) == len(edge_index)
    ):
        raise ValueError("selector input lengths differ")
    if pair_penalty != PAIR_PENALTY:
        raise ValueError("RID-QGRAPH-M1 fixes pair_penalty=1.0")
    eligible_array = np.asarray(eligible, dtype=bool)
    eligible_count = int(eligible_array.sum())
    requested = int(predicted_cardinality)
    if requested < 0 or requested > eligible_count:
        return SelectionResult(
            "INFEASIBLE_CARDINALITY", (), (), requested, eligible_count, None, False
        )
    if requested == 0:
        return SelectionResult("PASS_EXACT", (), (), 0, eligible_count, 0.0, True)
    canonical = sorted(range(count), key=lambda index: candidate_ids[index])
    remap = {old: new for new, old in enumerate(canonical)}
    reverse = {new: old for old, new in remap.items()}
    canonical_eligible = eligible_array[canonical]
    logits = np.asarray(node_validity_logits, dtype=float)[canonical]
    pair_map: dict[tuple[int, int], float] = {}
    for (left, right), probability in zip(
        np.asarray(edge_index, dtype=int),
        pair_competition_probabilities,
    ):
        a, b = sorted((remap[int(left)], remap[int(right)]))
        if (a, b) in pair_map:
            raise ValueError("duplicate selector edge")
        pair_map[(a, b)] = float(probability)
    eligible_canonical = [index for index in range(count) if canonical_eligible[index]]
    expected_pairs = {
        (left, right)
        for position, left in enumerate(eligible_canonical)
        for right in eligible_canonical[position + 1 :]
    }
    if set(pair_map) != expected_pairs:
        return SelectionResult(
            "INCOMPLETE_PAIR_GRAPH", (), (), requested, eligible_count, None, False
        )
    if requested <= 2:
        best_subset: tuple[int, ...] | None = None
        best_objective = -math.inf
        if requested == 1:
            subsets: Iterable[tuple[int, ...]] = (
                (left,) for left in eligible_canonical
            )
        elif requested == 2:
            subsets = (
                (left, right)
                for position, left in enumerate(eligible_canonical)
                for right in eligible_canonical[position + 1 :]
            )
        for subset in subsets:
            objective_value = float(logits[list(subset)].sum())
            if requested >= 2:
                objective_value -= pair_penalty * sum(
                    pair_map[(left, right)]
                    for position, left in enumerate(subset)
                    for right in subset[position + 1 :]
                )
            if (
                objective_value > best_objective
                or (
                    objective_value == best_objective
                    and (best_subset is None or subset < best_subset)
                )
            ):
                best_objective = objective_value
                best_subset = subset
        if best_subset is None:
            return SelectionResult(
                "EXACT_ENUMERATION_EMPTY",
                (),
                (),
                requested,
                eligible_count,
                None,
                False,
            )
        selected_original = tuple(sorted(reverse[index] for index in best_subset))
        selected_ids = tuple(
            sorted(candidate_ids[index] for index in selected_original)
        )
        return SelectionResult(
            "PASS_EXACT",
            selected_ids,
            selected_original,
            requested,
            eligible_count,
            best_objective,
            True,
        )
    pairs = sorted(pair_map)
    variable_count = count + len(pairs)
    objective = np.zeros(variable_count, dtype=float)
    objective[:count] = -logits
    objective[count:] = pair_penalty * np.asarray([pair_map[pair] for pair in pairs])
    lower = np.zeros(variable_count)
    upper = np.ones(variable_count)
    upper[:count] = canonical_eligible.astype(float)
    rows = 1 + 3 * len(pairs)
    constraints = lil_matrix((rows, variable_count), dtype=float)
    lower_constraints = np.full(rows, -np.inf)
    upper_constraints = np.full(rows, np.inf)
    constraints[0, :count] = 1.0
    lower_constraints[0] = upper_constraints[0] = requested
    row = 1
    for pair_index, (left, right) in enumerate(pairs):
        z = count + pair_index
        constraints[row, z] = 1
        constraints[row, left] = -1
        upper_constraints[row] = 0
        row += 1
        constraints[row, z] = 1
        constraints[row, right] = -1
        upper_constraints[row] = 0
        row += 1
        constraints[row, left] = 1
        constraints[row, right] = 1
        constraints[row, z] = -1
        upper_constraints[row] = 1
        row += 1
    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            constraints.tocsr(), lower_constraints, upper_constraints
        ),
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return SelectionResult(
            f"SOLVER_FLAG_{result.status}", (), (), requested, eligible_count, None, False
        )
    selected_canonical = [
        index for index in range(count) if result.x[index] >= 0.5
    ]
    if len(selected_canonical) != requested:
        return SelectionResult(
            "SOLVER_CARDINALITY_MISMATCH",
            (),
            (),
            requested,
            eligible_count,
            None,
            False,
        )
    selected_original = tuple(sorted(reverse[index] for index in selected_canonical))
    selected_ids = tuple(sorted(candidate_ids[index] for index in selected_original))
    return SelectionResult(
        "PASS_EXACT",
        selected_ids,
        selected_original,
        requested,
        eligible_count,
        float(-result.fun),
        True,
    )


def load_and_validate_folds(
    path: Path, *, expected_cases: set[str] | None = None
) -> dict[int, dict[str, set[str]]]:
    assert_source_development_path(path)
    folds: dict[int, dict[str, set[str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role = row["role"]
            if role not in {"fit", "calibration", "held_out"}:
                raise ValueError(f"unknown fold role: {role}")
            folds.setdefault(
                int(row["outer_fold"]),
                {"fit": set(), "calibration": set(), "held_out": set()},
            )[role].add(row["case_id"])
    if set(folds) != set(range(1, 7)):
        raise ValueError("expected six outer folds")
    held_counts: dict[str, int] = {}
    universe: set[str] | None = None
    for fold, roles in folds.items():
        if any(roles[left] & roles[right] for left in roles for right in roles if left < right):
            raise ValueError(f"case-role leakage in fold {fold}")
        current = set().union(*roles.values())
        if universe is None:
            universe = current
        elif current != universe:
            raise ValueError("case universe changes across folds")
        for case in roles["held_out"]:
            held_counts[case] = held_counts.get(case, 0) + 1
    if universe is None or any(held_counts.get(case, 0) != 1 for case in universe):
        raise ValueError("each case must be held out exactly once")
    if expected_cases is not None and universe != expected_cases:
        raise ValueError("fold case universe differs from frozen allowlist")
    return folds


def model_fairness_audit() -> dict[str, object]:
    node = NodeOnlyMLP()
    graph = TopologyAwareCandidateGraph()
    node_encoder_equal = {
        name: tuple(parameter.shape)
        for name, parameter in node.node_encoder.named_parameters()
    } == {
        name: tuple(parameter.shape)
        for name, parameter in graph.node_encoder.named_parameters()
    }
    shared_heads_equal = {
        name: tuple(parameter.shape)
        for name, parameter in node.heads.named_parameters()
    } == {
        name: tuple(parameter.shape)
        for name, parameter in graph.heads.named_parameters()
    }
    node_only_names = [name for name, _ in node.named_parameters()]
    return {
        "freeze_id": FREEZE_ID,
        "node_features_identical": True,
        "node_encoder_architecture_identical": node_encoder_equal,
        "shared_head_architecture_identical": shared_heads_equal,
        "cardinality_classes_identical": True,
        "node_only_has_edge_encoder": any("edge_encoder" in name for name in node_only_names),
        "node_only_has_message_layer": any("message_layers" in name for name in node_only_names),
        "only_graph_specific_components": [
            "edge_encoder",
            "two_message_layers",
            "edge_augmented_pair_head",
        ],
    }
