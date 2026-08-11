"""不依赖论文数据的最小端到端示例。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .rid_qgraph_core import (
    Candidate,
    NodeOnlyMLP,
    TopologyAwareCandidateGraph,
    build_candidate_graph,
    exact_structured_select,
    graph_to_tensors,
    set_deterministic_seed,
)


@dataclass(frozen=True)
class DemoSummary:
    candidate_count: int
    eligible_count: int
    edge_count: int
    graph_cardinality_classes: int
    node_only_cardinality_classes: int
    selected_count: int
    selector_status: str


def build_demo_graph():
    """构造三个合成候选，其中两个有效、一个为空。"""
    masks = [
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool),
        np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool),
        np.zeros((3, 3), dtype=bool),
    ]
    candidates = [
        Candidate(f"q{index}", index + 1, 0.9 - 0.2 * index, mask)
        for index, mask in enumerate(masks)
    ]
    return build_candidate_graph(
        "synthetic-case",
        "synthetic-frame",
        candidates,
        require_100_queries=False,
    )


def run_demo() -> DemoSummary:
    """依次运行图构造、两条模型路径和 MILP 选择器。"""
    set_deterministic_seed(0)
    graph = build_demo_graph()
    tensors = graph_to_tensors(graph)

    graph_model = TopologyAwareCandidateGraph().eval()
    node_only_model = NodeOnlyMLP().eval()
    with torch.no_grad():
        graph_output = graph_model(**tensors)
        node_only_output = node_only_model(**tensors)

    selection = exact_structured_select(
        graph.candidate_ids,
        graph_output["node_validity_logits"].numpy(),
        graph.edge_index,
        torch.sigmoid(graph_output["pair_competition_logits"]).numpy(),
        graph.eligible,
        predicted_cardinality=1,
    )

    return DemoSummary(
        candidate_count=len(graph.candidate_ids),
        eligible_count=int(graph.eligible.sum()),
        edge_count=len(graph.edge_index),
        graph_cardinality_classes=graph_output["cardinality_logits"].numel(),
        node_only_cardinality_classes=node_only_output["cardinality_logits"].numel(),
        selected_count=len(selection.selected_candidate_ids),
        selector_status=selection.status,
    )


def main() -> None:
    summary = run_demo()
    print("端到端示例运行完成")
    for key, value in asdict(summary).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
