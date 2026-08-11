"""候选图建模与结构化实例集选择。"""

from .rid_qgraph_core import (
    EDGE_FEATURES,
    NODE_FEATURES,
    Candidate,
    CandidateGraph,
    NodeOnlyMLP,
    SelectionResult,
    TopologyAwareCandidateGraph,
    build_candidate_graph,
    exact_structured_select,
    graph_to_tensors,
)

__all__ = [
    "EDGE_FEATURES",
    "NODE_FEATURES",
    "Candidate",
    "CandidateGraph",
    "NodeOnlyMLP",
    "SelectionResult",
    "TopologyAwareCandidateGraph",
    "build_candidate_graph",
    "exact_structured_select",
    "graph_to_tensors",
]

__version__ = "0.1.0"
