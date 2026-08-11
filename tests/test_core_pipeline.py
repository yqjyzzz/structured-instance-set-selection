from structured_instance_selection.demo import build_demo_graph, run_demo
from structured_instance_selection.rid_qgraph_core import EDGE_FEATURES, NODE_FEATURES


def test_candidate_graph_contract() -> None:
    graph = build_demo_graph()
    assert graph.node_features.shape == (3, len(NODE_FEATURES))
    assert graph.edge_features.shape == (1, len(EDGE_FEATURES))
    assert graph.eligible.tolist() == [True, True, False]


def test_end_to_end_demo() -> None:
    summary = run_demo()
    assert summary.candidate_count == 3
    assert summary.eligible_count == 2
    assert summary.edge_count == 1
    assert summary.graph_cardinality_classes == 11
    assert summary.node_only_cardinality_classes == 11
    assert summary.selected_count == 1
    assert summary.selector_status == "PASS_EXACT"
