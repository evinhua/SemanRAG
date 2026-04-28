"""Benchmark Leiden community detection on random graphs of varying sizes."""

import time

import networkx as nx
import pytest

EDGE_COUNTS = [1_000, 10_000, 100_000]


def _make_random_graph(num_edges: int) -> nx.Graph:
    # Approximate node count to achieve target edge count with Erdos-Renyi
    n = max(50, int((2 * num_edges) ** 0.5))
    p = (2 * num_edges) / (n * (n - 1)) if n > 1 else 0.5
    p = min(p, 1.0)
    return nx.erdos_renyi_graph(n, p, seed=42)


@pytest.mark.benchmark
@pytest.mark.parametrize("num_edges", EDGE_COUNTS, ids=["1k", "10k", "100k"])
def test_community_detection_leiden(num_edges: int):
    from graspologic.partition import hierarchical_leiden

    G = _make_random_graph(num_edges)
    actual_edges = G.number_of_edges()
    start = time.perf_counter()
    results = hierarchical_leiden(G, max_cluster_size=len(G.nodes()) + 1)
    elapsed = time.perf_counter() - start
    communities = {}
    for entry in results:
        communities.setdefault(str(entry.cluster), []).append(entry.node)
    print(
        f"\nleiden({num_edges} target edges, {actual_edges} actual, "
        f"{G.number_of_nodes()} nodes): {elapsed:.4f}s, {len(communities)} communities"
    )


if __name__ == "__main__":
    from graspologic.partition import hierarchical_leiden

    for num_edges in EDGE_COUNTS:
        G = _make_random_graph(num_edges)
        actual_edges = G.number_of_edges()
        start = time.perf_counter()
        results = hierarchical_leiden(G, max_cluster_size=len(G.nodes()) + 1)
        elapsed = time.perf_counter() - start
        communities = {}
        for entry in results:
            communities.setdefault(str(entry.cluster), []).append(entry.node)
        print(
            f"leiden({num_edges} target edges, {actual_edges} actual, "
            f"{G.number_of_nodes()} nodes): {elapsed:.4f}s, {len(communities)} communities"
        )
