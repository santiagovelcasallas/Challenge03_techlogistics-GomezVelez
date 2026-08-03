"""Directed-graph construction and centrality analysis over the network topology."""
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_directed_graph(df: pd.DataFrame,
                         source_col: str = "Source_Node",
                         target_col: str = "Target_Node") -> nx.DiGraph:
    """Build a directed graph from Source_Node -> Target_Node edges.

    Parallel edges are collapsed and their multiplicity stored as a ``weight``
    attribute (how many records traverse that link).
    """
    edges = df[[source_col, target_col]].dropna().astype(int)
    G = nx.DiGraph()
    counts = edges.value_counts().to_dict()  # {(s, t): weight}
    for (s, t), w in counts.items():
        G.add_edge(int(s), int(t), weight=int(w))
    return G


def weighted_degree(G: nx.DiGraph) -> dict:
    """Return {node: throughput}, i.e. the number of records (edge weights) that
    flow in+out of each node. This is the operational 'load' of a node."""
    out = {}
    for n in G.nodes():
        w_in = sum(d.get("weight", 1) for _, _, d in G.in_edges(n, data=True))
        w_out = sum(d.get("weight", 1) for _, _, d in G.out_edges(n, data=True))
        out[n] = int(w_in + w_out)
    return out


def is_betweenness_degenerate(G: nx.DiGraph, tol: float = 1e-9) -> bool:
    """True if every node has ~0 betweenness (e.g. a bipartite single-hop graph,
    where sources and targets are disjoint so no node is ever an intermediary)."""
    btw = nx.betweenness_centrality(G, normalized=True)
    return max(btw.values(), default=0.0) <= tol


def centrality_table(G: nx.DiGraph) -> pd.DataFrame:
    """Return per-node degree, betweenness and throughput as a tidy DataFrame,
    sorted by throughput (the meaningful ranking when betweenness is degenerate)."""
    deg = nx.degree_centrality(G)
    indeg = nx.in_degree_centrality(G)
    outdeg = nx.out_degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    wdeg = weighted_degree(G)
    rows = [{
        "node": n,
        "degree_centrality": deg.get(n, 0.0),
        "in_degree_centrality": indeg.get(n, 0.0),
        "out_degree_centrality": outdeg.get(n, 0.0),
        "betweenness_centrality": btw.get(n, 0.0),
        "throughput_records": wdeg.get(n, 0),
    } for n in G.nodes()]
    return (pd.DataFrame(rows)
            .sort_values(["betweenness_centrality", "throughput_records"],
                         ascending=False)
            .reset_index(drop=True))


def bottleneck_node(G: nx.DiGraph) -> tuple[int, float, str]:
    """Identify the network bottleneck node.

    Returns ``(node, score, method)``. Primary criterion is **betweenness
    centrality**. If betweenness is degenerate (all ~0, e.g. a bipartite
    source→target graph), we fall back to **throughput (weighted degree)** — the
    node handling the most records — which is the true operational bottleneck.
    """
    btw = nx.betweenness_centrality(G, normalized=True)
    if max(btw.values(), default=0.0) > 1e-9:
        node = max(btw, key=btw.get)
        return int(node), float(btw[node]), "betweenness"
    wdeg = weighted_degree(G)
    node = max(wdeg, key=wdeg.get)
    return int(node), float(wdeg[node]), "throughput"


def node_centrality_map(G: nx.DiGraph, kind: str = "degree") -> dict:
    """Return a {node: centrality} dict for use as an exogenous feature.

    ``kind`` in {"degree", "in_degree", "out_degree", "betweenness"}.
    """
    if kind == "degree":
        return nx.degree_centrality(G)
    if kind == "in_degree":
        return nx.in_degree_centrality(G)
    if kind == "out_degree":
        return nx.out_degree_centrality(G)
    if kind == "betweenness":
        return nx.betweenness_centrality(G, normalized=True)
    raise ValueError(f"unknown centrality kind: {kind}")
