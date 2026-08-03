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


def centrality_table(G: nx.DiGraph) -> pd.DataFrame:
    """Return per-node degree and betweenness centralities as a tidy DataFrame."""
    deg = nx.degree_centrality(G)
    indeg = nx.in_degree_centrality(G)
    outdeg = nx.out_degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    rows = [{
        "node": n,
        "degree_centrality": deg.get(n, 0.0),
        "in_degree_centrality": indeg.get(n, 0.0),
        "out_degree_centrality": outdeg.get(n, 0.0),
        "betweenness_centrality": btw.get(n, 0.0),
    } for n in G.nodes()]
    return (pd.DataFrame(rows)
            .sort_values("betweenness_centrality", ascending=False)
            .reset_index(drop=True))


def bottleneck_node(G: nx.DiGraph) -> tuple[int, float]:
    """Return the (node, betweenness) with the highest betweenness centrality."""
    btw = nx.betweenness_centrality(G, normalized=True)
    node = max(btw, key=btw.get)
    return int(node), float(btw[node])


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
