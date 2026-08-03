"""Visualization helpers (geo maps, spectra, graphs) with static PNG export."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go


def save_plotly(fig: go.Figure, path: str | Path, scale: int = 2) -> Path:
    """Export a Plotly figure to PNG via kaleido; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=scale)
    return path


def geo_sensor_map(df, lat="Latitude", lon="Longitude",
                   color="Agro_5", size="Agro_1",
                   color_label="NDVI", size_label="Humedad",
                   title="Sensores – Oriente Antioqueño") -> go.Figure:
    """Scatter map of sensors colored by an index and sized by another variable.

    Uses the free open-street-map style so no Mapbox token is required.
    """
    d = df.copy()
    # size must be positive for plotly marker scaling
    d["_size"] = (d[size] - d[size].min()) + 1e-6
    fig = px.scatter_mapbox(
        d, lat=lat, lon=lon, color=color, size="_size",
        color_continuous_scale="RdYlGn", size_max=14, zoom=8,
        hover_data={lat: ":.4f", lon: ":.4f", color: ":.3f", "_size": False},
        labels={color: color_label},
        title=title,
    )
    fig.update_layout(mapbox_style="open-street-map",
                      margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def draw_directed_graph(G: nx.DiGraph, bottleneck: int | None = None,
                        title: str = "Topología de red", ax=None,
                        max_labels: int = 25):
    """Draw a directed graph, highlighting the bottleneck node in red.

    Node size scales with **throughput** (weighted degree). Betweenness is not used
    for sizing because this topology is bipartite (all betweenness values are 0).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))
    wdeg = {n: sum(d.get("weight", 1) for _, _, d in G.in_edges(n, data=True))
               + sum(d.get("weight", 1) for _, _, d in G.out_edges(n, data=True))
            for n in G.nodes()}
    wmax = max(wdeg.values()) or 1
    pos = nx.spring_layout(G, seed=42, k=0.6)
    sizes = [200 + 2500 * wdeg[n] / wmax for n in G.nodes()]
    colors = ["#d62728" if n == bottleneck else "#1f77b4" for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.20, arrows=True,
                           arrowsize=7, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           alpha=0.85, linewidths=0.5, edgecolors="white")
    # label the highest-throughput nodes to avoid clutter
    top = sorted(wdeg, key=wdeg.get, reverse=True)[:max_labels]
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: str(n) for n in top}, font_size=7)
    ax.set_title(title)
    ax.axis("off")
    return ax


def savefig(path: str | Path, dpi: int = 150) -> Path:
    """Save the current matplotlib figure tightly and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
