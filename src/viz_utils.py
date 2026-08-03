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

    Node size scales with betweenness centrality.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))
    btw = nx.betweenness_centrality(G, normalized=True)
    pos = nx.spring_layout(G, seed=42, k=0.6)
    sizes = [300 + 6000 * btw.get(n, 0) for n in G.nodes()]
    colors = ["#d62728" if n == bottleneck else "#1f77b4" for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, arrows=True,
                           arrowsize=8, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           alpha=0.85, linewidths=0.5, edgecolors="white")
    # only label the most central nodes to avoid clutter
    top = sorted(btw, key=btw.get, reverse=True)[:max_labels]
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
