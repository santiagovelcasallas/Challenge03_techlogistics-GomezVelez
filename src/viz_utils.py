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
                        max_labels: int = 25, layout: str = "spring"):
    """Dibuja un grafo dirigido resaltando el cuello de botella en rojo brillante (alpha=1.0)
    y aplicando un 50% de transparencia (alpha=0.50) a los demás nodos para storytelling ejecutivo.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))
    
    # Cálculo de Throughput (Grado ponderado)
    wdeg = {n: sum(d.get("weight", 1) for _, _, d in G.in_edges(n, data=True))
               + sum(d.get("weight", 1) for _, _, d in G.out_edges(n, data=True))
            for n in G.nodes()}
    wmax = max(wdeg.values()) or 1

    # Disposición espacial (Bipartita en 2 columnas o Spring)
    sources = [n for n in G.nodes() if G.out_degree(n) > 0 and G.in_degree(n) == 0]
    if layout == "bipartite" and sources:
        pos = nx.bipartite_layout(G, sources)
    else:
        pos = nx.spring_layout(G, seed=42, k=0.6)

    node_list = list(G.nodes())
    sizes = [200 + 2500 * wdeg[n] / wmax for n in node_list]

    # Dibujar Aristas (Flujos)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.20, arrows=True,
                           arrowsize=8, edge_color="gray")

    # 1. Nodos Secundarios: Transparencia del 50% (alpha=0.50)
    other_nodes = [n for n in node_list if n != bottleneck]
    if other_nodes:
        other_sizes = [sizes[node_list.index(n)] for n in other_nodes]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=other_nodes, node_size=other_sizes,
                               node_color="#1f77b4", alpha=0.50, linewidths=0.5, edgecolors="white")

    # 2. Nodo Cuello de Botella: Opacidad 100% (alpha=1.0) + Rojo brillante + Borde negro
    if bottleneck is not None and bottleneck in G.nodes():
        b_idx = node_list.index(bottleneck)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[bottleneck], node_size=[sizes[b_idx]],
                               node_color="#d62728", alpha=1.0, linewidths=2.0, edgecolors="black")

    top = sorted(wdeg, key=wdeg.get, reverse=True)[:max_labels]
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: str(n) for n in top}, font_size=8, font_weight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    return ax


def savefig(path: str | Path, dpi: int = 150) -> Path:
    """Save the current matplotlib figure tightly and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
