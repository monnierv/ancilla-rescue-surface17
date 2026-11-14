from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from qec_config import build_code_graph as build_configured_graph
from qec_config import load_code_config


def build_code_graph(config_path: str | Path | None = None) -> nx.Graph:
    config = load_code_config(config_path, "surface17.json")
    return build_configured_graph(config)


def plot_code_map(
    output_path: str | None = None,
    show: bool = True,
    config_path: str | Path | None = None,
) -> None:
    graph = build_code_graph(config_path)
    positions = nx.get_node_attributes(graph, "pos")

    data_nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == "data"]
    check_nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == "ancilla"]
    labels = nx.get_node_attributes(graph, "label")

    figure, axis = plt.subplots(figsize=(7, 6), facecolor="#f7f5ef")
    axis.set_facecolor("#f7f5ef")
    nx.draw_networkx_edges(graph, positions, ax=axis, edge_color="#aab3ae", width=1.5)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=data_nodes,
        node_shape="o",
        node_color="#167d78",
        edgecolors="#104c4a",
        node_size=1000,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=check_nodes,
        node_shape="s",
        node_color="#e28b42",
        edgecolors="#87512d",
        node_size=850,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=9, font_color="white", ax=axis)
    axis.set_title(graph.graph["name"], fontsize=13, color="#263330")
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()

    if output_path:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)