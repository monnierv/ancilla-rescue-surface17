import math

import matplotlib.pyplot as plt
import networkx as nx
import stim


def build_code_graph() -> nx.Graph:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=1,
    )
    coordinates = circuit.get_final_qubit_coordinates()
    graph = nx.Graph()
    positions = {}

    data_qubits = []
    check_qubits = []
    for qubit, coordinate in coordinates.items():
        x, y = coordinate[:2]
        if int(x) % 2 == 1 and int(y) % 2 == 1:
            data_qubits.append(qubit)
        else:
            check_qubits.append(qubit)

    for index, qubit in enumerate(sorted(data_qubits), start=1):
        node = ("data", qubit)
        position = tuple(coordinates[qubit][:2])
        graph.add_node(node, kind="data", label=f"D{index}", qubit=qubit, pos=position)
        positions[node] = position

    for index, qubit in enumerate(sorted(check_qubits), start=1):
        node = ("check", qubit)
        position = tuple(coordinates[qubit][:2])
        graph.add_node(node, kind="check", label=f"S{index}", qubit=qubit, pos=position)
        positions[node] = position

    data_nodes = [node for node, attributes in graph.nodes(data=True) if attributes["kind"] == "data"]
    check_nodes = [node for node, attributes in graph.nodes(data=True) if attributes["kind"] == "check"]
    for data_node in data_nodes:
        for check_node in check_nodes:
            data_x, data_y = positions[data_node]
            check_x, check_y = positions[check_node]
            if math.isclose(abs(data_x - check_x), 1.0) and math.isclose(
                abs(data_y - check_y),
                1.0,
            ):
                graph.add_edge(data_node, check_node)

    return graph


def plot_code_map(output_path: str | None = None, show: bool = True) -> None:
    graph = build_code_graph()
    positions = nx.get_node_attributes(graph, "pos")

    data_nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == "data"]
    check_nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == "check"]
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
    axis.set_title("Surface-17 rotated code (distance 3)", fontsize=13, color="#263330")
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()

    if output_path:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)