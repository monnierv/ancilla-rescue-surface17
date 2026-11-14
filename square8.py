"""Repeated syndrome detection for an eight-data-qubit square layout."""

import argparse
from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymatching
import stim


DATA_COORDINATES = (
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
)
CHECKS = (
    ("top", "X", (0, 1, 2, 3), 8, (0, 2)),
    ("right", "Z", (2, 3, 4, 5), 9, (2, 0)),
    ("bottom", "X", (4, 5, 6, 7), 10, (0, -2)),
    ("left", "Z", (6, 7, 0, 1), 11, (-2, 0)),
)
DATA_QUBITS = tuple(range(8))
ANCILLA_QUBITS = tuple(range(8, 12))
LOGICAL_Z_SUPPORT = (0, 2)


@dataclass(frozen=True)
class Square8Result:
    shots: int
    rounds: int
    physical_error_rate: float
    detection_events: np.ndarray
    observable_flips: np.ndarray
    predicted_observable_flips: np.ndarray
    logical_failures: np.ndarray


def build_square8_circuit(
    rounds: int = 3,
    physical_error_rate: float = 0.001,
    ancilla_errors: bool = False,
) -> stim.Circuit:
    if rounds <= 0:
        raise ValueError("rounds must be greater than zero")
    if not math.isfinite(physical_error_rate) or not 0 <= physical_error_rate <= 1:
        raise ValueError("physical_error_rate must be between 0 and 1")

    circuit = stim.Circuit()
    for qubit, (x, y) in enumerate(DATA_COORDINATES):
        circuit.append("QUBIT_COORDS", [qubit], [x, y])
    for _, _, _, ancilla, (x, y) in CHECKS:
        circuit.append("QUBIT_COORDS", [ancilla], [x, y])

    circuit.append("R", DATA_QUBITS)
    x_ancillas = tuple(ancilla for _, basis, _, ancilla, _ in CHECKS if basis == "X")

    for round_index in range(rounds):
        circuit.append("R", ANCILLA_QUBITS)
        if physical_error_rate:
            circuit.append("DEPOLARIZE1", DATA_QUBITS, physical_error_rate)
            if ancilla_errors:
                circuit.append("X_ERROR", ANCILLA_QUBITS, physical_error_rate)
        circuit.append("H", x_ancillas)

        for _, basis, support, ancilla, _ in CHECKS:
            for data_qubit in support:
                targets = (
                    [ancilla, data_qubit]
                    if basis == "X"
                    else [data_qubit, ancilla]
                )
                circuit.append("CX", targets)
                if physical_error_rate:
                    circuit.append("DEPOLARIZE2", targets, physical_error_rate)

        circuit.append("H", x_ancillas)
        if physical_error_rate:
            if ancilla_errors:
                circuit.append("X_ERROR", ANCILLA_QUBITS, physical_error_rate)
        circuit.append("MR", ANCILLA_QUBITS)

        if round_index:
            for check_index, (_, _, _, _, (x, y)) in enumerate(CHECKS):
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(-4 + check_index), stim.target_rec(-8 + check_index)],
                    [x, y, round_index],
                )

    circuit.append("M", DATA_QUBITS, physical_error_rate)
    for check_index, (_, basis, support, _, _) in enumerate(CHECKS):
        if basis == "Z":
            targets = [stim.target_rec(-12 + check_index)]
            targets.extend(stim.target_rec(-8 + qubit) for qubit in support)
            circuit.append("DETECTOR", targets)

    logical_targets = [stim.target_rec(-8 + qubit) for qubit in LOGICAL_Z_SUPPORT]
    circuit.append("OBSERVABLE_INCLUDE", logical_targets, 0)
    return circuit


def run_square8(
    shots: int = 1000,
    rounds: int = 3,
    physical_error_rate: float = 0.001,
    seed: int | None = None,
    ancilla_errors: bool = False,
) -> Square8Result:
    if shots <= 0:
        raise ValueError("shots must be greater than zero")

    circuit = build_square8_circuit(rounds, physical_error_rate, ancilla_errors)
    detection_events, observable_flips = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots,
        separate_observables=True,
    )
    matching = pymatching.Matching.from_stim_circuit(circuit)
    predicted_observable_flips = matching.decode_batch(detection_events)
    logical_failures = np.any(predicted_observable_flips != observable_flips, axis=1)
    return Square8Result(
        shots=shots,
        rounds=rounds,
        physical_error_rate=physical_error_rate,
        detection_events=detection_events,
        observable_flips=observable_flips,
        predicted_observable_flips=predicted_observable_flips,
        logical_failures=logical_failures,
    )


def build_square8_graph() -> nx.Graph:
    graph = nx.Graph()
    for index, position in enumerate(DATA_COORDINATES):
        graph.add_node(f"D{index + 1}", kind="data", label=f"D{index + 1}", pos=position)
    for index, (side, basis, support, _, position) in enumerate(CHECKS):
        ancilla = f"A{index + 1}"
        graph.add_node(
            ancilla,
            kind="ancilla",
            label=f"A{index + 1} ({basis})",
            side=side,
            basis=basis,
            pos=position,
        )
        for data_index in support:
            graph.add_edge(ancilla, f"D{data_index + 1}")
    return graph


def plot_square8_map(output_path: str | None = None, show: bool = True) -> None:
    graph = build_square8_graph()
    positions = nx.get_node_attributes(graph, "pos")
    data_nodes = [node for node, attributes in graph.nodes(data=True) if attributes["kind"] == "data"]
    ancilla_nodes = [node for node, attributes in graph.nodes(data=True) if attributes["kind"] == "ancilla"]
    labels = nx.get_node_attributes(graph, "label")

    figure, axis = plt.subplots(figsize=(7, 7), facecolor="#f7f5ef")
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
        nodelist=ancilla_nodes,
        node_shape="s",
        node_color="#e28b42",
        edgecolors="#87512d",
        node_size=1200,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, font_color="white", ax=axis)
    axis.set_title("Square-8 perimeter syndrome layout", fontsize=13, color="#263330")
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()

    if output_path:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated syndrome detection on an eight-qubit square.")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.001, help="physical error probability")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ancilla-errors", action="store_true", help="include syndrome-ancilla reset/readout errors")
    parser.add_argument("--plot", action="store_true", help="display the square node map")
    parser.add_argument("--save-map", metavar="PATH", help="save the square node map")
    arguments = parser.parse_args()

    result = run_square8(
        shots=arguments.shots,
        rounds=arguments.rounds,
        physical_error_rate=arguments.p,
        seed=arguments.seed,
        ancilla_errors=arguments.ancilla_errors,
    )
    events_per_shot = np.count_nonzero(result.detection_events, axis=1)
    print("Square-8 perimeter syndrome simulation")
    print(f"shots: {result.shots}")
    print(f"syndrome rounds: {result.rounds}")
    print(f"physical error probability: {result.physical_error_rate:g}")
    print(f"ancilla reset/readout errors: {'enabled' if arguments.ancilla_errors else 'disabled'}")
    print(f"detectors per shot: {result.detection_events.shape[1]}")
    print(f"total detection events: {np.count_nonzero(result.detection_events)}")
    print(f"shots with detections: {np.count_nonzero(events_per_shot)}")
    print(f"mean detection events per shot: {np.mean(events_per_shot):.3f}")
    raw_logical_flips = np.count_nonzero(result.observable_flips)
    decoded_logical_failures = np.count_nonzero(result.logical_failures)
    print(f"tracked logical observable: Z{LOGICAL_Z_SUPPORT[0]} Z{LOGICAL_Z_SUPPORT[1]}")
    print(f"raw logical flips: {raw_logical_flips}")
    print(f"raw logical error rate: {raw_logical_flips / result.shots:.6f}")
    print(f"decoded logical failures: {decoded_logical_failures}")
    print(f"decoded logical error rate: {np.mean(result.logical_failures):.6f}")

    if arguments.plot or arguments.save_map:
        plot_square8_map(output_path=arguments.save_map, show=arguments.plot)


if __name__ == "__main__":
    main()