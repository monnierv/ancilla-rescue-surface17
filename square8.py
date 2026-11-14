"""Repeated syndrome detection for an eight-data-qubit square layout."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymatching
import stim

from qec_config import build_code_graph as build_configured_graph
from qec_config import build_memory_circuit, load_code_config


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
    config_path: str | Path | None = None,
) -> stim.Circuit:
    config = load_code_config(config_path, "square8.json")
    return build_memory_circuit(config, rounds, physical_error_rate, ancilla_errors)


def run_square8(
    shots: int = 1000,
    rounds: int = 3,
    physical_error_rate: float = 0.001,
    seed: int | None = None,
    ancilla_errors: bool = False,
    config_path: str | Path | None = None,
) -> Square8Result:
    if shots <= 0:
        raise ValueError("shots must be greater than zero")

    circuit = build_square8_circuit(rounds, physical_error_rate, ancilla_errors, config_path)
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


def build_square8_graph(config_path: str | Path | None = None) -> nx.Graph:
    config = load_code_config(config_path, "square8.json")
    return build_configured_graph(config)


def plot_square8_map(
    output_path: str | None = None,
    show: bool = True,
    config_path: str | Path | None = None,
) -> None:
    graph = build_square8_graph(config_path)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated syndrome detection on an eight-qubit square.")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.001, help="physical error probability")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ancilla-errors", action="store_true", help="include syndrome-ancilla reset/readout errors")
    parser.add_argument("--config", metavar="PATH", help="JSON code configuration (default: square8.json)")
    parser.add_argument("--plot", action="store_true", help="display the square node map")
    parser.add_argument("--save-map", metavar="PATH", help="save the square node map")
    arguments = parser.parse_args()

    config = load_code_config(arguments.config, "square8.json")
    result = run_square8(
        shots=arguments.shots,
        rounds=arguments.rounds,
        physical_error_rate=arguments.p,
        seed=arguments.seed,
        ancilla_errors=arguments.ancilla_errors,
        config_path=arguments.config,
    )
    events_per_shot = np.count_nonzero(result.detection_events, axis=1)
    print(config["name"])
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
    logical_observable = config["logical_observables"][0]
    logical_support = " ".join(logical_observable["qubits"])
    print(f"tracked logical observable: {logical_observable['basis']} {logical_support}")
    print(f"raw logical flips: {raw_logical_flips}")
    print(f"raw logical error rate: {raw_logical_flips / result.shots:.6f}")
    print(f"decoded logical failures: {decoded_logical_failures}")
    print(f"decoded logical error rate: {np.mean(result.logical_failures):.6f}")

    if arguments.plot or arguments.save_map:
        plot_square8_map(
            output_path=arguments.save_map,
            show=arguments.plot,
            config_path=arguments.config,
        )


if __name__ == "__main__":
    main()