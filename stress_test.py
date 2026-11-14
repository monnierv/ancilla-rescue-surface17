"""Measure decoded logical failures caused by forced single-qubit faults."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymatching

from qec_config import build_code_graph, build_memory_circuit, load_code_config


@dataclass(frozen=True)
class NodeStress:
    x_error_percent: float
    z_error_percent: float
    x_failures: int
    z_failures: int


@dataclass(frozen=True)
class StressTestResult:
    code_name: str
    trials_per_fault: int
    rounds: int
    node_results: dict[str, NodeStress]


def run_stress_test(
    config_path: str | Path | None,
    default_config: str,
    trials: int = 1000,
    rounds: int = 3,
    physical_error_rate: float = 0.001,
    ancilla_errors: bool = False,
    seed: int | None = None,
) -> StressTestResult:
    if trials <= 0:
        raise ValueError("trials must be greater than zero")

    config = load_code_config(config_path, default_config)
    decoder_error_rate = max(physical_error_rate, 0.001)
    decoder_circuit = build_memory_circuit(
        config,
        rounds,
        decoder_error_rate,
        ancilla_errors=True,
    )
    matching = pymatching.Matching.from_stim_circuit(decoder_circuit)
    node_results = {}
    seed_offset = 0
    for qubit in config["qubits"]:
        qubit_id = qubit["id"]
        rates = {}
        failures_by_pauli = {}
        for pauli in ("X", "Z"):
            circuit = build_memory_circuit(
                config,
                rounds,
                physical_error_rate,
                ancilla_errors,
                forced_fault=(qubit_id, pauli),
            )
            detector_samples, observable_samples = circuit.compile_detector_sampler(
                seed=None if seed is None else seed + seed_offset,
            ).sample(shots=trials, separate_observables=True)
            predictions = matching.decode_batch(detector_samples)
            failures = np.any(predictions != observable_samples, axis=1)
            failure_count = int(np.count_nonzero(failures))
            failures_by_pauli[pauli] = failure_count
            rates[pauli] = 100.0 * failure_count / trials
            seed_offset += 1

        node_results[qubit_id] = NodeStress(
            x_error_percent=rates["X"],
            z_error_percent=rates["Z"],
            x_failures=failures_by_pauli["X"],
            z_failures=failures_by_pauli["Z"],
        )

    return StressTestResult(
        code_name=config["name"],
        trials_per_fault=trials,
        rounds=rounds,
        node_results=node_results,
    )


def plot_stress_test(
    result: StressTestResult,
    config_path: str | Path | None,
    default_config: str,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    config = load_code_config(config_path, default_config)
    graph = build_code_graph(config)
    positions = nx.get_node_attributes(graph, "pos")
    data_nodes = [node for node, attributes in graph.nodes(data=True) if attributes["kind"] == "data"]
    ancilla_nodes = [node for node, attributes in graph.nodes(data=True) if attributes["kind"] == "ancilla"]
    labels: dict[Any, str] = {}
    for node in graph.nodes:
        node_result = result.node_results[node]
        check_label = f" ({graph.nodes[node]['basis']} check)" if node in ancilla_nodes else ""
        labels[node] = (
            f"{node}{check_label}\nX {node_result.x_error_percent:.1f}%  "
            f"Z {node_result.z_error_percent:.1f}%"
        )

    figure, axis = plt.subplots(figsize=(9, 8), facecolor="#f7f5ef")
    axis.set_facecolor("#f7f5ef")
    data_check_edges = [
        (first, second)
        for first, second, attributes in graph.edges(data=True)
        if attributes.get("relation") != "ancilla_stabilizer"
    ]
    parity_edges = [
        (first, second)
        for first, second, attributes in graph.edges(data=True)
        if attributes.get("relation") == "ancilla_stabilizer"
    ]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=data_check_edges,
        ax=axis,
        edge_color="#aab3ae",
        width=1.5,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=parity_edges,
        ax=axis,
        edge_color="#b88760",
        style="dashed",
        width=1.3,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=data_nodes,
        node_shape="o",
        node_color="#167d78",
        edgecolors="#104c4a",
        node_size=2500,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=ancilla_nodes,
        node_shape="s",
        node_color="#e28b42",
        edgecolors="#87512d",
        node_size=2700,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=6, font_color="white", ax=axis)
    axis.set_title(
        f"{result.code_name}\nLogical failure after forced X/Z fault ({result.trials_per_fault} trials each)",
        fontsize=12,
        color="#263330",
    )
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()

    if output_path:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def print_stress_summary(result: StressTestResult) -> None:
    print(f"Stress test: {result.code_name}")
    print(f"rounds: {result.rounds}; trials per node and Pauli: {result.trials_per_fault}")
    print("node      X failure      Z failure")
    for node_id, node_result in result.node_results.items():
        print(
            f"{node_id:<8} {node_result.x_error_percent:>6.1f}% "
            f"({node_result.x_failures}/{result.trials_per_fault})  "
            f"{node_result.z_error_percent:>6.1f}% "
            f"({node_result.z_failures}/{result.trials_per_fault})"
        )