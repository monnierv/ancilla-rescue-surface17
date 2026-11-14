"""Load code topology from JSON and build its syndrome-memory circuit."""

import json
import math
from pathlib import Path
import sys
from typing import Any

import networkx as nx
import stim


def load_code_config(config_path: str | Path | None, default_filename: str) -> dict[str, Any]:
    if config_path is not None:
        path = Path(config_path)
    else:
        path = Path(__file__).with_name(default_filename)
        if not path.exists():
            path = Path(sys.prefix) / "share" / "ancilla-rescue-surface17" / default_filename
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("configuration root must be a JSON object")
    if not isinstance(config.get("name"), str) or not config["name"]:
        raise ValueError("configuration must have a non-empty name")

    qubits = config.get("qubits")
    if not isinstance(qubits, list) or not qubits:
        raise ValueError("configuration must contain a non-empty qubits list")
    qubit_by_id = {}
    for qubit in qubits:
        if not isinstance(qubit, dict):
            raise ValueError("each qubit must be an object")
        qubit_id = qubit.get("id")
        if not isinstance(qubit_id, str) or not qubit_id or qubit_id in qubit_by_id:
            raise ValueError("qubit ids must be unique non-empty strings")
        if qubit.get("kind") not in {"data", "ancilla"}:
            raise ValueError(f"qubit {qubit_id} kind must be 'data' or 'ancilla'")
        try:
            coordinates = (float(qubit["x"]), float(qubit["y"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"qubit {qubit_id} must have numeric x and y coordinates") from error
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"qubit {qubit_id} coordinates must be finite")
        qubit_by_id[qubit_id] = qubit

    checks = config.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("configuration must contain a non-empty checks list")
    check_by_ancilla = {}
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("each check must be an object")
        ancilla_id = check.get("ancilla")
        if ancilla_id not in qubit_by_id or qubit_by_id[ancilla_id]["kind"] != "ancilla":
            raise ValueError(f"check ancilla {ancilla_id!r} is not a configured ancilla")
        if ancilla_id in check_by_ancilla:
            raise ValueError(f"ancilla {ancilla_id} is assigned to more than one check")
        if check.get("basis") not in {"X", "Z"}:
            raise ValueError(f"check on {ancilla_id} basis must be 'X' or 'Z'")
        check_by_ancilla[ancilla_id] = check

    connections = config.get("connections")
    if not isinstance(connections, list):
        raise ValueError("configuration must contain a connections list")
    support_by_ancilla = {ancilla_id: [] for ancilla_id in check_by_ancilla}
    seen_connections = set()
    for connection in connections:
        if not isinstance(connection, dict):
            raise ValueError("each connection must be an object")
        ancilla_id = connection.get("ancilla")
        data_id = connection.get("data")
        if ancilla_id not in support_by_ancilla:
            raise ValueError(f"connection references unknown check ancilla {ancilla_id!r}")
        if data_id not in qubit_by_id or qubit_by_id[data_id]["kind"] != "data":
            raise ValueError(f"connection data endpoint {data_id!r} is not a configured data qubit")
        pair = (ancilla_id, data_id)
        if pair in seen_connections:
            raise ValueError(f"duplicate connection from {ancilla_id} to {data_id}")
        seen_connections.add(pair)
        support_by_ancilla[ancilla_id].append(data_id)

    if any(not support for support in support_by_ancilla.values()):
        raise ValueError("every check must connect to at least one data qubit")

    for x_ancilla, x_check in check_by_ancilla.items():
        if x_check["basis"] != "X":
            continue
        x_support = set(support_by_ancilla[x_ancilla])
        for z_ancilla, z_check in check_by_ancilla.items():
            if z_check["basis"] == "Z":
                overlap = x_support.intersection(support_by_ancilla[z_ancilla])
                if len(overlap) % 2:
                    raise ValueError(f"X check {x_ancilla} and Z check {z_ancilla} do not commute")

    observables = config.get("logical_observables")
    if not isinstance(observables, list) or not observables:
        raise ValueError("configuration must contain at least one logical observable")
    for observable in observables:
        if not isinstance(observable, dict) or observable.get("basis") != "Z":
            raise ValueError("this simulator currently supports Z-basis logical observables")
        support = observable.get("qubits")
        if not isinstance(support, list) or not support:
            raise ValueError("each logical observable must have a non-empty qubits list")
        if any(qubit_id not in qubit_by_id or qubit_by_id[qubit_id]["kind"] != "data" for qubit_id in support):
            raise ValueError("logical observables must reference configured data qubits")
        for ancilla_id, check in check_by_ancilla.items():
            if check["basis"] == "X" and len(set(support).intersection(support_by_ancilla[ancilla_id])) % 2:
                raise ValueError(f"logical observable {observable.get('id')} does not commute with X check {ancilla_id}")

    config["_qubit_by_id"] = qubit_by_id
    config["_check_by_ancilla"] = check_by_ancilla
    config["_support_by_ancilla"] = support_by_ancilla
    return config


def build_code_graph(config: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph(name=config["name"])
    check_by_ancilla = config["_check_by_ancilla"]
    for qubit in config["qubits"]:
        qubit_id = qubit["id"]
        check = check_by_ancilla.get(qubit_id)
        label = f"{qubit_id} ({check['basis']})" if check else qubit_id
        graph.add_node(
            qubit_id,
            kind=qubit["kind"],
            label=label,
            pos=(qubit["x"], qubit["y"]),
            basis=check["basis"] if check else None,
        )
    for connection in config["connections"]:
        graph.add_edge(connection["ancilla"], connection["data"])
    return graph


def build_memory_circuit(
    config: dict[str, Any],
    rounds: int,
    physical_error_rate: float,
    ancilla_errors: bool = False,
) -> stim.Circuit:
    if rounds <= 0:
        raise ValueError("rounds must be greater than zero")
    if not math.isfinite(physical_error_rate) or not 0 <= physical_error_rate <= 1:
        raise ValueError("physical_error_rate must be between 0 and 1")

    qubits = config["qubits"]
    qubit_index = {qubit["id"]: index for index, qubit in enumerate(qubits)}
    data_ids = [qubit["id"] for qubit in qubits if qubit["kind"] == "data"]
    ancilla_ids = [qubit["id"] for qubit in qubits if qubit["kind"] == "ancilla"]
    data_targets = [qubit_index[qubit_id] for qubit_id in data_ids]
    ancilla_targets = [qubit_index[qubit_id] for qubit_id in ancilla_ids]
    checks = config["checks"]
    support_by_ancilla = config["_support_by_ancilla"]
    x_ancilla_targets = [qubit_index[check["ancilla"]] for check in checks if check["basis"] == "X"]
    circuit = stim.Circuit()

    for index, qubit in enumerate(qubits):
        circuit.append("QUBIT_COORDS", [index], [qubit["x"], qubit["y"]])
    circuit.append("R", data_targets)
    if physical_error_rate:
        circuit.append("X_ERROR", data_targets, physical_error_rate)

    for round_index in range(rounds):
        circuit.append("R", ancilla_targets)
        if physical_error_rate:
            circuit.append("DEPOLARIZE1", data_targets, physical_error_rate)
            if ancilla_errors:
                circuit.append("X_ERROR", ancilla_targets, physical_error_rate)
            if x_ancilla_targets:
                circuit.append("DEPOLARIZE1", x_ancilla_targets, physical_error_rate)
        if x_ancilla_targets:
            circuit.append("H", x_ancilla_targets)

        for check in checks:
            ancilla_id = check["ancilla"]
            ancilla_target = qubit_index[ancilla_id]
            for data_id in support_by_ancilla[ancilla_id]:
                data_target = qubit_index[data_id]
                targets = (
                    [ancilla_target, data_target]
                    if check["basis"] == "X"
                    else [data_target, ancilla_target]
                )
                circuit.append("CX", targets)
                if physical_error_rate:
                    circuit.append("DEPOLARIZE2", targets, physical_error_rate)

        if x_ancilla_targets:
            circuit.append("H", x_ancilla_targets)
        if ancilla_errors and physical_error_rate:
            circuit.append("X_ERROR", ancilla_targets, physical_error_rate)
        circuit.append("MR", ancilla_targets)
        if ancilla_errors and physical_error_rate:
            circuit.append("X_ERROR", ancilla_targets, physical_error_rate)

        for check_index, check in enumerate(checks):
            if round_index == 0:
                if check["basis"] == "Z":
                    detector_targets = [stim.target_rec(-len(checks) + check_index)]
                else:
                    continue
            else:
                detector_targets = [
                    stim.target_rec(-len(checks) + check_index),
                    stim.target_rec(-2 * len(checks) + check_index),
                ]
            qubit = config["_qubit_by_id"][check["ancilla"]]
            circuit.append(
                "DETECTOR",
                detector_targets,
                [qubit["x"], qubit["y"], round_index],
            )

    circuit.append("M", data_targets, physical_error_rate)
    for check_index, check in enumerate(checks):
        if check["basis"] != "Z":
            continue
        detector_targets = [stim.target_rec(-len(data_ids) - len(checks) + check_index)]
        detector_targets.extend(
            stim.target_rec(-len(data_ids) + qubit_index[data_id])
            for data_id in support_by_ancilla[check["ancilla"]]
        )
        qubit = config["_qubit_by_id"][check["ancilla"]]
        circuit.append(
            "DETECTOR",
            detector_targets,
            [qubit["x"], qubit["y"], rounds],
        )

    for observable_index, observable in enumerate(config["logical_observables"]):
        targets = [
            stim.target_rec(-len(data_ids) + qubit_index[data_id])
            for data_id in observable["qubits"]
        ]
        circuit.append("OBSERVABLE_INCLUDE", targets, observable_index)

    return circuit