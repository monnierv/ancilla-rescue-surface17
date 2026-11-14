from dataclasses import dataclass
import math

import numpy as np
import pymatching
import stim


@dataclass(frozen=True)
class SimulationResult:
    shots: int
    rounds: int
    physical_error_rate: float
    detection_events: np.ndarray
    observable_flips: np.ndarray
    predicted_observable_flips: np.ndarray
    logical_failures: np.ndarray


@dataclass(frozen=True)
class SyndromeSummary:
    shots: int
    detector_count: int
    total_detection_events: int
    shots_with_detections: int
    mean_detection_events_per_shot: float
    raw_logical_flips: int
    decoded_logical_failures: int
    decoded_logical_error_rate: float


def run_simulation(
    shots: int = 1000,
    rounds: int = 3,
    physical_error_rate: float = 0.001,
    seed: int | None = None,
    ancilla_errors: bool = False,
) -> SimulationResult:
    if shots <= 0:
        raise ValueError("shots must be greater than zero")
    if rounds <= 0:
        raise ValueError("rounds must be greater than zero")
    if not math.isfinite(physical_error_rate) or not 0 <= physical_error_rate <= 1:
        raise ValueError("physical_error_rate must be between 0 and 1")

    circuit = build_surface17_circuit(rounds, physical_error_rate, ancilla_errors)
    detection_events, observable_flips = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots,
        separate_observables=True,
    )

    matching = pymatching.Matching.from_stim_circuit(circuit)
    predicted_observable_flips = matching.decode_batch(detection_events)
    logical_failures = np.any(
        predicted_observable_flips != observable_flips,
        axis=1,
    )

    return SimulationResult(
        shots=shots,
        rounds=rounds,
        physical_error_rate=physical_error_rate,
        detection_events=detection_events,
        observable_flips=observable_flips,
        predicted_observable_flips=predicted_observable_flips,
        logical_failures=logical_failures,
    )


def build_surface17_circuit(
    rounds: int = 3,
    physical_error_rate: float = 0.001,
    ancilla_errors: bool = False,
) -> stim.Circuit:
    if rounds <= 0:
        raise ValueError("rounds must be greater than zero")
    if not math.isfinite(physical_error_rate) or not 0 <= physical_error_rate <= 1:
        raise ValueError("physical_error_rate must be between 0 and 1")

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=physical_error_rate,
        before_round_data_depolarization=physical_error_rate,
        after_reset_flip_probability=0,
        before_measure_flip_probability=0,
    )

    coordinates = circuit.get_final_qubit_coordinates()
    data_qubits = {
        qubit
        for qubit, coordinate in coordinates.items()
        if int(coordinate[0]) % 2 == 1 and int(coordinate[1]) % 2 == 1
    }
    ancilla_qubits = set(coordinates) - data_qubits
    noisy_circuit = stim.Circuit()

    for instruction in circuit.flattened():
        targets = instruction.targets_copy()
        if instruction.name in {"R", "MR"}:
            data_targets = [target for target in targets if target.value in data_qubits]
            ancilla_targets = [target for target in targets if target.value in ancilla_qubits]

            if instruction.name == "MR" and ancilla_errors and physical_error_rate:
                noisy_circuit.append("X_ERROR", ancilla_targets, physical_error_rate)
            noisy_circuit.append(instruction)

            reset_targets = data_targets
            if ancilla_errors:
                reset_targets = data_targets + ancilla_targets
            if reset_targets and physical_error_rate:
                noisy_circuit.append("X_ERROR", reset_targets, physical_error_rate)
        elif instruction.name == "M":
            data_targets = [target for target in targets if target.value in data_qubits]
            if data_targets and physical_error_rate:
                noisy_circuit.append("X_ERROR", data_targets, physical_error_rate)
            noisy_circuit.append(instruction)
        else:
            noisy_circuit.append(instruction)

    return noisy_circuit


def summarize(result: SimulationResult) -> SyndromeSummary:
    events_per_shot = np.count_nonzero(result.detection_events, axis=1)
    raw_flips = np.any(result.observable_flips, axis=1)
    failures = np.asarray(result.logical_failures, dtype=bool)
    return SyndromeSummary(
        shots=result.shots,
        detector_count=result.detection_events.shape[1],
        total_detection_events=int(np.count_nonzero(result.detection_events)),
        shots_with_detections=int(np.count_nonzero(events_per_shot)),
        mean_detection_events_per_shot=float(np.mean(events_per_shot)),
        raw_logical_flips=int(np.count_nonzero(raw_flips)),
        decoded_logical_failures=int(np.count_nonzero(failures)),
        decoded_logical_error_rate=float(np.mean(failures)),
    )