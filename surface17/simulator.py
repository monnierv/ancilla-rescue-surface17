from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymatching
import stim

from qec_config import build_memory_circuit, load_code_config


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
    config_path: str | Path | None = None,
) -> SimulationResult:
    if shots <= 0:
        raise ValueError("shots must be greater than zero")

    circuit = build_surface17_circuit(rounds, physical_error_rate, ancilla_errors, config_path)
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
    config_path: str | Path | None = None,
) -> stim.Circuit:
    config = load_code_config(config_path, "surface17.json")
    return build_memory_circuit(config, rounds, physical_error_rate, ancilla_errors)


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