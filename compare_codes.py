"""Compare decoded logical error rates for configured QEC codes."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymatching

from qec_config import build_memory_circuit, load_code_config


@dataclass(frozen=True)
class CodeCurve:
    name: str
    error_rates: np.ndarray
    logical_error_rates: np.ndarray
    standard_errors: np.ndarray
    failures: np.ndarray


@dataclass(frozen=True)
class ComparisonResult:
    shots: int
    rounds: int
    physical_error_rates: np.ndarray
    curves: dict[str, CodeCurve]


def run_comparison(
    shots: int = 5000,
    rounds: int = 3,
    p_min: float = 1e-4,
    p_max: float = 0.1,
    points: int = 20,
    seed: int | None = None,
    ancilla_errors: bool = False,
    surface17_config: str | Path | None = None,
    square8_plus_config: str | Path | None = None,
) -> ComparisonResult:
    if shots <= 0:
        raise ValueError("shots must be greater than zero")
    if rounds <= 0:
        raise ValueError("rounds must be greater than zero")
    if points < 2:
        raise ValueError("points must be at least two")
    if not math.isfinite(p_min) or not math.isfinite(p_max) or not 0 < p_min < p_max <= 1:
        raise ValueError("error-rate range must satisfy 0 < p_min < p_max <= 1")

    physical_error_rates = np.geomspace(p_min, p_max, points)
    configs = {
        "surface17": load_code_config(surface17_config, "surface17.json"),
        "square8_plus": load_code_config(square8_plus_config, "square8_plus.json"),
    }
    curves = {}
    seed_offset = 0

    for code_key, config in configs.items():
        logical_error_rates = []
        standard_errors = []
        failures_per_point = []
        for physical_error_rate in physical_error_rates:
            circuit = build_memory_circuit(
                config,
                rounds=rounds,
                physical_error_rate=float(physical_error_rate),
                ancilla_errors=ancilla_errors,
            )
            detection_events, observable_flips = circuit.compile_detector_sampler(
                seed=None if seed is None else seed + seed_offset,
            ).sample(shots=shots, separate_observables=True)
            matching = pymatching.Matching.from_stim_circuit(circuit)
            predictions = matching.decode_batch(detection_events)
            failures = np.any(predictions != observable_flips, axis=1)
            failure_count = int(np.count_nonzero(failures))
            error_rate = failure_count / shots
            logical_error_rates.append(error_rate)
            standard_errors.append(math.sqrt(error_rate * (1 - error_rate) / shots))
            failures_per_point.append(failure_count)
            seed_offset += 1

        curves[code_key] = CodeCurve(
            name=config["name"],
            error_rates=physical_error_rates.copy(),
            logical_error_rates=np.asarray(logical_error_rates),
            standard_errors=np.asarray(standard_errors),
            failures=np.asarray(failures_per_point, dtype=int),
        )

    return ComparisonResult(
        shots=shots,
        rounds=rounds,
        physical_error_rates=physical_error_rates,
        curves=curves,
    )


def plot_comparison(
    result: ComparisonResult,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5), facecolor="#f7f5ef")
    axis.set_facecolor("#f7f5ef")
    colors = {"surface17": "#167d78", "square8_plus": "#e28b42"}

    for code_key, curve in result.curves.items():
        axis.errorbar(
            curve.error_rates,
            curve.logical_error_rates * 100,
            yerr=curve.standard_errors * 100,
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            color=colors.get(code_key, "#596b9b"),
            label=curve.name,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Physical error probability per modeled location")
    axis.set_ylabel("Decoded logical error rate (%)")
    axis.set_title(f"Logical error rate comparison ({result.rounds} rounds, {result.shots} shots/point)")
    axis.grid(True, which="both", color="#c9cec9", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()

    if output_path:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def print_comparison(result: ComparisonResult) -> None:
    print(f"Decoded logical error rate: {result.rounds} rounds, {result.shots} shots per point")
    print("physical p   " + "   ".join(f"{curve.name}" for curve in result.curves.values()))
    for point_index, physical_error_rate in enumerate(result.physical_error_rates):
        values = []
        for curve in result.curves.values():
            failures = curve.failures[point_index]
            rate_percent = 100 * curve.logical_error_rates[point_index]
            values.append(f"{rate_percent:.3f}% ({failures}/{result.shots})")
        print(f"{physical_error_rate:<11.4g} " + "   ".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot decoded logical error rate versus physical error rate for surface17 and square8_plus."
    )
    parser.add_argument("--shots", type=int, default=5000, help="shots sampled at each physical error rate")
    parser.add_argument("--rounds", type=int, default=3, help="syndrome-measurement rounds")
    parser.add_argument("--p-min", type=float, default=1e-4, help="smallest physical error probability")
    parser.add_argument("--p-max", type=float, default=0.1, help="largest physical error probability")
    parser.add_argument("--points", type=int, default=20, help="number of logarithmically spaced error rates")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed")
    parser.add_argument("--ancilla-errors", action="store_true", help="include syndrome-ancilla reset/readout errors")
    parser.add_argument("--surface17-config", metavar="PATH", help="override surface17 JSON config")
    parser.add_argument("--square8-plus-config", metavar="PATH", help="override square8_plus JSON config")
    parser.add_argument("--save", metavar="PATH", help="save the graph image")
    parser.add_argument("--show", action="store_true", help="show the graph even when saving it")
    arguments = parser.parse_args()

    result = run_comparison(
        shots=arguments.shots,
        rounds=arguments.rounds,
        p_min=arguments.p_min,
        p_max=arguments.p_max,
        points=arguments.points,
        seed=arguments.seed,
        ancilla_errors=arguments.ancilla_errors,
        surface17_config=arguments.surface17_config,
        square8_plus_config=arguments.square8_plus_config,
    )
    print_comparison(result)
    plot_comparison(
        result,
        output_path=arguments.save,
        show=arguments.show or not arguments.save,
    )


if __name__ == "__main__":
    main()
