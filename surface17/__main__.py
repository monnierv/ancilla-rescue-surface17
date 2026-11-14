import argparse

from .node_map import plot_code_map
from .simulator import run_simulation, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate one noisy surface-17 logical qubit.")
    parser.add_argument("--shots", type=int, default=1000, help="number of independent memory experiments")
    parser.add_argument("--rounds", type=int, default=3, help="number of syndrome measurement rounds")
    parser.add_argument("--p", type=float, default=0.001, help="physical error probability per modeled location")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed")
    parser.add_argument("--plot", action="store_true", help="display the surface-17 node map")
    parser.add_argument("--save-map", metavar="PATH", help="save the node map image to PATH")
    arguments = parser.parse_args()

    result = run_simulation(
        shots=arguments.shots,
        rounds=arguments.rounds,
        physical_error_rate=arguments.p,
        seed=arguments.seed,
    )
    summary = summarize(result)

    print("Surface-17 logical memory simulation")
    print(f"shots: {summary.shots}")
    print(f"syndrome rounds: {result.rounds}")
    print(f"physical error probability: {result.physical_error_rate:g}")
    print(f"detectors per shot: {summary.detector_count}")
    print(f"total detection events: {summary.total_detection_events}")
    print(f"shots with detections: {summary.shots_with_detections}")
    print(f"mean detection events per shot: {summary.mean_detection_events_per_shot:.3f}")
    print(f"raw logical measurement flips: {summary.raw_logical_flips}")
    print(f"decoded logical failures: {summary.decoded_logical_failures}")
    print(f"decoded logical error rate: {summary.decoded_logical_error_rate:.6f}")

    if arguments.plot or arguments.save_map:
        plot_code_map(output_path=arguments.save_map, show=arguments.plot)


if __name__ == "__main__":
    main()