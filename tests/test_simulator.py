import unittest

from surface17.node_map import build_code_graph
from surface17.simulator import build_surface17_circuit, run_simulation, summarize


class Surface17SimulatorTests(unittest.TestCase):
    def test_zero_noise_has_no_syndrome_or_logical_failures(self) -> None:
        result = run_simulation(shots=32, rounds=2, physical_error_rate=0, seed=7)

        self.assertEqual(result.detection_events.shape, (32, 16))
        self.assertEqual(result.observable_flips.shape, (32, 1))
        self.assertFalse(result.detection_events.any())
        self.assertFalse(result.observable_flips.any())
        self.assertFalse(result.logical_failures.any())

    def test_summary_counts_syndrome_and_decoder_failures(self) -> None:
        result = run_simulation(shots=20, rounds=3, physical_error_rate=0, seed=11)
        summary = summarize(result)

        self.assertEqual(summary.shots, 20)
        self.assertEqual(summary.detector_count, 24)
        self.assertEqual(summary.total_detection_events, 0)
        self.assertEqual(summary.decoded_logical_failures, 0)
        self.assertEqual(summary.decoded_logical_error_rate, 0)

    def test_node_map_has_nine_data_and_eight_check_nodes(self) -> None:
        graph = build_code_graph()
        kinds = [attributes["kind"] for _, attributes in graph.nodes(data=True)]

        self.assertEqual(len(graph), 17)
        self.assertEqual(kinds.count("data"), 9)
        self.assertEqual(kinds.count("check"), 8)
        self.assertGreater(graph.number_of_edges(), 0)

    def test_invalid_simulation_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_simulation(shots=0)
        with self.assertRaises(ValueError):
            run_simulation(rounds=0)
        with self.assertRaises(ValueError):
            run_simulation(physical_error_rate=float("nan"))

    def test_ancilla_flag_adds_reset_and_measurement_noise(self) -> None:
        without_ancilla_errors = build_surface17_circuit(
            rounds=2,
            physical_error_rate=0.01,
        )
        with_ancilla_errors = build_surface17_circuit(
            rounds=2,
            physical_error_rate=0.01,
            ancilla_errors=True,
        )
        count_x_errors = lambda circuit: sum(
            instruction.name == "X_ERROR" for instruction in circuit.flattened()
        )

        self.assertGreater(count_x_errors(with_ancilla_errors), count_x_errors(without_ancilla_errors))


if __name__ == "__main__":
    unittest.main()