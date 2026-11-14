import unittest

from square8 import CHECKS, build_square8_circuit, build_square8_graph, run_square8


class Square8Tests(unittest.TestCase):
    def test_zero_noise_has_no_temporal_syndrome_events(self) -> None:
        result = run_square8(shots=24, rounds=3, physical_error_rate=0, seed=7)

        self.assertEqual(result.detection_events.shape, (24, 10))
        self.assertEqual(result.observable_flips.shape, (24, 1))
        self.assertEqual(result.predicted_observable_flips.shape, (24, 1))
        self.assertFalse(result.detection_events.any())
        self.assertFalse(result.observable_flips.any())
        self.assertFalse(result.logical_failures.any())

    def test_noise_produces_syndrome_events(self) -> None:
        result = run_square8(shots=100, rounds=3, physical_error_rate=0.05, seed=12)

        self.assertGreater(result.detection_events.sum(), 0)
        self.assertEqual(result.logical_failures.shape, (100,))

    def test_square_map_has_eight_data_and_four_edge_ancillas(self) -> None:
        graph = build_square8_graph()
        kinds = [attributes["kind"] for _, attributes in graph.nodes(data=True)]

        self.assertEqual(len(graph), 12)
        self.assertEqual(kinds.count("data"), 8)
        self.assertEqual(kinds.count("ancilla"), 4)
        self.assertTrue(all(graph.degree(f"A{index}") == 4 for index in range(1, 5)))

    def test_mixed_basis_checks_commute(self) -> None:
        x_checks = [support for _, basis, support, _, _ in CHECKS if basis == "X"]
        z_checks = [support for _, basis, support, _, _ in CHECKS if basis == "Z"]

        for x_support in x_checks:
            for z_support in z_checks:
                self.assertEqual(len(set(x_support) & set(z_support)) % 2, 0)

    def test_detector_count_and_parameters(self) -> None:
        circuit = build_square8_circuit(rounds=4, physical_error_rate=0)

        self.assertEqual(circuit.num_detectors, 14)
        with self.assertRaises(ValueError):
            run_square8(shots=0)
        with self.assertRaises(ValueError):
            build_square8_circuit(rounds=0)
        with self.assertRaises(ValueError):
            build_square8_circuit(physical_error_rate=float("nan"))

    def test_ancilla_flag_adds_reset_and_measurement_noise(self) -> None:
        without_ancilla_errors = build_square8_circuit(rounds=2, physical_error_rate=0.01)
        with_ancilla_errors = build_square8_circuit(
            rounds=2,
            physical_error_rate=0.01,
            ancilla_errors=True,
        )
        count_x_errors = lambda circuit: sum(
            instruction.name == "X_ERROR" for instruction in circuit.flattened()
        )

        self.assertEqual(count_x_errors(without_ancilla_errors), 0)
        self.assertGreater(count_x_errors(with_ancilla_errors), 0)


if __name__ == "__main__":
    unittest.main()