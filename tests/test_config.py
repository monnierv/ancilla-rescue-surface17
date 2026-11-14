import json
from pathlib import Path
import tempfile
import unittest

from qec_config import build_code_graph, build_memory_circuit, load_code_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _gf2_rank(rows: list[int]) -> int:
    pivots = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return len(pivots)


class CodeConfigTests(unittest.TestCase):
    def test_default_configs_build_expected_code_sizes(self) -> None:
        expected = {
            "surface17.json": (17, 8, 24),
            "square8.json": (12, 4, 12),
            "square8_plus.json": (17, 8, 25),
        }

        for filename, (qubit_count, check_count, detector_count) in expected.items():
            with self.subTest(config=filename):
                config = load_code_config(PROJECT_ROOT / filename, filename)
                circuit = build_memory_circuit(config, rounds=3, physical_error_rate=0)
                self.assertEqual(len(config["qubits"]), qubit_count)
                self.assertEqual(len(config["checks"]), check_count)
                self.assertEqual(circuit.num_detectors, detector_count)
                self.assertEqual(circuit.num_observables, 1)

    def test_square8_plus_has_nine_ancillas_and_one_logical_qubit(self) -> None:
        square8_plus = load_code_config(PROJECT_ROOT / "square8_plus.json", "square8_plus.json")
        data_ids = [qubit["id"] for qubit in square8_plus["qubits"] if qubit["kind"] == "data"]
        ancilla_ids = {qubit["id"] for qubit in square8_plus["qubits"] if qubit["kind"] == "ancilla"}
        support = {check["ancilla"]: set() for check in square8_plus["checks"]}
        for connection in square8_plus["connections"]:
            support[connection["ancilla"]].add(connection["data"])

        self.assertEqual(len(data_ids), 8)
        self.assertEqual(len(ancilla_ids), 9)
        self.assertEqual(len(square8_plus["checks"]), 8)
        self.assertEqual(sum(check["basis"] == "X" for check in square8_plus["checks"]), 4)
        self.assertEqual(sum(check["basis"] == "Z" for check in square8_plus["checks"]), 4)
        self.assertTrue(all(len(support[check["ancilla"]]) == 2 for check in square8_plus["checks"][:4]))
        self.assertTrue(all(len(support[check["ancilla"]]) == 4 for check in square8_plus["checks"][4:]))

        for data_id in data_ids:
            self.assertEqual(
                sum(data_id in support[check["ancilla"]] for check in square8_plus["checks"] if check["basis"] == "X"),
                1,
            )

        data_index = {data_id: index for index, data_id in enumerate(data_ids)}
        stabilizer_rows = []
        for check in square8_plus["checks"]:
            x_bits = 0
            z_bits = 0
            for data_id in support[check["ancilla"]]:
                bit = 1 << data_index[data_id]
                if check["basis"] == "X":
                    x_bits |= bit
                else:
                    z_bits |= bit
            stabilizer_rows.append(x_bits | (z_bits << len(data_ids)))
        self.assertEqual(_gf2_rank(stabilizer_rows), 7)

        y_check = square8_plus["ancilla_checks"][0]
        self.assertEqual(y_check["ancilla"], "A9")
        self.assertEqual(y_check["basis"], "Y")
        self.assertEqual(set(y_check["qubits"]), ancilla_ids - {"A9"})

        graph = build_code_graph(square8_plus)
        self.assertEqual(graph.nodes["A9"]["basis"], "Y")
        circuit = build_memory_circuit(square8_plus, rounds=3, physical_error_rate=0)
        self.assertEqual(circuit.num_observables, 1)
        self.assertEqual(circuit.compile_detector_sampler(seed=2).sample(shots=8).any(), False)

    def test_square8_plus_a9_fault_trips_y_parity_detector(self) -> None:
        config = load_code_config(PROJECT_ROOT / "square8_plus.json", "square8_plus.json")
        circuit = build_memory_circuit(
            config,
            rounds=2,
            physical_error_rate=0,
            forced_fault=("A9", "X"),
        )
        detectors, logical_observables = circuit.compile_detector_sampler(seed=4).sample(
            shots=8,
            separate_observables=True,
        )

        self.assertTrue(detectors[:, -1].all())
        self.assertFalse(logical_observables.any())

    def test_forced_data_fault_changes_sampled_logical_observable(self) -> None:
        config = load_code_config(PROJECT_ROOT / "square8_plus.json", "square8_plus.json")
        circuit = build_memory_circuit(
            config,
            rounds=2,
            physical_error_rate=0,
            forced_fault=("D1", "X"),
        )
        _, logical_observables = circuit.compile_detector_sampler(seed=4).sample(
            shots=8,
            separate_observables=True,
        )

        self.assertTrue(logical_observables[:, 0].all())

    def test_map_coordinates_come_from_json(self) -> None:
        source = PROJECT_ROOT / "square8.json"
        with source.open(encoding="utf-8") as config_file:
            config_data = json.load(config_file)
        config_data["qubits"][0]["x"] = -3

        with tempfile.TemporaryDirectory() as temporary_directory:
            custom_path = Path(temporary_directory) / "custom-square.json"
            with custom_path.open("w", encoding="utf-8") as config_file:
                json.dump(config_data, config_file)
            config = load_code_config(custom_path, "square8.json")
            graph = build_code_graph(config)

        self.assertEqual(graph.nodes["D1"]["pos"], (-3.0, 1.0))

    def test_noncommuting_check_config_is_rejected(self) -> None:
        source = PROJECT_ROOT / "square8.json"
        with source.open(encoding="utf-8") as config_file:
            config_data = json.load(config_file)
        config_data["connections"].remove({"ancilla": "A4", "data": "D2"})

        with tempfile.TemporaryDirectory() as temporary_directory:
            custom_path = Path(temporary_directory) / "invalid-square.json"
            with custom_path.open("w", encoding="utf-8") as config_file:
                json.dump(config_data, config_file)
            with self.assertRaisesRegex(ValueError, "do not commute"):
                load_code_config(custom_path, "square8.json")

    def test_forced_fault_is_inserted_on_the_named_qubit(self) -> None:
        config = load_code_config(PROJECT_ROOT / "square8.json", "square8.json")
        circuit = build_memory_circuit(
            config,
            rounds=2,
            physical_error_rate=0,
            forced_fault=("A1", "Z"),
        )

        self.assertIn("Z_ERROR(1) 8", str(circuit))
        with self.assertRaisesRegex(ValueError, "unknown qubit"):
            build_memory_circuit(
                config,
                rounds=2,
                physical_error_rate=0,
                forced_fault=("missing", "X"),
            )
        with self.assertRaisesRegex(ValueError, "must be 'X' or 'Z'"):
            build_memory_circuit(
                config,
                rounds=2,
                physical_error_rate=0,
                forced_fault=("D1", "Y"),
            )


if __name__ == "__main__":
    unittest.main()