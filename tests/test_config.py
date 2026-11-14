import json
from pathlib import Path
import tempfile
import unittest

from qec_config import build_code_graph, build_memory_circuit, load_code_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CodeConfigTests(unittest.TestCase):
    def test_default_configs_build_expected_code_sizes(self) -> None:
        expected = {
            "surface17.json": (17, 8, 24),
            "square8.json": (12, 4, 12),
        }

        for filename, (qubit_count, check_count, detector_count) in expected.items():
            with self.subTest(config=filename):
                config = load_code_config(PROJECT_ROOT / filename, filename)
                circuit = build_memory_circuit(config, rounds=3, physical_error_rate=0)
                self.assertEqual(len(config["qubits"]), qubit_count)
                self.assertEqual(len(config["checks"]), check_count)
                self.assertEqual(circuit.num_detectors, detector_count)
                self.assertEqual(circuit.num_observables, 1)

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


if __name__ == "__main__":
    unittest.main()