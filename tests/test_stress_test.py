import unittest

from stress_test import run_stress_test


class StressTestTests(unittest.TestCase):
    def test_runs_x_and_z_trials_for_each_configured_node(self) -> None:
        result = run_stress_test(
            config_path="square8.json",
            default_config="square8.json",
            trials=3,
            rounds=2,
            physical_error_rate=0,
            seed=17,
        )

        self.assertEqual(len(result.node_results), 12)
        self.assertEqual(result.trials_per_fault, 3)
        for node_result in result.node_results.values():
            self.assertIn(node_result.x_error_percent, {0, 100 / 3, 200 / 3, 100})
            self.assertIn(node_result.z_error_percent, {0, 100 / 3, 200 / 3, 100})

    def test_rejects_nonpositive_trial_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "trials must be greater than zero"):
            run_stress_test(
                config_path="square8.json",
                default_config="square8.json",
                trials=0,
            )


if __name__ == "__main__":
    unittest.main()