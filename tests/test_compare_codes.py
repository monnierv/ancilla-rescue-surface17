from pathlib import Path
import tempfile
import unittest

from compare_codes import plot_comparison, run_comparison


class CodeComparisonTests(unittest.TestCase):
    def test_comparison_builds_two_curves_and_saves_plot(self) -> None:
        result = run_comparison(
            shots=20,
            rounds=2,
            p_min=0.001,
            p_max=0.01,
            points=2,
            seed=13,
        )

        self.assertEqual(set(result.curves), {"surface17", "square8_plus"})
        self.assertEqual(result.physical_error_rates.shape, (2,))
        for curve in result.curves.values():
            self.assertEqual(curve.logical_error_rates.shape, (2,))
            self.assertEqual(curve.failures.shape, (2,))
            self.assertTrue(((curve.logical_error_rates >= 0) & (curve.logical_error_rates <= 1)).all())

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "comparison.png"
            plot_comparison(result, output_path=output_path, show=False)
            self.assertGreater(output_path.stat().st_size, 0)

    def test_invalid_comparison_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shots must be greater than zero"):
            run_comparison(shots=0)
        with self.assertRaisesRegex(ValueError, "error-rate range"):
            run_comparison(p_min=0, p_max=0.1)


if __name__ == "__main__":
    unittest.main()