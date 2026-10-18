"""Independent hand-calculated checks of sampling and held-out-label metrics."""

import unittest

from stability_gfn.metrics import sample_metrics


class MetricTests(unittest.TestCase):
    def test_duplicates_partial_coverage_and_reference_comparisons(self):
        result = sample_metrics(
            ["AA", "AG", "AG"],
            "AA",
            {"AA"},
            {"AA": 1.0, "AG": 3.0},
            measured={"AG": 2.0},
            reference_label=1.5,
            reference_prediction=1.0,
        )
        self.assertAlmostEqual(result["unique_fraction"], 2 / 3)
        self.assertAlmostEqual(result["mean_pairwise_hamming"], 2 / 3)
        self.assertAlmostEqual(result["measurement_coverage"], 2 / 3)
        self.assertAlmostEqual(result["mean_measured_improvement"], 0.5)
        self.assertAlmostEqual(result["mean_predicted_improvement"], 4 / 3)
        self.assertEqual(result["fraction_measured_stabilized"], 1)
        self.assertEqual(result["predictor_mae_on_covered"], 1)
        self.assertEqual(result["mutation_counts"], {"0": 1, "1": 2})


if __name__ == "__main__":
    unittest.main()
