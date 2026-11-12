"""Check the diagnostic identity using unequal parent sizes and known offsets."""

import unittest

from stability_gfn.predictor_diagnostics import decompose_errors


class PredictorDiagnosticsTests(unittest.TestCase):
    def test_one_parent_has_no_leave_parent_out_metric(self):
        rows = [{"parent": "a", "stability": 1.0}, {"parent": "a", "stability": 3.0}]
        result = decompose_errors([2.0, 2.0], rows, baseline=2.0)
        self.assertIsNone(result["per_parent"][0]["leave_parent_out_excess_mse"])

    def test_weighted_decomposition_and_leave_parent_out(self):
        rows = [
            {"parent": "a", "stability": 1.0},
            {"parent": "a", "stability": 3.0},
            {"parent": "b", "stability": 2.0},
        ]
        result = decompose_errors([4.0, 4.0, 1.0], rows, baseline=2.0)
        self.assertAlmostEqual(result["mse"], 11 / 3)
        self.assertAlmostEqual(result["parent_mean_error_mse"], 3.0)
        self.assertAlmostEqual(result["centered_variant_mse"], 2 / 3)
        self.assertAlmostEqual(
            result["per_parent"][0]["leave_parent_out_excess_mse"], 1.0
        )
        self.assertAlmostEqual(
            result["per_parent"][1]["leave_parent_out_excess_mse"], 4.0
        )

    def test_parent_offset_is_not_variant_error(self):
        rows = [
            {"parent": "a", "stability": 1.0},
            {"parent": "a", "stability": 3.0},
            {"parent": "b", "stability": 2.0},
            {"parent": "b", "stability": 4.0},
        ]
        result = decompose_errors([3.0, 5.0, 1.0, 3.0], rows, baseline=2.5)
        self.assertAlmostEqual(result["parent_mean_error_fraction"], 1.0)
        self.assertAlmostEqual(result["centered_variant_mse"], 0.0)
