"""Synthetic data leakage checks, mixed-length feature alignment and predictor replay."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.multiprotein import (
    check_multiprotein,
    evaluate_multiprotein,
    panel_metrics,
    pooled_features,
    run_multiprotein,
    selection_metrics,
)
from stability_gfn.multiprotein_data import (
    choose_panel,
    group_parents,
    prepare_multiprotein,
)
from stability_gfn.pipeline import ROOT
from stability_gfn.scorer import fit_scorer


class MultiProteinTests(unittest.TestCase):
    def fixture(self, root):
        torch.set_num_threads(1)
        source = root / "source.csv"
        rng = np.random.default_rng(8)
        residues = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        fields = [
            "name",
            "aa_seq",
            "WT_name",
            "WT_cluster",
            "mut_type",
            "dG_ML",
            "ddG_ML",
            "deltaG_95CI",
        ]
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for parent in range(12):
                reference = "".join(rng.choice(residues, size=6 + parent % 3))
                for variant in range(4):
                    sequence = (
                        reference
                        if variant == 0
                        else reference[:variant]
                        + ("A" if reference[variant] != "A" else "G")
                        + reference[variant + 1 :]
                    )
                    writer.writerow(
                        {
                            "name": f"p{parent}-{variant}",
                            "aa_seq": sequence,
                            "WT_name": f"p{parent}",
                            "WT_cluster": str(parent),
                            "mut_type": "wt" if variant == 0 else "mut",
                            "dG_ML": 2 + parent * 0.1 + variant * 0.2,
                            "ddG_ML": 0,
                            "deltaG_95CI": 0.2,
                        }
                    )
        exposure = root / "exposure.json"
        write_json(
            exposure, {"known_parent_names": ["p0"], "known_sequence_sha256": []}
        )
        settings = {
            "source_sha256": file_hash(source),
            "exposure_sha256": file_hash(exposure),
            "min_length": 6,
            "max_length": 8,
            "maximum_ci_width": 1,
            "identity_threshold": 0.95,
            "parent_count": 10,
            "variants_per_parent": 3,
            "selection_seed": 123,
            "split_seed": 42,
        }
        data = root / "data"
        manifest = prepare_multiprotein(
            source, data, exposure, settings, synthetic=True
        )
        scorer = json.loads((ROOT / "configs/biological-pilot.json").read_text())[
            "scorer"
        ]
        config = {
            "data": str(data),
            "manifest_sha256": file_hash(data / "manifest.json"),
            "feature_batch_size": 4,
            "selection_fraction": 0.1,
            "mlp_seeds": [0, 1, 2],
            "scorer": scorer,
        }
        return config, manifest

    def test_groups_connect_transitively_and_exposure_excludes_whole_group(self):
        parents = {
            "a": {
                "reference": "AAAA",
                "clusters": ["shared"],
                "rows": [{"sequence": "AAAA", "stability": 1}],
            },
            "b": {
                "reference": "CCCC",
                "clusters": ["shared"],
                "rows": [{"sequence": "CCCC", "stability": 2}],
            },
            "c": {
                "reference": "GGGG",
                "clusters": ["other"],
                "rows": [{"sequence": "CCCC", "stability": 3}],
            },
            "d": {
                "reference": "TTTT",
                "clusters": ["d"],
                "rows": [{"sequence": "TTTT", "stability": 4}],
            },
        }
        groups, _ = group_parents(parents, 0.9)
        self.assertEqual(groups, {"a": ["a", "b", "c"], "d": ["d"]})
        config = {
            "parent_count": 1,
            "variants_per_parent": 0,
            "selection_seed": 1,
            "split_seed": 1,
        }
        partitions, roles, excluded, _ = choose_panel(
            parents,
            groups,
            {"known_parent_names": ["b"], "known_sequence_sha256": []},
            config,
        )
        self.assertEqual(set(roles), {"d"})
        self.assertEqual(excluded[0]["group"], "a")
        for parent in parents.values():
            for row in parent["rows"]:
                row["stability"] += 1000
        _, unchanged, _, _ = choose_panel(
            parents,
            groups,
            {"known_parent_names": ["b"], "known_sequence_sha256": []},
            config,
        )
        self.assertEqual(roles, unchanged)

    def test_grouped_fixture_mixed_lengths_and_full_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, manifest = self.fixture(root)
            self.assertNotIn("p0", manifest["selected_parent_roles"])
            _, rows = check_multiprotein(config)
            backbone = FrozenProGen.tiny()
            mixed = rows["train"][:6]
            pooled = pooled_features(backbone, mixed, 2)
            independent = torch.cat(
                [backbone.sequence_features([row["sequence"]]) for row in mixed]
            )
            torch.testing.assert_close(pooled, independent, atol=1e-5, rtol=1e-5)
            run = root / "run"
            calls = []

            def checked_fit(*args, **kwargs):
                self.assertFalse((run / "evaluation_labels.json").exists())
                calls.append(True)
                return fit_scorer(*args, **kwargs)

            with patch(
                "stability_gfn.multiprotein.fit_scorer", side_effect=checked_fit
            ):
                run_multiprotein(config, run, "cpu", smoke=True)
            self.assertEqual(len(calls), 3)
            self.assertFalse(torch.cuda.is_initialized())
            result = json.loads((run / "evaluation/results.json").read_text())
            self.assertEqual(result, evaluate_multiprotein(run, root / "replay"))
            self.assertAlmostEqual(
                result["constant"]["pooled"]["lift_over_uniform"], 0, places=10
            )
            with self.assertRaises(FileExistsError):
                run_multiprotein(config, run, "cpu", smoke=True)
            with self.assertRaisesRegex(ValueError, "Real runs require"):
                run_multiprotein(config, root / "bad", "cpu")
            for name, expected in manifest["file_sha256"].items():
                self.assertEqual(file_hash(root / "data" / name), expected)
            (root / "data/test_identities.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "Prepared input changed"):
                check_multiprotein(config)

    def test_selection_ties_and_parent_metrics(self):
        self.assertAlmostEqual(
            selection_metrics([0, 0, 0, 0], [1, 2, 3, 4], 0.25)["lift_over_uniform"], 0
        )
        self.assertEqual(
            selection_metrics([2, 1, 1, 0], [10, 1, 3, 0], 0.5)[
                "selected_measured_mean"
            ],
            6,
        )
        rows = [
            {"parent": "a", "is_reference": True, "stability": 1},
            {"parent": "a", "is_reference": False, "stability": 2},
            {"parent": "b", "is_reference": True, "stability": 3},
            {"parent": "b", "is_reference": False, "stability": 4},
        ]
        result = panel_metrics([1, 2, 3, 4], rows, 0.5)
        self.assertEqual(result["macro_parent"]["rmse"], 0)
        self.assertEqual(result["reference_only"]["count"], 2)
        self.assertEqual(result["macro_parent"]["selection_lift"], 0.5)

    def test_primal_ridge_matches_dual_solution(self):
        torch.manual_seed(1)
        features = torch.randn(12, 4)
        labels = torch.randn(12)
        config = {"ridge_penalties": [2.0], "minimum_spearman": 0}
        model, _ = fit_scorer(
            features, labels, features, labels, config, family="ridge"
        )
        standardized = ((features - model.feature_mean) / model.feature_scale).double()
        targets = ((labels - model.label_mean) / model.label_scale).double()
        dual = standardized.T @ torch.linalg.solve(
            standardized @ standardized.T + 2 * torch.eye(12, dtype=torch.float64),
            targets,
        )
        torch.testing.assert_close(
            model.network.weight[0], dual.float(), atol=1e-6, rtol=1e-6
        )
