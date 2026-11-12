"""Synthetic coverage expansion: preserve evaluation, isolate fitting, replay baselines."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_multiprotein as fixtures
import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.coverage import check_coverage, run_coverage, scaled_penalty
from stability_gfn.coverage_data import addition_order, prepare_coverage
from stability_gfn.data import audit_table
from stability_gfn.multiprotein import panel_metrics, run_multiprotein
from stability_gfn.regularization import fixed_ridge
from stability_gfn.scorer import StabilityScorer


class CoverageTests(unittest.TestCase):
    def fixture(self, root):
        config, original = fixtures.MultiProteinTests().fixture(root)
        source = root / "source-run"
        run_multiprotein(config, source, "cpu", smoke=True)
        # All later operations must work without a test-label file.
        (Path(config["data"]) / "test_labels.json").unlink()
        fitting = json.loads((source / "fitting.json").read_text())["ridge-seed-0"]
        choice = min(fitting["candidates"], key=lambda row: row["rmse"])
        reference_penalty = float(choice["model"].removeprefix("ridge_"))
        count = original["parent_counts"]["train"]
        prep = {
            "source_run": str(source),
            "training_parent_counts": [count, count + 1],
            "source_sha256": {
                name: file_hash(source / name)
                for name in [
                    "features.pt",
                    "config.json",
                    "data_manifest.json",
                    "ridge-seed-0.pt",
                    "fitting.json",
                ]
            },
        }
        data = root / "coverage-data"
        manifest = prepare_coverage(prep, data)
        study = {
            "data": str(data),
            "manifest_sha256": file_hash(data / "manifest.json"),
            "reference_penalty": reference_penalty,
            "feature_batch_size": 4,
            "selection_fraction": 0.1,
        }
        return prep, study, manifest, config

    def test_identity_only_order_excludes_all_original_and_exposed_groups(self):
        groups = {
            name: [name]
            for name in ["train", "val", "test", "excluded", "new-a", "new-b"]
        }
        roles = {
            name: {"group": name, "split": role}
            for name, role in [
                ("train", "train"),
                ("val", "validation"),
                ("test", "test"),
            ]
        }
        order = addition_order(groups, roles, [{"group": "excluded"}], 42)
        self.assertEqual({row["parent"] for row in order}, {"new-a", "new-b"})
        self.assertEqual(
            order,
            addition_order(
                dict(reversed(list(groups.items()))), roles, [{"group": "excluded"}], 42
            ),
        )
        for size in [2860, 4160, 5590]:
            self.assertAlmostEqual(scaled_penalty(1000, size, 2860) / size, 1000 / 2860)
        with self.assertRaises(ValueError):
            scaled_penalty(0, 10, 10)

    def test_prepare_run_and_saved_predictions_without_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prep, study, manifest, original = self.fixture(root)
            self.assertEqual(manifest["additional_sequences"], 4)
            _, _, _, files, additional, stages, cache = check_coverage(study)
            before = file_hash(Path(original["data"]) / "validation.json")
            replicate = root / "coverage-reprepared"
            prepare_coverage(prep, replicate)
            self.assertEqual(
                (replicate / "additional_train.json").read_bytes(),
                (Path(study["data"]) / "additional_train.json").read_bytes(),
            )
            output = root / "coverage-run"
            training_labels = {
                row["stability"] for row in files["train.json"] + additional
            }

            def checked_fit(features, labels, metric_features, metric_labels, penalty):
                self.assertTrue(set(labels) <= training_labels)
                np.testing.assert_array_equal(labels, metric_labels)
                torch.testing.assert_close(features, metric_features)
                return fixed_ridge(
                    features, labels, metric_features, metric_labels, penalty
                )

            def checked_metrics(*args):
                self.assertTrue((output / "fitting_complete.json").exists())
                return panel_metrics(*args)

            with (
                patch("stability_gfn.coverage.fixed_ridge", side_effect=checked_fit),
                patch(
                    "stability_gfn.coverage.panel_metrics", side_effect=checked_metrics
                ),
            ):
                run_coverage(study, output, "cpu", smoke=True)
            self.assertEqual(
                before, file_hash(Path(original["data"]) / "validation.json")
            )
            self.assertFalse((Path(original["data"]) / "test_labels.json").exists())
            predicted = json.loads((output / "validation_predictions.json").read_text())
            fits = json.loads((output / "fitting_complete.json").read_text())
            for count in stages:
                saved = torch.load(output / f"ridge-{count}.pt", weights_only=False)
                model = StabilityScorer(saved["feature_size"], saved["hidden_size"])
                model.load_state_dict(saved["state_dict"])
                with torch.no_grad():
                    replay = model(cache["features"]["validation"]).numpy()
                np.testing.assert_allclose(replay, predicted[count]["ridge"], atol=1e-6)
                self.assertAlmostEqual(
                    fits[count]["penalty"] / fits[count]["training_rows"],
                    study["reference_penalty"] / len(files["train.json"]),
                )
            self.assertLess(
                json.loads((output / "baseline_check.json").read_text())[
                    "max_prediction_difference"
                ],
                1e-4,
            )
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"],
                "smoke_complete",
            )
            with self.assertRaises(FileExistsError):
                run_coverage(study, output, "cpu", smoke=True)
            with self.assertRaisesRegex(ValueError, "user CUDA"):
                run_coverage(study, root / "forbidden", "cpu")
            with self.assertRaises(FileExistsError):
                prepare_coverage(prep, Path(study["data"]))
            raw_parent = additional[0]["parent"]
            selected, _ = audit_table(
                Path(original["data"]).parent / "source.csv", 6, 8, {raw_parent}
            )
            self.assertEqual(set(selected), {raw_parent})

    def test_overlap_and_hash_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            prep, config, manifest, _ = self.fixture(Path(directory))
            _, _, _, files, additional, _, _ = check_coverage(config)
            data = Path(config["data"])
            additional[0]["group"] = files["validation.json"][0]["group"]
            write_json(data / "additional_train.json", additional)
            with self.assertRaisesRegex(ValueError, "input changed"):
                check_coverage(config)
            manifest["file_sha256"]["additional_train.json"] = file_hash(
                data / "additional_train.json"
            )
            write_json(data / "manifest.json", manifest)
            config["manifest_sha256"] = file_hash(data / "manifest.json")
            with self.assertRaisesRegex(ValueError, "group overlap"):
                check_coverage(config)
            prep["training_parent_counts"] = [7, 1000]
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                prepare_coverage(prep, Path(directory) / "too-many")
