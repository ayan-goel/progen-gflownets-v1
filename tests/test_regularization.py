"""Synthetic tests for group-held-out selection, fold normalization and input guards."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.regularization import (
    check_regularization,
    cross_validate,
    fixed_ridge,
    grouped_folds,
    run_regularization,
)
from stability_gfn.scorer import StabilityScorer


class RegularizationTests(unittest.TestCase):
    def fixture(self, root, validation_shift=0.0):
        """A small saved-feature run; no test files or backbone are provided."""
        torch.set_num_threads(1)
        rng = np.random.default_rng(5)
        source, data = root / "source", root / "data"
        source.mkdir(parents=True)
        data.mkdir()
        features, rows = {}, {}
        for split, groups in [("train", 6), ("validation", 2)]:
            features[split] = torch.tensor(
                rng.normal(size=(groups * 4, 3)), dtype=torch.float32
            )
            rows[split] = [
                {
                    "parent": f"{split}-parent-{index // 4}",
                    "group": f"{split}-group-{index // 4}",
                    "sequence": f"{split}-synthetic-sequence-{index}",
                    "is_reference": index % 4 == 0,
                    "stability": float(feature[0] - 0.5 * feature[1])
                    + (validation_shift if split == "validation" else 0),
                }
                for index, feature in enumerate(features[split])
            ]
            write_json(data / f"{split}.json", rows[split])
        manifest = {
            "synthetic": True,
            "file_sha256": {
                f"{split}.json": file_hash(data / f"{split}.json") for split in rows
            },
        }
        write_json(data / "manifest.json", manifest)
        write_json(source / "data_manifest.json", manifest)
        write_json(
            source / "config.json",
            {"data": str(data), "manifest_sha256": file_hash(data / "manifest.json")},
        )
        torch.save(
            {
                "features": features,
                "identities": {
                    split: [
                        {key: value for key, value in row.items() if key != "stability"}
                        for row in partition
                    ]
                    for split, partition in rows.items()
                },
            },
            source / "features.pt",
        )
        labels = [row["stability"] for row in rows["train"]]
        model = fixed_ridge(features["train"], labels, features["train"], labels, 1)
        torch.save(
            {"feature_size": 3, "hidden_size": 0, "state_dict": model.state_dict()},
            source / "ridge-seed-0.pt",
        )
        write_json(
            source / "fitting.json",
            {"ridge-seed-0": {"candidates": [{"model": "ridge_1", "rmse": 0.1}]}},
        )
        config = {
            "source_run": str(source),
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
            "ridge_penalties": [1, 10, 100],
            "reference_penalty": 1,
            "folds": 3,
            "fold_seed": 42,
            "selection_fraction": 0.1,
        }
        return config, features, rows

    def test_folds_normalization_and_oof_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            config, features, rows = self.fixture(Path(directory))
            ids, assignments = grouped_folds(rows["train"], 3, 42)
            self.assertEqual(assignments, grouped_folds(rows["train"], 3, 42)[1])
            labels = np.array([row["stability"] for row in rows["train"]])
            seen = []

            def checked_fit(
                train_features, train_labels, held_features, held_labels, penalty
            ):
                model = fixed_ridge(
                    train_features, train_labels, held_features, held_labels, penalty
                )
                torch.testing.assert_close(model.feature_mean, train_features.mean(0))
                self.assertAlmostEqual(
                    model.label_mean.item(), float(np.mean(train_labels)), places=6
                )
                seen.append(model.feature_mean.clone())
                return model

            with patch(
                "stability_gfn.regularization.fixed_ridge", side_effect=checked_fit
            ):
                result = cross_validate(features["train"], rows["train"], config)
            self.assertEqual(len(seen), 9)
            for fold in range(3):
                self.assertEqual(
                    len(
                        {
                            row["group"]
                            for row, index in zip(rows["train"], ids)
                            if index == fold
                        }
                    ),
                    2,
                )
                np.testing.assert_allclose(
                    np.array(result["predictions"]["constant"])[ids == fold],
                    labels[ids != fold].mean(),
                )
            for name, predicted in result["predictions"].items():
                self.assertAlmostEqual(
                    result["parent_balanced_rmse"][name],
                    float(np.sqrt(np.mean((np.array(predicted) - labels) ** 2))),
                )
            selected = result["selected_penalty"]
            self.assertEqual(
                result["parent_balanced_rmse"][str(selected)],
                min(
                    result["parent_balanced_rmse"][str(value)]
                    for value in config["ridge_penalties"]
                ),
            )

    def test_end_to_end_validation_does_not_select_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for index, shift in enumerate([0.0, 100.0]):
                config, features, rows = self.fixture(root / f"case-{index}", shift)
                output = root / f"run-{index}"
                run_regularization(config, output, "cpu", smoke=True)
                results.append(
                    json.loads((output / "cross_validation.json").read_text())
                )
                saved = torch.load(output / "selected-ridge.pt", weights_only=False)
                model = StabilityScorer(saved["feature_size"], saved["hidden_size"])
                model.load_state_dict(saved["state_dict"])
                with torch.no_grad():
                    replay = model(features["validation"]).numpy()
                recorded = json.loads(
                    (output / "validation_predictions.json").read_text()
                )
                np.testing.assert_allclose(
                    replay, recorded["models"]["selected_ridge"], atol=1e-6
                )
                self.assertEqual(
                    json.loads((output / "status.json").read_text())["status"],
                    "smoke_complete",
                )
                self.assertFalse((output / "test_labels.json").exists())
                with self.assertRaises(FileExistsError):
                    run_regularization(config, output, "cpu", smoke=True)
            self.assertEqual(results[0], results[1])

    def test_source_and_execution_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _ = self.fixture(root)
            check_regularization(config)
            with self.assertRaisesRegex(ValueError, "user CUDA"):
                run_regularization(config, root / "forbidden", "cpu")
            self.assertFalse((root / "forbidden").exists())
            (root / "source" / "features.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Source changed"):
                check_regularization(config)
