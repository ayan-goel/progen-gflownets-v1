"""Synthetic-only checks for diagnostic probabilities, splitting and reserved data."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen, cache_tree
from stability_gfn.data import REQUIRED_COLUMNS
from stability_gfn.environment import PrefixTree, SequenceContext
from stability_gfn.guidance import CachedTask
from stability_gfn.panel import (
    prepare_panel,
    read_development_parent,
    reference_identity,
)
from stability_gfn.pipeline import ROOT
from stability_gfn.scorer import fit_scorer
from stability_gfn.target_diagnostics import (
    candidate_priors,
    evaluate_inputs,
    extract_diagnostic_cache,
)
from stability_gfn.target_experiment import (
    evaluate_target_run,
    run_target_diagnostic,
    validate_target_config,
)


def synthetic_panel(directory):
    """Three dissimilar four-residue references; labels are arbitrary fixture values."""
    source = directory / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        for parent, reference in enumerate(("AAAA", "CCCC", "DDDD")):
            sequences = SequenceContext(reference, 1).enumerate_sequences()
            for index, sequence in enumerate(sequences):
                writer.writerow(
                    {
                        "name": f"{parent}-{index}",
                        "aa_seq": sequence,
                        "WT_name": reference,
                        "WT_cluster": str(parent),
                        "mut_type": "wt" if sequence == reference else "mut",
                        "dG_ML": index / 10,
                        "ddG_ML": "-",
                    }
                )
    config = json.loads((ROOT / "configs/target-diagnostic.json").read_text())
    config["panel"] = str(directory / "panel")
    config["panel_preparation"].update(
        anchor_parent="AAAA",
        development_parents=2,
        confirmation_parents=1,
        min_length=4,
        max_length=4,
        min_variants=10,
    )
    manifest = prepare_panel(
        source, config["panel"], config["panel_preparation"], synthetic=True
    )
    return config, manifest


class TargetDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_teacher_forced_paths_match_cached_prefix_tree(self):
        """Causal alignment must recover the old prior, including forced suffixes."""
        backbone = FrozenProGen.tiny()
        sequences = ["AAAC", "AACA", "ACAA", "CAAA"]
        rows = [{"sequence": sequence} for sequence in sequences]
        cache = extract_diagnostic_cache(backbone, rows, 2, include_logits=True)
        tree = PrefixTree(SequenceContext("AAAA", 1), sequences)
        features, logits = cache_tree(backbone, tree, 2)
        task = CachedTask(tree, features, logits, torch.zeros(len(sequences)))
        probabilities = candidate_priors(sequences, "AAAA", cache["path_logits"])
        torch.testing.assert_close(
            probabilities["local"],
            task.base_log_probabilities.double().log_softmax(0),
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            cache["features"], backbone.sequence_features(sequences)
        )
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(any(p.requires_grad for p in backbone.model.parameters()))

    def test_uniform_local_hazard_and_exact_target_expectations(self):
        sequences = ["AAAC", "AACA", "ACAA", "CAAA"]
        logits = torch.zeros(4, 4, 20)
        priors = candidate_priors(sequences, "AAAA", logits)
        # One available mutation at each position: local hazards favor early sites.
        torch.testing.assert_close(
            priors["local"].exp(),
            torch.tensor([0.125, 0.125, 0.25, 0.5], dtype=torch.float64),
        )
        torch.testing.assert_close(
            priors["complete_20aa"].exp(), torch.full((4,), 0.25, dtype=torch.float64)
        )
        inputs = {
            "sequences": sequences,
            "reference": "AAAA",
            "reference_label": 0.0,
            "labels": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "label_scale": 1.0,
            "path_logits": logits,
            "scorers": {
                "perfect": {
                    "predictions": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                    "reference_prediction": 0.0,
                }
            },
            "ensemble_members": torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 5),
            "ensemble_reference_predictions": torch.zeros(5),
        }
        settings = {
            "alphas": [1.0, 0.5, 0.25],
            "beta": 0.0,
            "raw_count": 16,
            "selected_count": 4,
        }
        result = evaluate_inputs(inputs, settings)
        for row in result["comparisons"]:
            self.assertAlmostEqual(row["measured_gain_vs_matched_prior"], 0.0)
            self.assertEqual(row["prior_probabilities"], row["target_probabilities"])
            self.assertEqual(len(row["matched_samples"]["prior"][0]["raw_indices"]), 16)
        settings["beta"] = 1.0
        tilted = evaluate_inputs(inputs, settings)
        self.assertTrue(
            all(
                row["measured_gain_vs_matched_prior"] > 0
                for row in tilted["comparisons"]
            )
        )
        expected = (priors["local"] + inputs["labels"]).softmax(0)
        torch.testing.assert_close(
            torch.tensor(
                tilted["comparisons"][0]["target_probabilities"], dtype=torch.float64
            ),
            expected,
        )
        changed = copy.deepcopy(inputs)
        changed["labels"] = -inputs["labels"]
        reversed_labels = evaluate_inputs(changed, settings)
        self.assertEqual(
            tilted["comparisons"][0]["target_probabilities"],
            reversed_labels["comparisons"][0]["target_probabilities"],
        )
        self.assertEqual(
            tilted["comparisons"][0]["matched_samples"]["exact_target"][0][
                "selected_indices"
            ],
            reversed_labels["comparisons"][0]["matched_samples"]["exact_target"][0][
                "selected_indices"
            ],
        )

    def test_panel_splits_coverage_selection_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config, manifest = synthetic_panel(directory)
            self.assertEqual(
                [p["parent"] for p in manifest["parents"]], ["AAAA", "CCCC", "DDDD"]
            )
            self.assertEqual(
                [p["role"] for p in manifest["parents"]],
                ["development", "development", "confirmation"],
            )
            entry = manifest["parents"][0]
            rows = read_development_parent(config["panel"], entry)
            sets = [{row["sequence"] for row in values} for values in rows.values()]
            self.assertFalse(
                sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
            )
            self.assertIn("AAAA", sets[0])
            test_rows = json.loads(
                (Path(config["panel"]) / entry["id"] / "test_labels.json").read_text()
            )
            self.assertFalse(set.union(*sets) & {row["sequence"] for row in test_rows})
            self.assertAlmostEqual(reference_identity("AAAA", "AAAA"), 1.0)
            self.assertAlmostEqual(reference_identity("AAAA", "CCCC"), 0.0)
            with self.assertRaises(ValueError):
                read_development_parent(config["panel"], manifest["parents"][-1])
            with self.assertRaises(FileExistsError):
                prepare_panel(
                    directory / "source.csv",
                    config["panel"],
                    config["panel_preparation"],
                    synthetic=True,
                )
            # Parent selection is unchanged by reversing every outcome value.
            source = directory / "source.csv"
            with source.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            for row in csv_rows:
                row["dG_ML"] = str(-float(row["dG_ML"]))
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
                writer.writeheader()
                writer.writerows(csv_rows)
            alternate = prepare_panel(
                source,
                directory / "alternate",
                config["panel_preparation"],
                synthetic=True,
            )
            self.assertEqual(
                [(p["parent"], p["role"]) for p in manifest["parents"]],
                [(p["parent"], p["role"]) for p in alternate["parents"]],
            )
            path = Path(config["panel"]) / entry["id"] / "fit.json"
            path.write_text("[]")
            with self.assertRaises(ValueError):
                read_development_parent(config["panel"], entry)

    def test_smoke_roundtrip_and_reserved_file_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config, manifest = synthetic_panel(directory)
            # Make every reserved file unreadable as JSON: runner must never consume it.
            for entry in manifest["parents"]:
                folder = Path(config["panel"]) / entry["id"]
                for path in folder.glob("*.json"):
                    if (
                        entry["role"] == "confirmation"
                        or path.name == "test_labels.json"
                    ):
                        path.write_text("reserved: do not read")
            run = directory / "run"
            with patch(
                "stability_gfn.target_experiment.fit_scorer",
                wraps=fit_scorer,
            ) as fit:
                run_target_diagnostic(config, run, "cpu", smoke=True)
                self.assertEqual(fit.call_count, 12)
                for call in fit.call_args_list:
                    self.assertEqual(len(call.args[1]), 44)
                    self.assertEqual(len(call.args[3]), 10)
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"],
                "smoke_complete",
            )
            self.assertFalse((run / manifest["parents"][-1]["id"]).exists())
            self.assertFalse(list(run.rglob("test_labels.json")))
            self.assertFalse(list(run.rglob("gfn.pt")))
            original_hashes = {
                path: file_hash(path) for path in run.rglob("*") if path.is_file()
            }
            repeated = directory / "repeated"
            evaluate_target_run(run, repeated)
            for original in (run / "evaluation").iterdir():
                self.assertEqual(
                    original.read_bytes(), (repeated / original.name).read_bytes()
                )
            self.assertTrue(
                all(
                    file_hash(path) == digest
                    for path, digest in original_hashes.items()
                )
            )
            with self.assertRaises(FileExistsError):
                run_target_diagnostic(config, run, "cpu", smoke=True)
            with self.assertRaises(ValueError):
                run_target_diagnostic(config, directory / "invalid", "cpu", smoke=False)
            with self.assertRaises(ValueError):
                run_target_diagnostic(config, directory / "invalid", "cuda", smoke=True)
            self.assertFalse(torch.cuda.is_initialized())

    def test_config_rejects_unsupported_parameters(self):
        config = json.loads((ROOT / "configs/target-diagnostic.json").read_text())
        validate_target_config(config)
        for key, value in [
            ("alphas", [0.0]),
            ("alphas", [1.0, 1.0]),
            ("beta", float("nan")),
            ("ensemble_seeds", [0]),
            ("selected_count", 9999),
        ]:
            invalid = copy.deepcopy(config)
            invalid[key] = value
            with self.assertRaises(ValueError):
                validate_target_config(invalid)


if __name__ == "__main__":
    unittest.main()
