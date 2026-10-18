"""Synthetic CPU checks for prior correction, unique selection and ablation replay."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from test_target_diagnostic import synthetic_panel

from stability_gfn.artifacts import file_hash
from stability_gfn.diversity import (
    corrected_path_prior,
    descendant_log_counts,
    policy_report,
    select_unique,
)
from stability_gfn.diversity_experiment import (
    check_diversity_inputs,
    development_source_hashes,
    evaluate_diversity,
    run_diversity,
)
from stability_gfn.environment import PrefixTree, SequenceContext
from stability_gfn.guidance import CachedTask
from stability_gfn.pipeline import ROOT
from stability_gfn.target_experiment import run_target_diagnostic


class DiversityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.sequences = ["AAAC", "AACA", "ACAA", "CAAA"]
        self.tree = PrefixTree(SequenceContext("AAAA", 1), self.sequences)

    def test_counts_remove_uniform_branching_bias_and_match_path_cache(self):
        log_counts = descendant_log_counts(self.tree)
        original = CachedTask(
            self.tree,
            torch.zeros(len(self.tree.prefixes), 4),
            torch.zeros_like(log_counts).float(),
            torch.zeros(4),
        )
        corrected = CachedTask(
            self.tree, original.features, log_counts.float(), torch.zeros(4)
        )
        torch.testing.assert_close(
            original.base_log_probabilities.exp(),
            torch.tensor([0.125, 0.125, 0.25, 0.5]),
        )
        torch.testing.assert_close(
            corrected.base_log_probabilities.exp(), torch.full((4,), 0.25)
        )
        torch.manual_seed(7)
        logits = torch.randn_like(original.logits)
        cached = CachedTask(
            self.tree, original.features, logits + log_counts.float(), torch.zeros(4)
        )
        paths = corrected_path_prior(self.tree, logits[original.paths])
        torch.testing.assert_close(
            paths,
            cached.base_log_probabilities.double().log_softmax(0),
            atol=1e-6,
            rtol=1e-6,
        )
        head = cached.head(8)
        torch.testing.assert_close(
            cached.all_log_probabilities(head), cached.base_log_probabilities
        )

    def test_deduplicate_before_ranking_no_refill_and_stable_ties(self):
        predictions = torch.tensor([1.0, 3.0, 3.0, 0.0])
        draws = torch.tensor([2, 2, 1, 0, 1, 3])
        self.assertEqual(select_unique(draws, predictions, 2).tolist(), [2, 1])
        self.assertEqual(
            select_unique(torch.tensor([2, 2]), predictions, 32).tolist(), [2]
        )
        inputs = {
            "sequences": self.sequences,
            "reference": "AAAA",
            "reference_label": 0.0,
            "labels": torch.arange(4).double(),
            "scorers": {"mlp": {"predictions": predictions}},
        }
        settings = {"raw_count": 8, "selected_count": 4, "repeats": 3}
        first = policy_report(torch.tensor([0.0, 0.0, 1.0, 0.0]), inputs, settings, 123)
        inputs["labels"] = -inputs["labels"]
        second = policy_report(
            torch.tensor([0.0, 0.0, 1.0, 0.0]), inputs, settings, 123
        )
        self.assertEqual(first["exact"]["expected_unique_raw"], 1)
        self.assertEqual(first["selection"]["unique"]["count"], 1)
        self.assertEqual(first["selection"]["unique"]["fill_fraction"], 0.25)
        for before, after in zip(first["samples"], second["samples"]):
            self.assertEqual(before["raw_indices"], after["raw_indices"])
            self.assertEqual(
                before["unique"]["selected_indices"],
                after["unique"]["selected_indices"],
            )
            self.assertEqual(len(before["raw_indices"]), 8)

    def test_source_integrity_and_configuration_rejections(self):
        import copy

        config = json.loads((ROOT / "configs/diversity.json").read_text())
        for key, value in [
            ("repeats", 0),
            ("selected_count", 257),
            ("seed_offset", -1),
        ]:
            changed = copy.deepcopy(config)
            changed["evaluation"][key] = value
            with self.assertRaises(ValueError):
                check_diversity_inputs(changed)
        changed = copy.deepcopy(config)
        changed["arms"] = ["original"]
        with self.assertRaises(ValueError):
            check_diversity_inputs(changed)

    def test_smoke_original_and_corrected_arms_failures_replay_and_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_config, manifest = synthetic_panel(directory)
            # Any accidental reserved-label read must fail, including during replay.
            for entry in manifest["parents"]:
                folder = Path(source_config["panel"]) / entry["id"]
                for path in folder.glob("*.json"):
                    if (
                        entry["role"] == "confirmation"
                        or path.name == "test_labels.json"
                    ):
                        path.write_text("forbidden reserved data")
            source = directory / "source-run"
            run_target_diagnostic(source_config, source, "cpu", smoke=True)
            entries = json.loads((source / "diagnostic_manifest.json").read_text())[
                "parents"
            ]
            # Fixture deterministically exercises one passing and one failing screen.
            for index, entry in enumerate(entries):
                path = source / entry["id"] / "scorer_fits.json"
                reports = json.loads(path.read_text())
                reports["mlp-0"]["gate_passed"] = index == 0
                path.write_text(json.dumps(reports))
            config = json.loads((ROOT / "configs/diversity.json").read_text())
            config["development_run"] = str(source)
            config["source_sha256"] = development_source_hashes(source)
            original = {
                path: file_hash(path) for path in source.rglob("*") if path.is_file()
            }
            with self.assertRaises(ValueError):
                run_diversity(config, directory / "bad", "cpu", smoke=False)
            with self.assertRaises(ValueError):
                run_diversity(config, directory / "bad", "cuda", smoke=True)
            with self.assertRaises(ValueError):
                check_diversity_inputs(config, smoke=False)
            run = directory / "ablation"
            run_diversity(config, run, "cpu", smoke=True)
            replay = directory / "replay"
            evaluate_diversity(run, replay)
            for path in (run / "evaluation").iterdir():
                if path.name != "evaluation_costs.json":
                    self.assertEqual(
                        path.read_bytes(), (replay / path.name).read_bytes()
                    )
            results = json.loads((run / "evaluation/summary.json").read_text())
            for arm in config["arms"]:
                self.assertEqual(
                    results["parents"][0]["arms"][arm]["fit_status"], "complete"
                )
                self.assertEqual(
                    results["parents"][1]["arms"][arm]["fit_status"],
                    "scorer_gate_failed",
                )
            self.assertTrue(
                all(file_hash(path) == digest for path, digest in original.items())
            )
            with self.assertRaises(FileExistsError):
                run_diversity(config, run, "cpu", smoke=True)
            # Corrupting a fitted task must block replay before loading its contents.
            (run / entries[0]["id"] / "original/task.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                evaluate_diversity(run, directory / "tampered")
            # Source corruption is rejected against the frozen pre-run manifest.
            (source / entries[0]["id"] / "scorer_fits.json").write_text("{}")
            with self.assertRaises(ValueError):
                check_diversity_inputs(config, smoke=True)
            self.assertFalse(torch.cuda.is_initialized())
