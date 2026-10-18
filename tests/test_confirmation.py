"""Synthetic CPU confirmation checks, especially all-parent outcome isolation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from stability_gfn.artifacts import file_hash
from stability_gfn.confirmation import (
    fit_confirmation_heads,
    run_confirmation,
    validate_confirmation_config,
)
from stability_gfn.confirmation_data import prepare_confirmation_support
from stability_gfn.confirmation_evaluation import (
    batch_metrics,
    evaluate_confirmation,
    reveal_labels,
)
from stability_gfn.data import REQUIRED_COLUMNS
from stability_gfn.environment import SequenceContext
from stability_gfn.panel import prepare_panel
from stability_gfn.pipeline import ROOT
from stability_gfn.scorer import fit_scorer


def make_fixture(directory):
    """Two confirmation parents and an anchor, all four-residue artificial data."""
    config = json.loads((ROOT / "configs/confirmation.json").read_text())
    source = directory / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        for parent, reference in enumerate(("AAAA", "CCCC", "DDDD")):
            for index, sequence in enumerate(
                SequenceContext(reference, 1).enumerate_sequences()
            ):
                writer.writerow(
                    {
                        "name": f"{parent}-{index}",
                        "WT_name": reference,
                        "WT_cluster": str(parent),
                        "aa_seq": sequence,
                        "mut_type": "wt" if sequence == reference else "mut",
                        "dG_ML": index / 10,
                        "ddG_ML": "-",
                    }
                )
    panel_settings = dict(
        anchor_parent="AAAA",
        development_parents=1,
        confirmation_parents=2,
        min_variants=10,
        min_length=4,
        max_length=4,
        maximum_reference_identity=0.3,
        split_seed=42,
        inner_seed=43,
        inner_validation_fraction=0.2,
    )
    config["panel"] = str(directory / "panel")
    config["candidate_support"] = str(directory / "support")
    panel = prepare_panel(source, config["panel"], panel_settings, synthetic=True)
    return config, panel


def passing_fixture_fit(*args, **kwargs):
    """Exercise guidance even if the intentionally four-update synthetic fit is weak."""
    model, report = fit_scorer(*args, **kwargs)
    report["gate_passed"] = True
    report["synthetic_forced_screen"] = True
    return model, report


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_support_reconstruction_does_not_open_reserved_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, panel = make_fixture(directory)
            expected = {}
            for entry in panel["parents"]:
                if entry["role"] != "confirmation":
                    continue
                folder = Path(config["panel"]) / entry["id"]
                expected[entry["id"]] = sorted(
                    row["sequence"]
                    for row in json.loads((folder / "test_labels.json").read_text())
                )
                for name in ("test_labels.json", "development.json"):
                    (folder / name).write_text("forbidden reserved outcomes")
            prepare_confirmation_support(config["panel"], config["candidate_support"])
            for identifier, sequences in expected.items():
                self.assertEqual(
                    sequences,
                    json.loads(
                        (
                            Path(config["candidate_support"]) / f"{identifier}.json"
                        ).read_text()
                    ),
                )
            with self.assertRaises(FileExistsError):
                prepare_confirmation_support(
                    config["panel"], config["candidate_support"]
                )

    def test_all_parent_barrier_matched_budgets_and_cpu_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, panel = make_fixture(directory)
            prepare_confirmation_support(config["panel"], config["candidate_support"])
            for entry in panel["parents"]:
                # No development labels are needed in confirmation, even after fitting.
                (Path(config["panel"]) / entry["id"] / "development.json").write_text(
                    "not part of confirmation"
                )
            completed = []

            def record_fit(*args, **kwargs):
                fit_confirmation_heads(*args, **kwargs)
                completed.append(args[1]["id"])

            def guarded_reveal(*args, **kwargs):
                self.assertEqual(len(completed), 2)
                return reveal_labels(*args, **kwargs)

            run = directory / "run"
            with (
                patch(
                    "stability_gfn.confirmation.fit_scorer",
                    side_effect=passing_fixture_fit,
                ),
                patch(
                    "stability_gfn.confirmation.fit_confirmation_heads",
                    side_effect=record_fit,
                ),
                patch(
                    "stability_gfn.confirmation_evaluation.reveal_labels",
                    side_effect=guarded_reveal,
                ),
            ):
                run_confirmation(config, run, "cpu", smoke=True)
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"],
                "smoke_complete",
            )
            self.assertEqual(len(list(run.glob("parent-*/seed-0/gfn.pt"))), 2)
            self.assertFalse((run / "parent-00").exists())
            for entry in panel["parents"][1:]:
                result = json.loads(
                    (run / "evaluation" / entry["id"] / "result.json").read_text()
                )
                self.assertEqual(len(result["exact"]), 4)
                for row in result["sampled"]:
                    self.assertEqual(row["count"], 16 if row["batch"] == "raw" else 4)
                    self.assertEqual(row["measurement_coverage"], 1.0)
                exact = next(
                    row for row in result["exact"] if row["method"] == "exact_target"
                )
                self.assertLess(exact["distribution"]["total_variation"], 1e-6)
            seal = json.loads((run / "fits_complete.json").read_text())
            replay = directory / "replay"
            evaluate_confirmation(run, replay)
            for path in (run / "evaluation").rglob("*"):
                if path.is_file() and path.name != "evaluation_costs.json":
                    self.assertEqual(
                        path.read_bytes(),
                        (replay / path.relative_to(run / "evaluation")).read_bytes(),
                    )
            self.assertTrue(
                all(
                    file_hash(run / name) == digest
                    for name, digest in seal["file_sha256"].items()
                )
            )
            # A partial/deleted seed must invalidate the seal before evaluation.
            (run / "parent-01/seed-0/gfn.pt").write_bytes(b"changed checkpoint")
            with self.assertRaises(ValueError):
                evaluate_confirmation(run, directory / "invalid-replay")
            with self.assertRaises(FileExistsError):
                run_confirmation(config, run, "cpu", smoke=True)
            self.assertFalse(torch.cuda.is_initialized())

    def test_interruption_before_all_fits_keeps_outcomes_unrevealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, _ = make_fixture(directory)
            prepare_confirmation_support(config["panel"], config["candidate_support"])
            run = directory / "run"
            with (
                patch(
                    "stability_gfn.confirmation.fit_scorer",
                    side_effect=passing_fixture_fit,
                ),
                patch(
                    "stability_gfn.confirmation.fit_confirmation_heads",
                    side_effect=RuntimeError("synthetic interruption"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    run_confirmation(config, run, "cpu", smoke=True)
            self.assertFalse(list(run.rglob("evaluation_labels.json")))
            self.assertFalse((run / "fits_complete.json").exists())
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"], "failed"
            )
            with self.assertRaises(ValueError):
                evaluate_confirmation(run, directory / "premature")

    def test_failed_scorer_remains_in_full_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, _ = make_fixture(directory)
            prepare_confirmation_support(config["panel"], config["candidate_support"])

            def failed_fit(*args, **kwargs):
                model, report = fit_scorer(*args, **kwargs)
                report["gate_passed"] = False
                return model, report

            run = directory / "run"
            with patch("stability_gfn.confirmation.fit_scorer", side_effect=failed_fit):
                run_confirmation(config, run, "cpu", smoke=True)
            summary = json.loads((run / "evaluation/summary.json").read_text())
            self.assertEqual(summary["declared_parent_count"], 2)
            self.assertEqual(summary["scorer_passed_parent_count"], 0)
            self.assertFalse(list(run.rglob("gfn.pt")))
            for row in summary["aggregates"]:
                self.assertEqual(
                    row["reported_parents"],
                    0 if row["method"] in ("gfn", "supervised") else 2,
                )
                if row["method"] in ("gfn", "supervised"):
                    self.assertIsNone(row["mean_gain"])

    def test_sampled_metrics_count_duplicate_pairs(self):
        result = batch_metrics(
            torch.tensor([0, 0, 1]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([1.0, 3.0]),
            2.0,
            torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
        )
        self.assertEqual(result["unique_count"], 2)
        self.assertAlmostEqual(result["mean_measured_stability"], 5 / 3, places=6)
        self.assertAlmostEqual(result["mean_pairwise_hamming"], 4 / 3)
        self.assertAlmostEqual(result["duplicate_fraction"], 1 / 3)

    def test_fixed_configuration_and_no_accidental_real_cpu_runs(self):
        config = json.loads((ROOT / "configs/confirmation.json").read_text())
        validate_confirmation_config(config)
        for field, value in (("alpha", 0.5), ("scorer_seed", 1)):
            invalid = dict(config, **{field: value})
            with self.assertRaises(ValueError):
                validate_confirmation_config(invalid)
        config["evaluation"]["filter_pool_count"] = 1024
        with self.assertRaises(ValueError):
            validate_confirmation_config(config)
        config["evaluation"]["filter_pool_count"] = 256
        with self.assertRaises(ValueError):
            run_confirmation(config, "/tmp/never-create-real-confirmation", "cpu")
        with self.assertRaises(ValueError):
            run_confirmation(
                config, "/tmp/never-create-real-confirmation", "cuda", smoke=True
            )


if __name__ == "__main__":
    unittest.main()
