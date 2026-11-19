"""Tiny CPU tests for the stretched schedule and fresh preserved-policy controls."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_budget_data import budget_fixture
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.budget_experiment import reevaluate_budget, run_budget
from stability_gfn.idr.lastblock_evaluation import read_jsonl


class BudgetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_stretched_schedule_preserved_controls_and_snapshot_recovery(self):
        backbone = FrozenProGen.tiny(11)
        reference = {
            key: value.clone() for key, value in backbone.model.state_dict().items()
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real ProGen"),
            ),
        ):
            root = Path(directory)
            (root / "history").mkdir()
            settings = budget_fixture(root / "history", backbone)
            old_files = {
                path: file_hash(path)
                for path in (root / "history").rglob("*")
                if path.is_file()
            }
            output = root / "budget"
            run_budget(settings, output, "cpu", backbone, fixture_predictor)
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "complete"
            )
            rows = read_jsonl(output / "samples.jsonl")
            self.assertEqual(len(rows), 48)
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
            self.assertEqual(
                {row["arm"] for row in rows},
                {
                    "prior",
                    "lastblock_400",
                    "lastblock_tb",
                    "frozen_tb",
                    "unchanged",
                    "continued_supervised",
                },
            )
            for replica in settings["replicates"]:
                name = replica["name"]
                source = root / "history/confirmation" / name / "fit"
                fit = output / name / "fit"
                control = output / name / "control400"
                self.assertEqual(
                    file_hash(control / "checkpoint.pt"),
                    file_hash(source / "lastblock_tb/checkpoint.pt"),
                )
                self.assertEqual(
                    file_hash(fit / "initial.pt"), file_hash(source / "initial.pt")
                )
                self.assertEqual(
                    json.loads((fit / "config.json").read_text())["evaluation_seed"],
                    replica["evaluation_seed"],
                )
                old_trace = read_jsonl(source / "lastblock_tb/training.jsonl")
                new_trace = read_jsonl(fit / "lastblock_tb/training.jsonl")
                self.assertEqual(len(old_trace), 4)
                self.assertEqual(len(new_trace), 8)
                # Paired seeds reproduce the shared prefix until the old schedule
                # decays. The new midpoint is deliberately not the old checkpoint.
                for old, new in zip(old_trace[:3], new_trace[:3]):
                    self.assertEqual(old["example_ids"], new["example_ids"])
                    self.assertEqual(old["rates"], new["rates"])
                    self.assertAlmostEqual(old["tb_loss"], new["tb_loss"], places=6)
                self.assertEqual(new_trace[3]["rates"], new_trace[0]["rates"])
                self.assertEqual(new_trace[6]["rates"], old_trace[3]["rates"])
                self.assertNotEqual(new_trace[3]["rates"], old_trace[3]["rates"])
                initial = torch.load(
                    fit / "initial.pt", weights_only=True, map_location="cpu"
                )
                unchanged = torch.load(
                    fit / "unchanged/checkpoint.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                for key, value in initial["head"].items():
                    torch.testing.assert_close(
                        value, unchanged["head"][key], rtol=0, atol=0
                    )
                final = torch.load(
                    fit / "lastblock_tb/checkpoint.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                self.assertEqual(final["step"], 8)
            for path, digest in old_files.items():
                self.assertEqual(file_hash(path), digest)
            for key, value in reference.items():
                torch.testing.assert_close(
                    backbone.model.state_dict()[key], value, rtol=0, atol=0
                )
            self.assertTrue(
                all(parameter.grad is None for parameter in backbone.model.parameters())
            )
            with self.assertRaises(FileExistsError):
                run_budget(settings, output, "cpu", backbone, fixture_predictor)
            # Complete generation is self-contained for scoring. Neither the
            # old run folders nor a fitting/model-loading function is needed.
            (root / "history").rename(root / "hidden-history")
            with patch(
                "stability_gfn.idr.budget_experiment.run_lastblock",
                side_effect=AssertionError("Recovery must not fit"),
            ):
                reevaluate_budget(output, root / "recovered", fixture_predictor)
            for filename in (
                "metrics.json",
                "decision.json",
                "report.md",
                "sequence_metrics.json",
            ):
                self.assertEqual(
                    (output / filename).read_text(),
                    (root / "recovered" / filename).read_text(),
                )
            with (output / "replicate-1/control400/samples.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_budget(output, root / "corrupt", fixture_predictor)
            self.assertFalse(torch.cuda.is_initialized())

    def test_final_control_scoring_failure_can_recover_without_fitting(self):
        backbone = FrozenProGen.tiny(11)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real ProGen"),
            ),
        ):
            root = Path(directory)
            (root / "history").mkdir()
            settings = budget_fixture(root / "history", backbone)
            output = root / "budget"
            with patch(
                "stability_gfn.idr.budget_experiment.evaluate_budget",
                side_effect=RuntimeError("Injected final scoring interruption"),
            ):
                with self.assertRaisesRegex(RuntimeError, "scoring interruption"):
                    run_budget(settings, output, "cpu", backbone, fixture_predictor)
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "failed"
            )
            self.assertTrue((output / "generation.json").is_file())
            (root / "history").rename(root / "hidden-history")
            report = reevaluate_budget(output, root / "recovered", fixture_predictor)
            self.assertTrue(report.is_file())
            self.assertFalse(torch.cuda.is_initialized())
