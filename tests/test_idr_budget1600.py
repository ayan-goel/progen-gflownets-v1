"""Tiny CPU integration checks for the bounded 1,600-versus-800 comparison."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_budget1600_data import budget1600_fixture
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.budget1600_experiment import (
    reevaluate_budget1600,
    run_budget1600,
)
from stability_gfn.idr.lastblock_evaluation import read_jsonl


class Budget1600IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_preserved_controls_stretched_schedule_and_source_free_recovery(self):
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
            history = root / "history"
            history.mkdir()
            settings = budget1600_fixture(history, backbone)
            source = Path(settings["budget_run"])
            old_files = {
                path: file_hash(path) for path in history.rglob("*") if path.is_file()
            }
            output = root / "budget1600"
            run_budget1600(settings, output, "cpu", backbone, fixture_predictor)
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
                    "lastblock_800",
                    "lastblock_tb",
                    "frozen_tb",
                    "unchanged",
                    "continued_supervised",
                },
            )
            for replica in settings["replicates"]:
                name = replica["name"]
                old_fit, fit = source / name / "fit", output / name / "fit"
                control = output / name / "control800"
                self.assertEqual(
                    file_hash(control / "checkpoint.pt"),
                    file_hash(old_fit / "lastblock_tb/checkpoint.pt"),
                )
                self.assertEqual(
                    file_hash(fit / "initial.pt"), file_hash(old_fit / "initial.pt")
                )
                self.assertEqual(
                    file_hash(output / name / "history400_training_rows.jsonl"),
                    file_hash(source / name / "control400/training_rows.jsonl"),
                )
                self.assertEqual(
                    json.loads((fit / "config.json").read_text())["evaluation_seed"],
                    replica["evaluation_seed"],
                )
                old_trace = read_jsonl(old_fit / "lastblock_tb/training.jsonl")
                new_trace = read_jsonl(fit / "lastblock_tb/training.jsonl")
                self.assertEqual(len(old_trace), 8)
                self.assertEqual(len(new_trace), 16)
                for old, new in zip(old_trace[:6], new_trace[:6]):
                    self.assertEqual(old["example_ids"], new["example_ids"])
                    self.assertEqual(old["rates"], new["rates"])
                    self.assertAlmostEqual(old["tb_loss"], new["tb_loss"], places=6)
                self.assertEqual(new_trace[11]["rates"], new_trace[0]["rates"])
                self.assertEqual(new_trace[12]["rates"], old_trace[6]["rates"])
                self.assertNotEqual(new_trace[6]["rates"], old_trace[6]["rates"])
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
                        value, unchanged["head"][key], atol=0, rtol=0
                    )
                final = torch.load(
                    fit / "lastblock_tb/checkpoint.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                self.assertEqual(final["step"], 16)
            for path, digest in old_files.items():
                self.assertEqual(file_hash(path), digest)
            for key, value in reference.items():
                torch.testing.assert_close(
                    backbone.model.state_dict()[key], value, atol=0, rtol=0
                )
            self.assertTrue(
                all(parameter.grad is None for parameter in backbone.model.parameters())
            )
            with self.assertRaises(FileExistsError):
                run_budget1600(settings, output, "cpu", backbone, fixture_predictor)
            history.rename(root / "hidden-history")
            with patch(
                "stability_gfn.idr.budget1600_experiment.run_lastblock",
                side_effect=AssertionError("Recovery must not fit"),
            ):
                reevaluate_budget1600(output, root / "recovered", fixture_predictor)
            for filename in (
                "metrics.json",
                "decision.json",
                "report.md",
                "sequence_metrics.json",
                "copy_definition.json",
            ):
                self.assertEqual(
                    (output / filename).read_text(),
                    (root / "recovered" / filename).read_text(),
                )
            with (output / "replicate-1/history400_training_rows.jsonl").open(
                "a"
            ) as handle:
                handle.write("\n")
            with self.assertRaises(ValueError):
                reevaluate_budget1600(output, root / "corrupted", fixture_predictor)
            self.assertFalse(torch.cuda.is_initialized())

    def test_final_scoring_failure_keeps_generation_recoverable(self):
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
            history = root / "history"
            history.mkdir()
            settings = budget1600_fixture(history, backbone)
            output = root / "interrupted"
            with (
                patch(
                    "stability_gfn.idr.budget1600_experiment.evaluate_budget1600",
                    side_effect=RuntimeError("fixture scoring interruption"),
                ),
                self.assertRaisesRegex(RuntimeError, "scoring interruption"),
            ):
                run_budget1600(settings, output, "cpu", backbone, fixture_predictor)
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "failed"
            )
            self.assertTrue(
                json.loads((output / "generation.json").read_text())["complete"]
            )
            sealed = {
                path: file_hash(path) for path in output.rglob("*") if path.is_file()
            }
            history.rename(root / "hidden-history")
            with (
                patch(
                    "stability_gfn.idr.budget1600_experiment.run_lastblock",
                    side_effect=AssertionError("Recovery must not fit"),
                ),
                patch(
                    "stability_gfn.idr.budget1600_experiment._generate_controls",
                    side_effect=AssertionError("Recovery must not generate"),
                ),
            ):
                reevaluate_budget1600(output, root / "recovered", fixture_predictor)
            self.assertEqual(
                json.loads((root / "recovered/status.json").read_text())["status"],
                "complete",
            )
            for path, digest in sealed.items():
                self.assertEqual(file_hash(path), digest)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
