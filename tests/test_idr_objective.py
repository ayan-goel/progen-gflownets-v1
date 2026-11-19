"""Tiny CPU integration for matched objectives, protected inputs and recovery."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_idr_objective_data import objective_fixture
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.lastblock_evaluation import read_jsonl
from stability_gfn.idr.objective_experiment import (
    _save_policy,
    reevaluate_objective,
    run_objective,
)


class ObjectiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_failed_checkpoint_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            path.write_bytes(b"previous complete checkpoint")

            def fail_after_partial_write(checkpoint, destination):
                destination.write_bytes(b"partial replacement")
                raise OSError("Synthetic disk write failure")

            with patch(
                "stability_gfn.idr.objective_experiment.torch.save",
                fail_after_partial_write,
            ):
                with self.assertRaises(OSError):
                    _save_policy(
                        path,
                        "supervised",
                        SimpleNamespace(last_block=torch.nn.Linear(2, 2)),
                        torch.nn.Linear(2, 2),
                        None,
                        [{"id": "prompt"}],
                        100,
                    )
            self.assertEqual(path.read_bytes(), b"previous complete checkpoint")

    def test_shared_training_final_generation_and_source_free_recovery(self):
        backbone = FrozenProGen.tiny(11)
        original = {
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
            settings = objective_fixture(history, backbone)
            historical_hashes = {
                path: file_hash(path) for path in history.rglob("*") if path.is_file()
            }
            output = root / "objective"
            run_objective(settings, output, "cpu", backbone, fixture_predictor)
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "complete"
            )
            records = read_jsonl(output / "samples.jsonl")
            self.assertEqual(len(records), 24)
            self.assertEqual(len({row["id"] for row in records}), 24)
            self.assertEqual(
                {row["arm"] for row in records}, {"gfn", "supervised", "prior"}
            )
            for plan in settings["replicates"]:
                folder = output / plan["name"]
                gfn = read_jsonl(folder / "gfn/training.jsonl")
                supervised = read_jsonl(folder / "supervised/training.jsonl")
                self.assertEqual(len(gfn), 4)
                self.assertEqual(
                    [row["example_ids"] for row in gfn],
                    [row["example_ids"] for row in supervised],
                )
                self.assertEqual(gfn[0]["rates"]["base"], gfn[2]["rates"]["base"])
                self.assertAlmostEqual(
                    gfn[3]["rates"]["base"], gfn[0]["rates"]["base"] * 0.1
                )
                self.assertEqual(len(read_jsonl(folder / "training_rows.jsonl")), 12)
                for arm in ("gfn", "supervised"):
                    checkpoint = torch.load(
                        folder / arm / "checkpoint.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.assertEqual(checkpoint["step"], 4)
                    self.assertEqual(checkpoint["arm"], arm)
                    self.assertEqual(checkpoint["log_z"] is None, arm == "supervised")
                    self.assertFalse((folder / arm / "latest.pt").exists())
                    self.assertFalse(list((folder / arm).glob("step-*.pt")))
                    self.assertTrue(
                        any(
                            not torch.equal(value, original[f"transformer.h.1.{key}"])
                            for key, value in checkpoint["last_block"].items()
                        )
                    )
            for key, value in backbone.model.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertTrue(
                all(parameter.grad is None for parameter in backbone.model.parameters())
            )
            for path, digest in historical_hashes.items():
                self.assertEqual(file_hash(path), digest)
            with self.assertRaises(FileExistsError):
                run_objective(settings, output, "cpu", backbone, fixture_predictor)
            # The saved generation/evaluator needs no old source run directories.
            history.rename(root / "hidden-history")
            recovery = root / "recovery"
            reevaluate_objective(output, recovery, fixture_predictor)
            for name in (
                "metrics.json",
                "decision.json",
                "training_summary.json",
                "sequence_metrics.json",
            ):
                self.assertEqual(
                    (output / name).read_bytes(), (recovery / name).read_bytes()
                )
            with (output / "replicate-1/samples.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_objective(output, root / "corrupt", fixture_predictor)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
