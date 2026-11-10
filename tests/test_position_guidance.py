"""Synthetic-only tests for fixed grouped rewards, TB training and CPU replay."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_split_study as split_fixture
import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.evaluation import load_task
from stability_gfn.position_guidance import (
    evaluate_position_guidance,
    prepare_position_guidance,
    run_position_guidance,
)
from stability_gfn.split_study import run_split_study
from stability_gfn.training import train_tb


class PositionGuidanceTests(unittest.TestCase):
    def fixture(self, root):
        config = split_fixture.SplitStudyTests().fixture(root)
        config["split_unit"] = "position"
        source = root / "positions"
        run_split_study(config, source, "cpu", smoke=True)
        paths = [
            source / name for name in ["config.json", "status.json", "provenance.json"]
        ]
        for seed in [43, 44, 45]:
            directory = source / f"split-{seed}"
            paths += [
                directory / name
                for name in [
                    "partitions.json",
                    "mutation_positions.json",
                    "prior.pt",
                    "results.json",
                ]
            ]
            paths += list(directory.glob("*-ridge.pt"))
            paths += list(directory.glob("*-ridge-predictions.json"))
            paths += list(directory.glob("*-mlp-predictions.json"))
        return {
            "source_run": str(source),
            "input_sha256": {str(path): file_hash(path) for path in paths},
        }

    def test_crossed_training_replay_and_label_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            run = root / "gfn"
            _, before, readiness = prepare_position_guidance(config)
            self.assertEqual(readiness["new_gfn_fits"], 9)
            self.assertLess(
                max(row["score_rounding_target_tv"] for row in readiness["splits"]),
                1e-6,
            )
            calls = []

            def checked_training(task, settings, seed, progress):
                self.assertFalse(list(run.glob("split-*/evaluation_labels.json")))
                self.assertEqual(
                    set(vars(task)), set(vars(load_task(run / "split-43")[0]))
                )
                self.assertEqual(settings["steps"], 4)
                calls.append(seed)
                return train_tb(task, settings, seed, progress)

            with patch(
                "stability_gfn.position_guidance.train_tb", side_effect=checked_training
            ):
                run_position_guidance(config, run, "cpu", smoke=True)
            self.assertEqual(calls, [0, 1, 0, 1, 0, 1])
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"],
                "smoke_complete",
            )
            results = json.loads((run / "evaluation/results.json").read_text())
            self.assertEqual(len(results), 15)
            self.assertEqual(results, evaluate_position_guidance(run, root / "replay"))
            for split in [43, 44, 45]:
                task, payload = load_task(run / f"split-{split}")
                self.assertNotIn("measurements", payload)
                self.assertNotIn("reference_label", payload)
                torch.testing.assert_close(
                    payload["logits"], before[split]["logits"], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    task.scores, before[split]["scores"], atol=0, rtol=0
                )
                ridge = next(
                    row
                    for row in results
                    if row["split_seed"] == split
                    and row["method"] == "Exact ridge target"
                )
                self.assertLess(ridge["total_variation"], 1e-6)
                self.assertAlmostEqual(
                    ridge["measured_gap_from_ridge_target"], 0, places=10
                )
                for seed in [0, 1]:
                    history = json.loads(
                        (
                            run / f"split-{split}/seed-{seed}/gfn_history.json"
                        ).read_text()
                    )
                    self.assertEqual(history[-1]["step"], 4)
                    self.assertIn("distribution", history[-1])
            for path, expected in config["input_sha256"].items():
                self.assertEqual(file_hash(Path(path)), expected)
            with self.assertRaises(FileExistsError):
                evaluate_position_guidance(run, root / "replay")
            with self.assertRaises(FileExistsError):
                run_position_guidance(config, run, "cpu", smoke=True)
            self.assertFalse(torch.cuda.is_initialized())
            # Post-training labels affect biological metrics, not saved heads or the sampling distribution.
            hashes = json.loads((run / "evaluation_inputs.json").read_text())
            label_path = run / "split-43/evaluation_labels.json"
            labels = json.loads(label_path.read_text())
            for row in labels["rows"]:
                row["stability"] += 10
            write_json(label_path, labels)
            with self.assertRaisesRegex(ValueError, "Evaluation input changed"):
                evaluate_position_guidance(run, root / "changed")
            hashes[str(label_path.relative_to(run))] = file_hash(label_path)
            write_json(run / "evaluation_inputs.json", hashes)
            changed = evaluate_position_guidance(run, root / "changed")
            for old, new in zip(results, changed):
                self.assertEqual(old["total_variation"], new["total_variation"])
                difference = 10 if old["split_seed"] == 43 else 0
                self.assertAlmostEqual(
                    new["expected_measured_dg"] - old["expected_measured_dg"],
                    difference,
                    places=8,
                )

    def test_cpu_guard_missing_hash_and_stale_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Real runs require"):
                run_position_guidance(config, root / "invalid", "cpu")
            self.assertFalse((root / "invalid").exists())
            missing = dict(config, input_sha256=dict(config["input_sha256"]))
            missing["input_sha256"].pop(str(root / "positions/split-43/prior.pt"))
            with self.assertRaisesRegex(ValueError, "Missing per-split"):
                prepare_position_guidance(missing)
            (root / "positions/status.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                prepare_position_guidance(config)

    def test_backbone_mismatch_stops_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            run = root / "invalid"
            with patch(
                "stability_gfn.position_guidance.FrozenProGen.tiny",
                return_value=split_fixture.FrozenProGen.tiny(seed=5),
            ):
                with patch("stability_gfn.position_guidance.train_tb") as fitting:
                    with self.assertRaises(AssertionError):
                        run_position_guidance(config, run, "cpu", smoke=True)
                    fitting.assert_not_called()
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"], "failed"
            )
            self.assertFalse(list(run.glob("split-*/seed-*/gfn.pt")))
