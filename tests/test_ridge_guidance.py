"""Synthetic CPU-only end-to-end checks of saved-ridge reward substitution."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.environment import PrefixTree, SequenceContext
from stability_gfn.evaluation import evaluate_run, load_task
from stability_gfn.guidance import CachedTask
from stability_gfn.pipeline import ROOT
from stability_gfn.ridge_guidance import prepare_ridge_task, run_ridge_guidance
from stability_gfn.scorer import StabilityScorer


class RidgeGuidanceTests(unittest.TestCase):
    def fixture(self, root):
        torch.set_num_threads(1)
        pilot, study = root / "pilot", root / "study"
        pilot.mkdir()
        study.mkdir()
        original = json.loads((ROOT / "configs/biological-pilot.json").read_text())
        original["smoke"] = True
        write_json(pilot / "config.json", original)
        context = SequenceContext("AAA", 2, "AG")
        tree = PrefixTree(context, ["AGG", "GAG", "GGA"])
        generator = torch.Generator().manual_seed(12)
        features = torch.randn(len(tree.prefixes), 4, generator=generator)
        logits = torch.randn(len(tree.prefixes), 2, generator=generator)
        task = CachedTask(tree, features, logits, [-2.0, -1.0, 0.0])
        training = ["AAA", "AAG", "AGA", "GAA"]
        labels = [
            {"sequence": s, "stability": 4.0 + i} for i, s in enumerate(tree.sequences)
        ]
        write_json(pilot / "evaluation_labels.json", labels)
        payload = {
            "context": asdict(context),
            "sequences": tree.sequences,
            "features": features,
            "logits": logits,
            "scores": task.scores,
            "predictions": task.scores,
            "reference_prediction": 0.0,
            "reference_label": 4.0,
            "training_sequences": training,
            "evaluation_labels_sha256": file_hash(pilot / "evaluation_labels.json"),
        }
        torch.save(payload, pilot / "task.pt")
        pooled = {
            "train": torch.randn(4, 4, generator=generator),
            "test": torch.randn(3, 4, generator=generator),
        }
        model = StabilityScorer(4, 0).eval()
        with torch.no_grad():
            model.network.weight.fill_(0.1)
            model.network.bias.fill_(4.0)
            predictions = model(pooled["test"])
            reference = model(pooled["train"][:1]).item()
        torch.save(
            {"state_dict": model.state_dict(), "hidden_size": 0, "feature_size": 4},
            study / "ridge.pt",
        )
        torch.save(
            {
                "features": pooled,
                "rows": {"train": [{"sequence": s} for s in training], "test": labels},
                "reference_index": 0,
                "log_prior": task.base_log_probabilities,
            },
            study / "features.pt",
        )
        write_json(
            study / "predictions.json",
            {"sequences": tree.sequences, "predictions": predictions.tolist()},
        )
        write_json(
            study / "results.json",
            [
                {
                    "family": "ridge",
                    "train_size": 4,
                    "seed": 0,
                    "scale": 1.0,
                    "reference_prediction": reference,
                }
            ],
        )
        inputs = list(pilot.iterdir()) + list(study.iterdir())
        return {
            "pilot_run": str(pilot),
            "predictor_run": str(study),
            "predictor_seed": 0,
            "scorer_file": "ridge.pt",
            "predictions_file": "predictions.json",
            "input_sha256": {str(p): file_hash(p) for p in inputs},
        }

    def test_replacement_training_evaluation_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            payload, _, metadata = prepare_ridge_task(config)
            prior, old = load_task(root / "pilot")
            torch.testing.assert_close(payload["logits"], old["logits"], atol=0, rtol=0)
            torch.testing.assert_close(
                payload["features"], old["features"], atol=0, rtol=0
            )
            self.assertFalse(torch.equal(payload["scores"], old["scores"]))
            torch.testing.assert_close(
                payload["scores"],
                (payload["predictions"] - metadata["reference_prediction"])
                / metadata["scale"],
            )
            run = root / "new"
            run_ridge_guidance(config, run, "cpu", smoke=True)
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"],
                "smoke_complete",
            )
            self.assertFalse((run / "seed-0/supervised.pt").exists())
            history = json.loads((run / "seed-0/gfn_history.json").read_text())
            self.assertEqual(history[-1]["step"], 4)
            self.assertIn("distribution", history[-1])
            comparison = json.loads((run / "exact_comparison.json").read_text())
            self.assertEqual(len(comparison), 3)
            self.assertLess(comparison[1]["total_variation"], 1e-12)
            self.assertEqual(comparison[0]["measured_gain_over_progen"], 0.0)
            replay = root / "replay"
            evaluate_run(run, replay)
            self.assertEqual(
                json.loads((run / "evaluation/summary.json").read_text()),
                json.loads((replay / "summary.json").read_text()),
            )
            for path, expected in config["input_sha256"].items():
                self.assertEqual(file_hash(Path(path)), expected)
            self.assertFalse(torch.cuda.is_initialized())
            with self.assertRaises(FileExistsError):
                run_ridge_guidance(config, run, "cpu", smoke=True)

    def test_changed_input_and_accidental_real_cpu_training_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            with self.assertRaises(ValueError):
                run_ridge_guidance(config, root / "bad", "cpu")
            self.assertFalse((root / "bad").exists())
            (root / "study/predictions.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                prepare_ridge_task(config)
