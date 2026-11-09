"""Synthetic checks for fixed-noise semantics, crossed fitting and exact evaluation."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.environment import PrefixTree, SequenceContext
from stability_gfn.guidance import CachedTask
from stability_gfn.pipeline import ROOT, save_head
from stability_gfn.reward_noise import (
    check_noise_inputs,
    evaluate_noise_study,
    fixed_score_noise,
    run_noise_study,
)


class RewardNoiseTests(unittest.TestCase):
    def fixture(self, directory):
        torch.set_num_threads(1)
        clean = directory / "clean"
        clean.mkdir()
        original = json.loads((ROOT / "configs/biological-pilot.json").read_text())
        original.update(smoke=True, seeds=[0, 1])
        original["training"].update(hidden_size=8, steps=4, batch_size=4, log_every=2)
        write_json(clean / "config.json", original)
        write_json(clean / "status.json", {"status": "smoke_complete"})
        write_json(
            clean / "reward.json",
            {"family": "ridge", "scale": 1.2, "candidate_count": 3},
        )
        context = SequenceContext("AAA", 2, "AG")
        tree = PrefixTree(context, ["AGG", "GAG", "GGA"])
        generator = torch.Generator().manual_seed(6)
        features = torch.randn(len(tree.prefixes), 4, generator=generator)
        logits = torch.randn(len(tree.prefixes), 2, generator=generator)
        scores = torch.tensor([-0.5, 0.2, 0.8])
        write_json(
            clean / "evaluation_labels.json",
            [
                {"sequence": s, "stability": 3.0 + i}
                for i, s in enumerate(tree.sequences)
            ],
        )
        payload = {
            "context": asdict(context),
            "sequences": tree.sequences,
            "features": features,
            "logits": logits,
            "scores": scores,
            "predictions": 4 + 1.2 * scores,
            "reference_label": 4.0,
            "evaluation_labels_sha256": file_hash(clean / "evaluation_labels.json"),
        }
        torch.save(payload, clean / "task.pt")
        task = CachedTask(tree, features, logits, scores)
        for seed in [0, 1]:
            (clean / f"seed-{seed}").mkdir()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                save_head(clean / f"seed-{seed}/gfn.pt", task.head(8), 8, 0.0)
        names = [
            "task.pt",
            "config.json",
            "reward.json",
            "status.json",
            "evaluation_labels.json",
            "seed-0/gfn.pt",
            "seed-1/gfn.pt",
        ]
        return {
            "clean_run": str(clean),
            "noise_sigmas": [0.25, 0.5],
            "noise_seeds": [100, 101],
            "input_sha256": {name: file_hash(clean / name) for name in names},
        }

    def test_fixed_noise_zero_pairing_and_rng_isolation(self):
        scores = torch.tensor([-0.8, 0.3, 0.9])
        state = torch.random.get_rng_state().clone()
        zero, standard = fixed_score_noise(scores, 0, 100)
        torch.testing.assert_close(zero, scores, atol=0, rtol=0)
        low, same = fixed_score_noise(scores, 0.25, 100)
        high, _ = fixed_score_noise(scores, 0.5, 100)
        torch.testing.assert_close(standard, same, atol=0, rtol=0)
        torch.testing.assert_close(high - scores, 2 * (low - scores))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertFalse(torch.equal(same, fixed_score_noise(scores, 0.25, 101)[1]))
        for sigma in [-0.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                fixed_score_noise(scores, sigma, 100)

    def test_crossed_panel_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            run_noise_study(config, root / "run", "cpu", smoke=True)
            original = json.loads((root / "run/evaluation/results.json").read_text())
            self.assertEqual(len(original["conditions"]), 8)
            self.assertEqual(len(original["baseline"]["clean_gfn"]), 2)
            clean_value = original["baseline"]["clean_target"]["expected_measured_dg"]
            for r in original["conditions"]:
                self.assertAlmostEqual(
                    r["policy"]["expected_measured_dg"] - clean_value,
                    r["target_measured_shift_from_clean"]
                    + r["policy_measured_gap_from_noisy_target"],
                )
                self.assertIn("expected_clean_ridge_predicted_dg", r["policy"])
            replay = evaluate_noise_study(root / "run", root / "replay")
            self.assertEqual(original, replay)
            for name, expected in config["input_sha256"].items():
                self.assertEqual(file_hash(Path(config["clean_run"]) / name), expected)
            with self.assertRaises(FileExistsError):
                run_noise_study(config, root / "run", "cpu", smoke=True)
            self.assertFalse(torch.cuda.is_initialized())

    def test_input_checks_and_real_cpu_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            _, readiness = check_noise_inputs(config)
            self.assertEqual(readiness["new_fits"], 8)
            with self.assertRaises(ValueError):
                run_noise_study(config, root / "bad", "cpu")
            self.assertFalse((root / "bad").exists())
            (root / "clean/reward.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Clean input changed"):
                check_noise_inputs(config)
