"""CPU-only continuation initialization, control comparison and snapshot recovery."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_replay import fixture_predictor, fixture_source

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.replay_experiment import run_replay
from stability_gfn.idr.warmstart_data import (
    check_warmstart_inputs,
    initialize_normalizer,
)
from stability_gfn.idr.warmstart_evaluation import (
    progression_decision,
    summarize_fit,
    summarize_target_shape,
)
from stability_gfn.idr.warmstart_experiment import reevaluate_warmstart, run_warmstart


class WarmstartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_normalizer_centers_training_strata_without_holdout(self):
        bank = [
            {"id": "positive", "log_factor": 0},
            {"id": "negative", "log_factor": -4},
            {"id": "heldout", "log_factor": 1000},
        ]
        rows = [
            {
                "id": "positive",
                "prompt_id": "a",
                "split": "train",
                "joint_pass": True,
                "log_prior": -10,
                "log_policy": -9,
            },
            {
                "id": "negative",
                "prompt_id": "a",
                "split": "train",
                "joint_pass": False,
                "log_prior": -10,
                "log_policy": -12,
            },
            {
                "id": "heldout",
                "prompt_id": "a",
                "split": "holdout",
                "joint_pass": True,
                "log_prior": 0,
                "log_policy": 999,
            },
        ]
        offset, _ = initialize_normalizer(rows, bank, ["a"])
        self.assertEqual(offset.tolist(), [-1.5])
        # Residuals: z + (q - p0) - log factor = -0.5 and +0.5.
        self.assertAlmostEqual((float(offset[0]) + 1 + float(offset[0]) + 2) / 2, 0)
        changed = copy.deepcopy(rows)
        changed[-1]["log_policy"] = -999
        torch.testing.assert_close(
            initialize_normalizer(changed, bank, ["a"])[0], offset
        )

    def test_progression_requires_both_controls_and_noncopy_gain(self):
        def metrics(success, unique=1, noncopy=None):
            return {
                "joint_success_rate": success,
                "unique_tail_fraction": unique,
                "joint_success_without_exact_training_copy": success
                if noncopy is None
                else noncopy,
            }

        arms = {
            "unchanged": metrics(0.08),
            "continued_supervised": metrics(0.15),
            "warm_tb": metrics(0.17),
        }
        self.assertFalse(progression_decision(arms)["passes_practical_screen"])
        arms["warm_tb"] = metrics(0.23, noncopy=0.16)
        self.assertFalse(progression_decision(arms)["passes_practical_screen"])
        arms["warm_tb"] = metrics(0.23)
        self.assertTrue(progression_decision(arms)["passes_practical_screen"])
        arms["warm_tb"] = metrics(0.23, unique=0.9)
        self.assertFalse(progression_decision(arms)["passes_practical_screen"])

    def test_relative_target_diagnostic_ignores_offset_and_common_exposure(self):
        bank = [{"id": str(i), "log_factor": -i, "split": "holdout"} for i in range(3)]
        fits = []
        for arm in ("unchanged", "continued_supervised", "warm_tb"):
            for stage in ("initial", "final"):
                for i in range(3):
                    fits.append(
                        {
                            "id": str(i),
                            "arm": arm,
                            "stage": stage,
                            "split": "holdout",
                            "prompt_id": "a",
                            "log_prior": -10,
                            "log_policy": -10 - i + (3 if arm == "warm_tb" else 0),
                            "fresh_training_overlap": arm == "warm_tb" and i == 2,
                            "joint_pass": False,
                            "token_nll": 2,
                            "tb_residual": 0,
                        }
                    )
        result = summarize_target_shape(fits, bank)
        summary = summarize_fit(fits)
        for arm in ("unchanged", "continued_supervised", "warm_tb"):
            group = result[f"{arm}/final/holdout"]
            self.assertEqual(group["mean_prompt_centered_rmse"], 0)
            self.assertEqual(group["by_prompt"]["a"]["count"], 2)
            self.assertEqual(summary[f"{arm}/initial/bank_holdout/fail"]["count"], 2)

    def test_cpu_continuations_preserve_start_backbone_history_and_recovery(self):
        backbone = FrozenProGen.tiny(11)
        frozen = {k: v.clone() for k, v in backbone.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_settings, _, _, _ = fixture_source(root, backbone)
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No pretrained execution"),
            ):
                run_replay(
                    replay_settings, root / "replay", "cpu", backbone, fixture_predictor
                )
                settings = {
                    "source_run": str(root / "replay"),
                    "source_generation_sha256": file_hash(
                        root / "replay/generation.json"
                    ),
                    "training_steps": 3,
                    "evaluation_samples_per_prompt": 2,
                    "training_seed": 92000,
                    "evaluation_seed": 93000,
                    "bank_sampling_seed": 95000,
                }
                original = {
                    p: file_hash(p) for p in (root / "replay").rglob("*") if p.is_file()
                }
                check_warmstart_inputs(settings, False)
                with self.assertRaises(ValueError):
                    check_warmstart_inputs(settings, True)
                bad = dict(settings, source_generation_sha256="bad")
                with self.assertRaises(ValueError):
                    check_warmstart_inputs(bad, False)
                run_warmstart(
                    settings, root / "run", "cpu", backbone, fixture_predictor
                )
                with self.assertRaises(FileExistsError):
                    run_warmstart(
                        settings, root / "run", "cpu", backbone, fixture_predictor
                    )
                for path, digest in original.items():
                    self.assertEqual(file_hash(path), digest)
                initial = torch.load(
                    root / "run/initial.pt", map_location="cpu", weights_only=True
                )
                self.assertEqual(
                    file_hash(root / "run/initial.pt"),
                    file_hash(root / "run/unchanged/checkpoint.pt"),
                )
                fits = {
                    arm: json.loads((root / f"run/{arm}/bank_fit.json").read_text())
                    for arm in ("unchanged", "continued_supervised", "warm_tb")
                }
                start = [
                    r["log_policy"]
                    for r in fits["unchanged"]
                    if r["stage"] == "initial"
                ]
                for arm in ("continued_supervised", "warm_tb"):
                    self.assertEqual(
                        start,
                        [r["log_policy"] for r in fits[arm] if r["stage"] == "initial"],
                    )
                    checkpoint = torch.load(
                        root / f"run/{arm}/step-3.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.assertTrue(
                        any(
                            not torch.equal(initial["head"][k], v)
                            for k, v in checkpoint["head"].items()
                        )
                    )
                    self.assertEqual(
                        checkpoint["log_z"] is None, arm == "continued_supervised"
                    )
                    self.assertTrue(
                        all(
                            int(state["step"]) == 3
                            for state in checkpoint["optimizer"]["state"].values()
                        )
                    )
                    logs = [
                        json.loads(line)
                        for line in (root / f"run/{arm}/training.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    self.assertAlmostEqual(logs[0]["rates"]["task_branch"], 0.001)
                    self.assertAlmostEqual(logs[-1]["rates"]["task_branch"], 0.0001)
                summary = json.loads((root / "run/training_summary.json").read_text())
                self.assertEqual(summary["warm_tb"]["trajectory_examples"], 12)
                self.assertEqual(
                    summary["continued_supervised"]["trajectory_examples"], 12
                )
                self.assertEqual(summary["warm_tb"]["fresh_trajectories"], 6)
                self.assertEqual(summary["warm_tb"]["stored_success_draws"], 3)
                self.assertEqual(
                    summary["continued_supervised"]["stored_success_draws"], 12
                )
                self.assertEqual(
                    summary["continued_supervised"]["fresh_trajectories"], 0
                )
                # Source and input references can disappear: scoring recovery is self-contained.
                (root / "replay").rename(root / "hidden-replay")
                (root / "source").rename(root / "hidden-source")
                (root / "data").rename(root / "hidden-data")
                reevaluate_warmstart(root / "run", root / "rescored", fixture_predictor)
            for name in (
                "metrics.json",
                "sequence_metrics.json",
                "decision.json",
                "fit_summary.json",
                "training_summary.json",
                "target_shape.json",
                "report.md",
            ):
                self.assertEqual(
                    (root / "run" / name).read_text(),
                    (root / "rescored" / name).read_text(),
                )
            scored = json.loads((root / "run/sequence_metrics.json").read_text())
            self.assertEqual(len(scored), 12)
            for row in scored:
                if row["arm"] == "warm_tb":
                    self.assertAlmostEqual(
                        row["own_reward_tb_residual"],
                        row["log_z"]
                        + row["log_policy"]
                        - row["log_prior"]
                        - row["acceptance"]["log_factor"],
                    )
                else:
                    self.assertNotIn("own_reward_tb_residual", row)
            (root / "run/initial_training.jsonl").write_text("")
            with self.assertRaises(ValueError):
                reevaluate_warmstart(root / "run", root / "bad", fixture_predictor)
        self.assertTrue(
            all(
                torch.equal(frozen[k], v)
                for k, v in backbone.model.state_dict().items()
            )
        )
        self.assertTrue(all(p.grad is None for p in backbone.model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
