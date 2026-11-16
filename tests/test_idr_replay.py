"""Replay split isolation, causal caching and tiny CPU end-to-end execution."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_idr_charge import configuration

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.acceptance_reward import (
    AcceptanceRewardConfig,
    joint_acceptance_reward,
)
from stability_gfn.idr.charge import path_log_probabilities, rollout
from stability_gfn.idr.context_experiment import make_context_head
from stability_gfn.idr.replay_data import (
    bank_indices,
    cache_sequences,
    check_replay_inputs,
    make_bank,
)
from stability_gfn.idr.replay_evaluation import summarize_fit
from stability_gfn.idr.replay_experiment import reevaluate_replay, run_replay


def fixture_predictor(sequences, version):
    """Known deterministic scores, never execute metapredict."""
    return [
        np.full(len(sequence), 0.9 if version == "V3" else 0.85)
        for sequence in sequences
    ]


def fixture_source(root, backbone):
    """Explicitly marked synthetic source, with real tiny-backbone probabilities."""
    config = configuration()
    config.update(steps=4, beta=1, reward_version="joint_acceptance_v1")
    config["reward"] = dict(config["reward"], block_targets=[0, 0], fcr_target=0)
    reward = replace(
        AcceptanceRewardConfig(), tail_length=4, block_targets=[0, 0], fcr_target=0
    )
    prompts = [
        {"id": "a", "role": "prompt", "sequence": "GSKQAAAA"},
        {"id": "b", "role": "prompt", "sequence": "AQPSGGGG"},
    ]
    data = root / "data"
    data.mkdir()
    write_json(data / "references.json", prompts)
    write_json(
        data / "manifest.json",
        {
            "count": 2,
            "prompt_count": 2,
            "references_sha256": file_hash(data / "references.json"),
        },
    )
    config.update(data=str(data), manifest_sha256=file_hash(data / "manifest.json"))
    source = root / "source"
    source.mkdir()
    (source / "seed-0").mkdir()
    write_json(source / "pilot_config.json", config)
    write_json(source / "config.json", {"reward": asdict(reward)})
    write_json(source / "references.json", prompts)
    write_json(
        source / "provenance.json", {"packages": {}, "files": {}, "fixture": True}
    )
    write_json(source / "status.json", {"status": "complete", "fixture": True})
    (source / "samples.jsonl").write_text("")
    rows = []
    for step, tail in enumerate(("AAAA", "GGGG", "KKKK", "DDDD"), 1):
        sequences = [p["sequence"][:4] + tail for p in prompts]
        _, logits = backbone.forward([s[:-1] for s in sequences])
        base = logits[:, 4:, backbone.residue_ids("ACDEFGHIKLMNPQRSTVWY")].log_softmax(
            -1
        )
        action = "ACDEFGHIKLMNPQRSTVWY".index(tail[0])
        log_prior = base[:, :, action].sum(1)
        values = joint_acceptance_reward(
            [tail, tail], fixture_predictor([tail, tail], "V3"), reward
        )
        for index, prompt in enumerate(prompts):
            rows.append(
                {
                    "step": step,
                    "seed": 0,
                    "prompt_id": prompt["id"],
                    "tail": tail,
                    "isolated_tail_v3_scores": [0.9] * 4,
                    "log_prior": float(log_prior[index]),
                    "behavior_log_policy": float(log_prior[index]),
                    "joint_pass": bool(values["joint_pass"][index]),
                    "log_factor": float(values["log_factor"][index]),
                }
            )
    (source / "seed-0/reward_inputs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    captured = [p for p in source.rglob("*") if p.is_file() and p.name != "status.json"]
    write_json(
        source / "generation.json",
        {
            "complete": True,
            "count": 0,
            "files": {str(p.relative_to(source)): file_hash(p) for p in captured},
        },
    )
    settings = json.loads(
        (Path(__file__).parents[1] / "configs/idr-replay.json").read_text()
    )
    settings.update(
        source_run=str(source),
        source_generation_sha256=file_hash(source / "generation.json"),
        training_steps=3,
        evaluation_samples_per_prompt=2,
        cache_batch_size=2,
        near_misses_per_prompt=2,
        holdout_failures_per_prompt=1,
    )
    return settings, config, prompts, rows


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_teacher_forcing_matches_incremental_rollout_and_current_policy(self):
        backbone = FrozenProGen.tiny(11)
        config = configuration()
        prompts = [
            {"id": "a", "sequence": "GSKQEKAG"},
            {"id": "b", "sequence": "AQPSEKGS"},
        ]
        head = make_context_head(backbone, 2, config, 0, "direct_context")
        sampled = rollout(
            backbone,
            head,
            prompts,
            [0, 1],
            config,
            torch.Generator().manual_seed(12),
            retain=True,
        )
        rows = [
            {
                "sequence": sequence,
                "tail": sequence[4:],
                "prompt_index": index,
                "log_prior": float(sampled["log_prior"][index]),
            }
            for index, sequence in enumerate(sampled["sequences"])
        ]
        cache = cache_sequences(backbone, rows, prompts, config, 2)
        for key in cache:
            torch.testing.assert_close(cache[key], sampled[key], rtol=2e-5, atol=2e-5)
            self.assertFalse(cache[key].requires_grad)
        with torch.no_grad():
            head.output.bias[0] += 1.0
        actual = path_log_probabilities(head, cache)
        expected = path_log_probabilities(head, sampled)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, sampled["log_policy"]))
        actual.sum().backward()
        self.assertGreater(head.output.bias.grad.norm(), 0)
        self.assertTrue(all(p.grad is None for p in backbone.model.parameters()))

    def test_split_groups_duplicate_tails_and_sampling_avoids_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, config, prompts, rows = fixture_source(
                Path(directory), FrozenProGen.tiny(11)
            )
            settings["prefix_length"] = config["prefix_length"]
            bank, summary = make_bank(rows + rows, prompts, settings)
            self.assertEqual(summary["source_unique_prompt_tails"], len(rows))
            self.assertEqual(len({r["id"] for r in bank}), len(bank))
            train_tails = {r["tail"] for r in bank if r["split"] == "train"}
            self.assertFalse(
                train_tails.intersection(
                    r["tail"] for r in bank if r["split"] == "holdout"
                )
            )
            for step in range(1, 5):
                selected = bank_indices(
                    bank, 2, torch.Generator().manual_seed(step), step
                )
                self.assertEqual(len(selected), 2)
                self.assertEqual(sum(bank[i]["joint_pass"] for i in selected), 1)
                self.assertTrue(all(bank[i]["split"] == "train" for i in selected))
                supervised = bank_indices(
                    bank, 2, torch.Generator().manual_seed(step), step, True
                )
                self.assertEqual(len(supervised), 4)
                self.assertTrue(
                    all(
                        bank[i]["joint_pass"] and bank[i]["split"] == "train"
                        for i in supervised
                    )
                )
            with self.assertRaises(ValueError):
                make_bank([r for r in rows if not r["joint_pass"]], prompts, settings)

    def test_tiny_end_to_end_budget_integrity_and_standalone_recovery(self):
        backbone = FrozenProGen.tiny(11)
        before = {
            name: value.clone() for name, value in backbone.model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, config, prompts, _ = fixture_source(root, backbone)
            source_hashes = {
                p: file_hash(p) for p in (root / "source").rglob("*") if p.is_file()
            }
            checked = check_replay_inputs(settings, False)
            self.assertGreater(len(checked[-2]), 0)
            with self.assertRaises(ValueError):
                check_replay_inputs(settings, True)
            bad = copy.deepcopy(settings)
            bad["source_generation_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                check_replay_inputs(bad, False)
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No pretrained execution"),
            ):
                run_replay(settings, root / "run", "cpu", backbone, fixture_predictor)
                with self.assertRaises(FileExistsError):
                    run_replay(
                        settings, root / "run", "cpu", backbone, fixture_predictor
                    )
                for path, digest in source_hashes.items():
                    self.assertEqual(file_hash(path), digest)
                (root / "source").rename(root / "hidden-source")
                (root / "data").rename(root / "hidden-data")
                reevaluate_replay(root / "run", root / "rescored", fixture_predictor)
            for name in (
                "metrics.json",
                "sequence_metrics.json",
                "decision.json",
                "fit_summary.json",
                "training_summary.json",
            ):
                self.assertEqual(
                    json.loads((root / "run" / name).read_text()),
                    json.loads((root / "rescored" / name).read_text()),
                )
            self.assertEqual(
                (root / "run/report.md").read_text(),
                (root / "rescored/report.md").read_text(),
            )
            summary = json.loads((root / "run/training_summary.json").read_text())
            self.assertEqual(
                [summary[arm]["trajectory_examples"] for arm in summary], [12, 12, 12]
            )
            self.assertEqual(summary["on_policy_tb"]["fresh_trajectories"], 12)
            self.assertEqual(summary["replay_tb"]["fresh_trajectories"], 6)
            self.assertEqual(summary["replay_tb"]["stored_success_draws"], 3)
            self.assertEqual(summary["supervised"]["stored_success_draws"], 12)
            self.assertEqual(summary["supervised"]["fresh_trajectories"], 0)
            fits = []
            for arm in ("on_policy_tb", "replay_tb", "supervised"):
                checkpoint = torch.load(
                    root / f"run/{arm}/step-3.pt", map_location="cpu", weights_only=True
                )
                self.assertGreater(checkpoint["head"]["task_weight"].norm(), 0)
                self.assertEqual(checkpoint["log_z"] is None, arm == "supervised")
                logs = [
                    json.loads(line)
                    for line in (root / f"run/{arm}/training.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertAlmostEqual(logs[0]["rates"]["task_branch"], 0.001)
                self.assertAlmostEqual(logs[-1]["rates"]["task_branch"], 0.0001)
                if arm == "supervised":
                    self.assertNotIn("normalizer", logs[0]["rates"])
                fits.append(json.loads((root / f"run/{arm}/bank_fit.json").read_text()))
            for left, right in zip(fits[0], fits[1]):
                if left["stage"] == "initial":
                    self.assertEqual(left["log_policy"], right["log_policy"])
            scored = json.loads((root / "run/sequence_metrics.json").read_text())
            self.assertEqual(len(scored), 16)
            for row in scored:
                if row["arm"] in ("on_policy_tb", "replay_tb"):
                    self.assertEqual(
                        row["own_reward_tb_residual"],
                        row["log_z"]
                        + row["log_policy"]
                        - row["log_prior"]
                        - row["acceptance"]["log_factor"],
                    )
                else:
                    self.assertNotIn("own_reward_tb_residual", row)
            (root / "run/bank.json").write_text("[]\n")
            with self.assertRaises(ValueError):
                reevaluate_replay(root / "run", root / "bad", fixture_predictor)
        self.assertTrue(
            all(
                torch.equal(before[name], value)
                for name, value in backbone.model.state_dict().items()
            )
        )
        self.assertTrue(all(p.grad is None for p in backbone.model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_fit_holdout_exposure_is_removed_from_both_stages(self):
        rows = [
            {
                "id": "seen",
                "arm": "replay_tb",
                "stage": stage,
                "split": "holdout",
                "joint_pass": True,
                "fresh_training_overlap": stage == "final",
                "token_nll": 1,
                "tb_residual": 1,
            }
            for stage in ("initial", "final")
        ]
        result = summarize_fit(rows)
        for stage in ("initial", "final"):
            self.assertEqual(result[f"replay_tb/{stage}/bank_holdout/pass"]["count"], 0)
            self.assertIsNone(result[f"replay_tb/{stage}/bank_holdout/pass"]["tb_rmse"])


if __name__ == "__main__":
    unittest.main()
