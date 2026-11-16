"""Full-support probabilities, conditioning, accumulated gradients and CPU pipeline."""

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
from stability_gfn.idr.charge import path_log_probabilities
from stability_gfn.idr.context_experiment import DirectContextHead
from stability_gfn.idr.continuous_experiment import run_continuous
from stability_gfn.idr.replay_experiment import run_replay
from stability_gfn.idr.transfer_data import check_transfer_inputs
from stability_gfn.idr.transfer_evaluation import progression_decision
from stability_gfn.idr.transfer_experiment import reevaluate_transfer, run_transfer
from stability_gfn.idr.transfer_head import condition_head
from stability_gfn.idr.transfer_training import batch_gradient, make_proposal
from stability_gfn.idr.warmstart_experiment import run_warmstart


def synthetic_cache():
    torch.manual_seed(15)
    cache = dict(
        features=torch.randn(4, 3, 6),
        context=torch.randn(4, 3, 7),
        logits=torch.randn(4, 3, 20),
        actions=torch.randint(20, (4, 3)),
    )
    cache["features"][:, 0] = cache["features"][0, 0]
    cache["log_prior"] = (
        cache["logits"]
        .log_softmax(-1)
        .gather(-1, cache["actions"][..., None])
        .sum((1, 2))
    )
    head = DirectContextHead(6, 7, 20, 8)
    with torch.no_grad():
        head.output.weight.normal_(0, 0.05)
    bank = [
        dict(
            id=str(i),
            prompt_index=i // 2,
            prompt_id=str(i // 2),
            joint_pass=i % 2 == 0,
            log_factor=-float(i),
        )
        for i in range(4)
    ]
    return head, cache, bank


def transfer_source(root, backbone):
    settings, _, _, _ = fixture_source(root, backbone)
    run_replay(settings, root / "replay", "cpu", backbone, fixture_predictor)
    warm = dict(
        source_run=str(root / "replay"),
        source_generation_sha256=file_hash(root / "replay/generation.json"),
        training_steps=3,
        evaluation_samples_per_prompt=2,
        training_seed=92000,
        evaluation_seed=93000,
        bank_sampling_seed=95000,
    )
    run_warmstart(warm, root / "warm", "cpu", backbone, fixture_predictor)
    continuous = dict(
        warm,
        control_run=str(root / "warm"),
        control_generation_sha256=file_hash(root / "warm/generation.json"),
        evaluation_seed=103000,
        penalty=2.5,
    )
    run_continuous(continuous, root / "continuous", "cpu", backbone, fixture_predictor)
    return dict(
        source_run=str(root / "continuous"),
        source_generation_sha256=file_hash(root / "continuous/generation.json"),
        training_steps=3,
        evaluation_samples_per_prompt=2,
        bank_sampling_seed=98000,
        evaluation_seed=113000,
        penalty=2.5,
        conditioning_maximum_gain=32,
        draws_per_prompt=2,
        snapshot_interval=2,
        microbatch_size=3,
    )


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rank_deficient_conditioning_preserves_function_and_folding(self):
        initial, cache, _ = synthetic_cache()
        head, info = condition_head(initial, cache)
        self.assertLess(info["active_directions"], info["dimensions"])
        self.assertLessEqual(info["actual_maximum_scale"], 32)
        self.assertEqual(
            sum(p.numel() for p in initial.parameters()),
            sum(p.numel() for p in head.parameters()),
        )
        features, context = torch.randn(11, 6), torch.randn(11, 7)
        torch.testing.assert_close(
            initial(features, context), head(features, context), atol=1e-5, rtol=1e-5
        )
        with torch.no_grad():
            head.hidden.weight.add_(0.001)
            head.normalization.weight.mul_(1.01)
        torch.testing.assert_close(
            head(features, context),
            head.fold()(features, context),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_all_residues_compete_at_every_cached_position(self):
        head, cache, _ = synthetic_cache()
        before = path_log_probabilities(head, cache)
        changed = {k: v.clone() for k, v in cache.items()}
        # A residue absent from this trajectory must still compete in its softmax.
        unused = next(i for i in range(20) if i not in cache["actions"][0].tolist())
        changed["logits"][0, :, unused] += 10
        after = path_log_probabilities(head, changed)
        self.assertLess(float(after[0]), float(before[0]) - 1)
        torch.testing.assert_close(after[1:], before[1:])

    def test_microbatch_gradient_is_globally_weighted_and_proposal_is_not_policy(self):
        initial, cache, bank = synthetic_cache()
        head, _ = condition_head(initial, cache)
        offsets = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
        indices = [0, 0, 2, 3]
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        first_loss, first = batch_gradient(
            head, offsets, bank, cache, indices, weights, 4
        )
        second_loss, second = batch_gradient(
            head, offsets, bank, cache, indices, weights, 1
        )
        self.assertAlmostEqual(first_loss, second_loss, places=5)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
        fits = [dict(id=str(i), log_policy=-20 - i) for i in range(4)]
        proposal, groups = make_proposal(bank, fits, 2)
        self.assertAlmostEqual(float(proposal.sum()), 1)
        self.assertTrue((proposal > 0).all())
        for group in groups:
            self.assertAlmostEqual(float(proposal[group].sum()), 0.5)
        # Changing proposal normalization does not renormalize q on this bank.
        self.assertLess(float(path_log_probabilities(head, cache).exp().sum()), 0.1)

    def test_screen_requires_noncopy_gain_over_both_controls(self):
        arms = {
            name: dict(
                joint_success_rate=value,
                joint_success_without_seed_bank_copy=value,
                unique_tail_fraction=1.0,
            )
            for name, value in [
                ("unchanged", 0.1),
                ("plain_tb", 0.13),
                ("svrg_tb", 0.25),
            ]
        }
        self.assertTrue(progression_decision(arms)["passes_practical_screen"])
        arms["svrg_tb"]["joint_success_without_seed_bank_copy"] = 0.14
        self.assertFalse(progression_decision(arms)["passes_practical_screen"])

    def test_cpu_full_pipeline_frozen_backbone_source_isolation_and_recovery(self):
        backbone = FrozenProGen.tiny(11)
        frozen = {k: v.clone() for k, v in backbone.model.state_dict().items()}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No pretrained execution"),
            ),
        ):
            root = Path(directory)
            settings = transfer_source(root, backbone)
            inputs = check_transfer_inputs(settings, False)
            self.assertTrue(all(row["split"] == "train" for row in inputs[5]))
            before = {
                p: file_hash(p) for p in (root / "continuous").rglob("*") if p.is_file()
            }
            run_transfer(
                settings, root / "transfer", "cpu", backbone, fixture_predictor
            )
            for path, digest in before.items():
                self.assertEqual(file_hash(path), digest)
            for name, value in backbone.model.state_dict().items():
                torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in backbone.model.parameters()))
            costs = json.loads((root / "transfer/costs.json").read_text())
            self.assertEqual(costs["training"]["svrg_tb"]["snapshot_refreshes"], 2)
            self.assertEqual(costs["training"]["svrg_tb"]["sampled_occurrences"], 12)
            self.assertEqual(costs["training"]["svrg_tb"]["fresh_training_rollouts"], 0)
            for arm in ("plain_tb", "svrg_tb"):
                saved = torch.load(
                    root / f"transfer/{arm}/checkpoint.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(saved["architecture"], "direct_context")
            for name in ("continuous", "warm", "replay", "data"):
                (root / name).rename(root / f"hidden-{name}")
            reevaluate_transfer(
                root / "transfer", root / "recovered", fixture_predictor
            )
            for name in ("metrics.json", "decision.json", "bank_diagnostics.json"):
                self.assertEqual(
                    json.loads((root / "transfer" / name).read_text()),
                    json.loads((root / "recovered" / name).read_text()),
                )
            with (root / "transfer/samples.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_transfer(
                    root / "transfer", root / "corrupt", fixture_predictor
                )
        self.assertFalse(torch.cuda.is_initialized())
