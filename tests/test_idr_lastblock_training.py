"""CPU fixtures for causal likelihoods and matched GFlowNet gradient updates."""

import copy
import unittest

import torch

from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.charge import rollout
from stability_gfn.idr.context_experiment import DirectContextHead
from stability_gfn.idr.lastblock_backbone import LastBlockProGen
from stability_gfn.idr.lastblock_training import (
    bank_fits,
    make_optimizer,
    paired_update,
    trajectory_terms,
)


def training_fixture():
    reference = FrozenProGen.tiny(seed=19)
    adapted = LastBlockProGen(reference)
    config = dict(
        prefix_length=4,
        length=12,
        learning_rate=1e-4,
        normalizer_learning_rate=1e-2,
        gradient_clip=1.0,
        steps=4,
        decay_fraction=0.75,
        final_lr_factor=0.1,
        reward=dict(block_targets=[0.125, -0.125]),
    )
    settings = dict(
        microbatch_size=2, last_block_learning_rate=1e-5, block_gradient_clip=1.0
    )
    prompts = [
        dict(id="first", sequence="AGLSAAAA"),
        dict(id="second", sequence="KDEGAAAA"),
    ]
    head = DirectContextHead(reference.hidden_size, len(prompts) + 5, 20, 8)
    with torch.no_grad():
        head.output.weight.fill_(0.003)
        head.task_weight[0, 0] = 0.02
    heads = {arm: copy.deepcopy(head) for arm in ("frozen_tb", "lastblock_tb")}
    normalizers = {
        arm: torch.nn.Parameter(torch.tensor([-0.25, -0.5])) for arm in heads
    }
    draw = rollout(
        reference, head, prompts, [0, 1, 0], config, torch.Generator().manual_seed(24)
    )
    rows = [
        dict(
            id=f"row-{i}",
            sequence=sequence,
            tail=sequence[config["prefix_length"] :],
            prompt_index=prompt,
            prompt_id=prompts[prompt]["id"],
            log_prior=float(draw["log_prior"][i]),
            log_factor=-0.4 * (i + 1),
            split="train",
            joint_pass=i == 0,
        )
        for i, (sequence, prompt) in enumerate(zip(draw["sequences"], [0, 1, 0]))
    ]
    return reference, adapted, config, settings, prompts, heads, normalizers, rows, draw


class LastBlockTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_teacher_forced_terms_match_incremental_rollout(self):
        reference, adapted, config, _, prompts, heads, _, rows, draw = (
            training_fixture()
        )
        features, logits = reference.forward([row["sequence"] for row in rows])
        log_q, log_prior, mass = trajectory_terms(
            features, logits, heads["frozen_tb"], rows, config, len(prompts)
        )
        torch.testing.assert_close(log_q, draw["log_policy"], atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(log_prior, draw["log_prior"], atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(mass, draw["canonical_mass"], atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            adapted.last_block.mlp.fc_out.bias.add_(
                torch.linspace(-0.01, 0.01, reference.hidden_size)
            )
        moved = rollout(
            adapted,
            heads["lastblock_tb"],
            prompts,
            [0, 1, 0],
            config,
            torch.Generator().manual_seed(24),
        )
        moved_rows = [
            dict(row, sequence=sequence, tail=sequence[config["prefix_length"] :])
            for row, sequence in zip(rows, moved["sequences"])
        ]
        features, logits = adapted.forward([row["sequence"][:-1] for row in moved_rows])
        actual = trajectory_terms(
            features, logits, heads["lastblock_tb"], moved_rows, config, len(prompts)
        )
        torch.testing.assert_close(actual[0], moved["log_policy"], atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(actual[1], moved["log_prior"], atol=3e-6, rtol=1e-5)

    def test_paired_update_preserves_reference_and_uses_same_examples(self):
        reference, adapted, config, settings, prompts, heads, normalizers, rows, _ = (
            training_fixture()
        )
        original = {
            name: value.clone() for name, value in reference.model.state_dict().items()
        }
        optimizers = {
            arm: make_optimizer(
                heads[arm],
                normalizers[arm],
                config,
                settings,
                adapted.last_block if arm == "lastblock_tb" else None,
            )
            for arm in heads
        }
        before = bank_fits(
            adapted,
            heads["lastblock_tb"],
            normalizers["lastblock_tb"],
            rows,
            config,
            prompts,
            "lastblock_tb",
            set(),
            2,
        )
        metrics = paired_update(
            adapted, heads, normalizers, optimizers, rows, config, settings, 4
        )
        self.assertEqual(
            metrics["frozen_tb"]["example_ids"], metrics["lastblock_tb"]["example_ids"]
        )
        self.assertAlmostEqual(
            metrics["frozen_tb"]["tb_loss"],
            metrics["lastblock_tb"]["tb_loss"],
            places=6,
        )
        self.assertAlmostEqual(metrics["lastblock_tb"]["rates"]["last_block"], 1e-6)
        self.assertAlmostEqual(metrics["frozen_tb"]["rates"]["base"], 1e-5)
        for name, value in reference.model.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
        after = bank_fits(
            adapted,
            heads["lastblock_tb"],
            normalizers["lastblock_tb"],
            [dict(row, split="holdout") for row in rows],
            config,
            prompts,
            "lastblock_tb",
            {rows[-1]["tail"]},
            2,
        )
        self.assertTrue(after[-1]["fresh_training_overlap"])
        self.assertEqual(after[-1]["split"], "holdout")
        self.assertNotEqual(before[0]["log_policy"], after[0]["log_policy"])
        self.assertEqual(before[0]["log_prior"], after[0]["log_prior"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_adapted_reward_stays_anchored_to_original_prior(self):
        _, adapted, config, settings, prompts, heads, normalizers, rows, _ = (
            training_fixture()
        )
        with torch.no_grad():
            adapted.last_block.mlp.fc_out.bias.add_(
                torch.linspace(-0.02, 0.02, adapted.hidden_size)
            )
        features, logits, reference_features, reference_logits = (
            adapted.forward_with_reference([row["sequence"] for row in rows])
        )
        log_q, moving_base, _ = trajectory_terms(
            features, logits, heads["lastblock_tb"], rows, config, len(prompts)
        )
        _, fixed_base, _ = trajectory_terms(
            reference_features, reference_logits, None, rows, config, len(prompts)
        )
        self.assertFalse(torch.allclose(moving_base, fixed_base))
        indices = torch.tensor([row["prompt_index"] for row in rows])
        factors = torch.tensor([row["log_factor"] for row in rows], dtype=torch.float64)
        expected = (
            (
                normalizers["lastblock_tb"][indices].double()
                + log_q.double()
                - fixed_base.double()
                - factors
            )
            .square()
            .mean()
        )
        optimizers = {
            arm: make_optimizer(
                heads[arm],
                normalizers[arm],
                config,
                settings,
                adapted.last_block if arm == "lastblock_tb" else None,
            )
            for arm in heads
        }
        metrics = paired_update(
            adapted, heads, normalizers, optimizers, rows, config, settings, 1
        )
        self.assertAlmostEqual(
            metrics["lastblock_tb"]["tb_loss"], float(expected.detach()), places=5
        )
        with self.assertRaisesRegex(ValueError, "Held-out"):
            paired_update(
                adapted,
                heads,
                normalizers,
                optimizers,
                [dict(rows[0], split="holdout")],
                config,
                settings,
                2,
            )

    def test_accumulation_matches_single_batch_and_fixed_reference_is_checked(self):
        fixture = training_fixture()
        reference, adapted, config, settings, _, heads, normalizers, rows, _ = fixture
        other_adapted = LastBlockProGen(reference)
        other_heads = copy.deepcopy(heads)
        other_normalizers = {
            arm: torch.nn.Parameter(value.detach().clone())
            for arm, value in normalizers.items()
        }

        def optimizers_for(backbone, arm_heads, arm_normalizers):
            return {
                arm: make_optimizer(
                    arm_heads[arm],
                    arm_normalizers[arm],
                    config,
                    settings,
                    backbone.last_block if arm == "lastblock_tb" else None,
                )
                for arm in arm_heads
            }

        first_optimizers = optimizers_for(adapted, heads, normalizers)
        second_optimizers = optimizers_for(
            other_adapted, other_heads, other_normalizers
        )
        first = paired_update(
            adapted,
            heads,
            normalizers,
            first_optimizers,
            rows,
            config,
            dict(settings, microbatch_size=1),
            1,
        )
        second = paired_update(
            other_adapted,
            other_heads,
            other_normalizers,
            second_optimizers,
            rows,
            config,
            dict(settings, microbatch_size=3),
            1,
        )
        for arm in heads:
            self.assertAlmostEqual(
                first[arm]["tb_loss"], second[arm]["tb_loss"], places=5
            )
            for left, right in zip(
                heads[arm].parameters(), other_heads[arm].parameters()
            ):
                torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-4)
        for left, right in zip(adapted.parameters(), other_adapted.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-4)
        invalid = [dict(row, log_prior=row["log_prior"] + 1) for row in rows]
        with self.assertRaisesRegex(ValueError, "Original-reference"):
            paired_update(
                adapted,
                heads,
                normalizers,
                first_optimizers,
                invalid,
                config,
                settings,
                2,
            )


if __name__ == "__main__":
    unittest.main()
