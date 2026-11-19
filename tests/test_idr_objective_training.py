"""Tiny CPU checks for capacity-matched GFN and weighted supervised fitting."""

import copy
import unittest

import torch

from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.context_experiment import DirectContextHead
from stability_gfn.idr.lastblock_backbone import LastBlockProGen
from stability_gfn.idr.lastblock_training import trajectory_terms
from stability_gfn.idr.objective_training import (
    ARMS,
    _supervised_weights,
    make_objective_optimizer,
    objective_update,
)


def training_fixture():
    """Unequal prompt counts and tiny rewards expose normalization mistakes."""
    reference = FrozenProGen.tiny(seed=19)
    policies = {arm: LastBlockProGen(reference) for arm in ARMS}
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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24)
        head = DirectContextHead(reference.hidden_size, 7, 20, 8)
        with torch.no_grad():
            head.output.weight.normal_(0, 0.02)
            head.task_weight.normal_(0, 0.01)
    heads = {arm: copy.deepcopy(head) for arm in ARMS}
    normalizer = torch.nn.Parameter(torch.tensor([-0.25, -0.5]))
    sequences = ["AGLSKKKKDDDD", "AGLSAAAAEEEE", "AGLSRRRRNNNN", "KDEGGGGGSSSS"]
    factors = [-1000.0, -1002.0, -1001.0, -1010.0]
    rows = [
        dict(
            id=f"row-{index}",
            sequence=sequence,
            tail=sequence[config["prefix_length"] :],
            prompt_index=int(index == 3),
            prompt_id="second" if index == 3 else "first",
            log_factor=factors[index],
            split="train",
            joint_pass=False,
        )
        for index, sequence in enumerate(sequences)
    ]
    features, logits = reference.forward([sequence[:-1] for sequence in sequences])
    _, log_prior, _ = trajectory_terms(features, logits, None, rows, config, 2)
    for row, prior in zip(rows, log_prior):
        row["log_prior"] = float(prior)
    optimizers = {
        arm: make_objective_optimizer(
            heads[arm],
            config,
            settings,
            policies[arm].last_block,
            normalizer if arm == "gfn" else None,
        )
        for arm in ARMS
    }
    return reference, policies, heads, normalizer, optimizers, rows, config, settings


class ObjectiveTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_heads_and_blocks_learn_with_frozen_reference_and_same_rows(self):
        reference, policies, heads, normalizer, optimizers, rows, config, settings = (
            training_fixture()
        )
        reference_before = copy.deepcopy(reference.model.state_dict())
        heads_before = {arm: copy.deepcopy(heads[arm].state_dict()) for arm in ARMS}
        blocks_before = {
            arm: copy.deepcopy(policies[arm].last_block.state_dict()) for arm in ARMS
        }
        normalizer_before = normalizer.detach().clone()
        metrics = objective_update(
            policies, heads, normalizer, optimizers, rows, config, settings, 4
        )
        optimizer_ids = [
            {
                id(parameter)
                for group in optimizers[arm].param_groups
                for parameter in group["params"]
            }
            for arm in ARMS
        ]
        self.assertFalse(optimizer_ids[0] & optimizer_ids[1])
        for arm in ARMS:
            self.assertEqual(metrics[arm]["example_ids"], [row["id"] for row in rows])
            self.assertGreater(metrics[arm]["head_gradient_norm_before_clip"], 0)
            self.assertGreater(metrics[arm]["block_gradient_norm_before_clip"], 0)
            self.assertGreater(metrics[arm]["update_seconds"], 0)
            self.assertAlmostEqual(metrics[arm]["rates"]["base"], 1e-5)
            self.assertAlmostEqual(metrics[arm]["rates"]["task"], 1e-4)
            self.assertAlmostEqual(metrics[arm]["rates"]["last_block"], 1e-6)
            for module, before in (
                (heads[arm], heads_before[arm]),
                (policies[arm].last_block, blocks_before[arm]),
            ):
                self.assertTrue(
                    all(parameter.grad is not None for parameter in module.parameters())
                )
                self.assertTrue(
                    any(
                        not torch.equal(value, before[name])
                        for name, value in module.state_dict().items()
                    )
                )
                self.assertLessEqual(
                    float(
                        torch.stack(
                            [parameter.grad.norm() for parameter in module.parameters()]
                        ).norm()
                    ),
                    1.00001,
                )
            self.assertGreater(float(heads[arm].task_weight.grad.norm()), 0)
            self.assertGreater(float(heads[arm].hidden.weight.grad.norm()), 0)
        self.assertFalse(torch.equal(normalizer, normalizer_before))
        self.assertAlmostEqual(metrics["gfn"]["rates"]["normalizer"], 1e-3)
        self.assertNotIn("normalizer", metrics["supervised"]["rates"])
        self.assertNotIn("log_z", metrics["supervised"])
        self.assertEqual(metrics["gfn"]["loss"], metrics["gfn"]["tb_loss"])
        self.assertEqual(
            metrics["supervised"]["loss"], metrics["supervised"]["weighted_nll"]
        )
        for name, value in reference.model.state_dict().items():
            torch.testing.assert_close(value, reference_before[name], atol=0, rtol=0)
        self.assertTrue(
            all(parameter.grad is None for parameter in reference.model.parameters())
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_weights_are_stable_prompt_normalized_ranked_and_prior_independent(self):
        _, _, _, _, _, rows, config, _ = training_fixture()
        weights, summary = _supervised_weights(rows, 2, config)
        expected = torch.tensor(
            [-1000.0, -1002.0, -1001.0], dtype=torch.float64
        ).softmax(0)
        torch.testing.assert_close(weights[:3], expected / 2)
        self.assertEqual(float(weights[3]), 0.5)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(float(weights[0]), float(weights[2]))
        self.assertGreater(float(weights[2]), float(weights[1]))
        self.assertTrue(bool((weights > 0).all()))
        self.assertEqual(summary["represented_prompts"], 2)
        self.assertFalse(summary["prior_in_weights"])
        self.assertTrue(summary["normalized_before_microbatch"])
        self.assertEqual([item["examples"] for item in summary["prompts"]], [3, 1])
        self.assertAlmostEqual(summary["prompts"][0]["weight_sum"], 1)
        self.assertAlmostEqual(
            summary["prompts"][0]["effective_sample_size"],
            float(expected.square().sum().reciprocal()),
        )
        changed = [
            dict(row, log_prior=-100 * (index + 1), joint_pass=index == 1)
            for index, row in enumerate(rows)
        ]
        changed_weights, _ = _supervised_weights(changed, 2, config)
        torch.testing.assert_close(weights, changed_weights, atol=0, rtol=0)
        reward = torch.tensor(-1000.0, requires_grad=True)
        detached_weights, _ = _supervised_weights(
            [dict(rows[0], log_factor=reward)], 2, config
        )
        self.assertFalse(detached_weights.requires_grad)
        self.assertIsNone(reward.grad)

    def test_losses_match_original_prior_tb_and_reward_weighted_token_nll(self):
        _, policies, heads, normalizer, optimizers, rows, config, settings = (
            training_fixture()
        )
        with torch.no_grad():
            for policy in policies.values():
                policy.last_block.mlp.fc_out.bias.add_(
                    torch.linspace(-0.1, 0.1, policy.hidden_size)
                )
        expected = {}
        for arm in ARMS:
            features, logits, reference_features, reference_logits = policies[
                arm
            ].forward_with_reference([row["sequence"][:-1] for row in rows])
            log_q, moving_prior, _ = trajectory_terms(
                features, logits, heads[arm], rows, config, 2
            )
            _, fixed_prior, _ = trajectory_terms(
                reference_features, reference_logits, None, rows, config, 2
            )
            self.assertFalse(torch.allclose(moving_prior, fixed_prior))
            if arm == "gfn":
                indices = torch.tensor([row["prompt_index"] for row in rows])
                factors = torch.tensor(
                    [row["log_factor"] for row in rows], dtype=torch.float64
                )
                expected[arm] = float(
                    (
                        normalizer[indices].double()
                        + log_q.double()
                        - fixed_prior.double()
                        - factors
                    )
                    .square()
                    .mean()
                    .detach()
                )
            else:
                prompt_weights = torch.tensor(
                    [row["log_factor"] for row in rows[:3]], dtype=torch.float64
                ).softmax(0)
                nll = -log_q.detach().double() / (
                    config["length"] - config["prefix_length"]
                )
                expected[arm] = float(((prompt_weights * nll[:3]).sum() + nll[3]) / 2)
                self.assertNotAlmostEqual(expected[arm], float(nll.mean()), places=5)
        metrics = objective_update(
            policies, heads, normalizer, optimizers, rows, config, settings, 1
        )
        for arm in ARMS:
            self.assertAlmostEqual(
                metrics[arm]["loss"],
                expected[arm],
                delta=0.001 if arm == "gfn" else 1e-6,
            )

    def test_microbatch_accumulation_preserves_full_prompt_weights_and_gradients(self):
        first = training_fixture()
        second = training_fixture()
        results = []
        for fixture, microbatch_size in zip((first, second), (1, 4)):
            _, policies, heads, normalizer, optimizers, rows, config, settings = fixture
            results.append(
                objective_update(
                    policies,
                    heads,
                    normalizer,
                    optimizers,
                    rows,
                    config,
                    dict(settings, microbatch_size=microbatch_size),
                    1,
                )
            )
        for arm in ARMS:
            self.assertAlmostEqual(
                results[0][arm]["loss"],
                results[1][arm]["loss"],
                delta=0.001 if arm == "gfn" else 1e-6,
            )
            for first_module, second_module in (
                (first[1][arm].last_block, second[1][arm].last_block),
                (first[2][arm], second[2][arm]),
            ):
                for left, right in zip(
                    first_module.parameters(), second_module.parameters()
                ):
                    torch.testing.assert_close(
                        left.grad, right.grad, atol=1e-5, rtol=1e-4
                    )
                    torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-4)
        self.assertEqual(
            results[0]["supervised"]["weighting"], results[1]["supervised"]["weighting"]
        )
        torch.testing.assert_close(first[3], second[3], atol=1e-7, rtol=1e-5)

    def test_invalid_training_rows_are_rejected_before_parameter_updates(self):
        _, policies, heads, normalizer, optimizers, rows, config, settings = (
            training_fixture()
        )
        cases = [
            (dict(rows[-1], split="holdout"), "Held-out"),
            (dict(rows[-1], log_factor=float("nan")), "finite"),
            (dict(rows[-1], log_prior=float("inf")), "finite"),
            (dict(rows[-1], log_prior=rows[-1]["log_prior"] + 1), "Original-reference"),
            (dict(rows[-1], tail="AAAAAAAA"), "Trajectory"),
            (dict(rows[-1], prompt_index=2), "Trajectory"),
            (dict(rows[0], prompt_id="wrong"), "prompt identity"),
        ]
        original = {
            arm: [
                parameter.detach().clone()
                for parameter in [*heads[arm].parameters(), *policies[arm].parameters()]
            ]
            for arm in ARMS
        }
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    objective_update(
                        policies,
                        heads,
                        normalizer,
                        optimizers,
                        [*rows[:-1], invalid],
                        config,
                        settings,
                        1,
                    )
                for arm in ARMS:
                    for actual, before in zip(
                        [*heads[arm].parameters(), *policies[arm].parameters()],
                        original[arm],
                    ):
                        torch.testing.assert_close(actual, before, atol=0, rtol=0)

    def test_shared_trainable_parameters_and_reference_gradients_are_rejected(self):
        reference, policies, heads, normalizer, optimizers, rows, config, settings = (
            training_fixture()
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            objective_update(
                policies,
                heads,
                normalizer,
                dict(optimizers, supervised=optimizers["gfn"]),
                rows,
                config,
                settings,
                1,
            )
        first_parameter = next(reference.model.parameters())
        first_parameter.grad = torch.zeros_like(first_parameter)
        with self.assertRaisesRegex(ValueError, "gradient-free"):
            objective_update(
                policies, heads, normalizer, optimizers, rows, config, settings, 1
            )


if __name__ == "__main__":
    unittest.main()
