"""Fixed-objective gradient and line-search budget safety checks."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_exact_diagnostic import diagnostic, fixture

from stability_gfn.idr.context_experiment import DirectContextHead


def load_script(name, dependencies):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


updates = load_script(
    "diagnose_idr_updates_cpu", {"diagnose_idr_exact_cpu": diagnostic}
)
optimizer_check = load_script(
    "fit_idr_exact_optimizer_cpu", {"diagnose_idr_updates_cpu": updates}
)
conditioning = load_script(
    "condition_idr_exact_cpu",
    {
        "diagnose_idr_updates_cpu": updates,
        "fit_idr_exact_optimizer_cpu": optimizer_check,
    },
)


class UpdateDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rows, cache = fixture()
        self.tree = diagnostic.FiniteBankTree(rows, cache)
        self.tree.log_reward = self.tree.log_prior + torch.tensor([0.0, -2.0, 0.0])
        self.tree.target = self.tree.log_reward.softmax(0)
        torch.manual_seed(21)
        self.head = DirectContextHead(4, 6, 20, 8)
        log_q = self.tree.log_probabilities(self.head).detach().double()
        passed = self.tree.passed
        self.tree.fixed_weights = (
            0.5 * log_q.exp()
            + 0.25 * passed / passed.sum()
            + 0.25 * (~passed) / (~passed).sum()
        )
        self.normalizer = torch.nn.Parameter(torch.zeros(1))
        self.config = dict(
            steps=400,
            learning_rate=0.0001,
            normalizer_learning_rate=0.01,
            gradient_clip=1.0,
            decay_fraction=0.75,
            final_lr_factor=0.1,
        )

    def test_fixed_weight_normalizer_gradient_and_repeatability(self):
        tree = self.tree
        self.assertAlmostEqual(float(tree.fixed_weights.sum()), 1.0, places=6)
        self.assertTrue((tree.fixed_weights > 0).all())
        self.assertFalse(tree.fixed_weights.requires_grad)
        loss = updates.objective([tree], self.head, self.normalizer)
        gradient = torch.autograd.grad(loss, self.normalizer)[0]
        residual = (
            self.normalizer.detach()[0]
            + tree.log_probabilities(self.head).detach()
            - tree.log_reward
        )
        expected = 2 * (tree.fixed_weights * residual).sum()
        self.assertAlmostEqual(float(gradient[0]), float(expected), places=6)
        self.assertEqual(
            float(loss), float(updates.objective([tree], self.head, self.normalizer))
        )

    def test_interrupted_line_search_restores_initial_accepted_state(self):
        initial = {k: v.clone() for k, v in self.head.state_dict().items()}
        result, final, normalizer = optimizer_check.fit_fixed_tb(
            [self.tree], self.head, self.normalizer, self.config, "lbfgs", budget=1
        )
        self.assertEqual(result["gradient_evaluations"], 1)
        self.assertEqual(result["accepted_updates"], 0)
        self.assertIn("restored", result["stop_reason"])
        for key, value in final.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        torch.testing.assert_close(normalizer, self.normalizer, rtol=0, atol=0)

    def test_optimizers_reduce_same_objective_and_leave_source_unchanged(self):
        baseline = float(updates.objective([self.tree], self.head, self.normalizer))
        before = {k: v.clone() for k, v in self.head.state_dict().items()}
        for arm in ("adam_clipped", "adam_unclipped", "lbfgs"):
            result, _, _ = optimizer_check.fit_fixed_tb(
                [self.tree], self.head, self.normalizer, self.config, arm, budget=80
            )
            self.assertLessEqual(result["gradient_evaluations"], 80)
            self.assertLess(result["fixed_tb_loss"], baseline)
            self.assertEqual(result["actual_clipping"], arm == "adam_clipped")
            if arm == "lbfgs":
                self.assertLess(result["mean_tv"], 0.02)
        for key, value in self.head.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_conditioning_preserves_function_capacity_and_folding(self):
        conditioned = conditioning.ConditionedHead(self.head, [self.tree])
        self.assertEqual(
            sum(p.numel() for p in conditioned.parameters()),
            sum(p.numel() for p in self.head.parameters()),
        )
        check = conditioning.equivalence([self.tree], self.head, conditioned)
        self.assertLess(check["maximum_prompt_tv_between_policies"], 1e-5)
        generator = torch.Generator().manual_seed(50)
        features = torch.randn(12, 4, generator=generator)
        context = torch.randn(12, 6, generator=generator)
        torch.testing.assert_close(
            conditioned(features, context),
            self.head(features, context),
            atol=1e-5,
            rtol=1e-5,
        )
        with torch.no_grad():
            conditioned.hidden.weight.add_(0.01)
            conditioned.normalization.weight.mul_(1.01)
        torch.testing.assert_close(
            conditioned(features, context),
            conditioned.fold()(features, context),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertAlmostEqual(
            conditioned.conditioning["transformed_condition_number_double"],
            1.0,
            places=8,
        )

    def test_conditioning_does_not_use_reward_labels(self):
        first = conditioning.ConditionedHead(self.head, [self.tree])
        self.tree.target = self.tree.target.flip(0)
        self.tree.log_reward = self.tree.log_reward.flip(0)
        second = conditioning.ConditionedHead(self.head, [self.tree])
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
