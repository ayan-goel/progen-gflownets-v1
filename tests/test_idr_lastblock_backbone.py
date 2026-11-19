"""Tiny CPU checks for trainable-policy/frozen-reference separation."""

import unittest

import torch

from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.lastblock_backbone import LastBlockProGen


class LastBlockBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.reference = FrozenProGen.tiny(seed=14)
        self.policy = LastBlockProGen(self.reference)
        self.sequences = ["AGLSKA", "KDEGLL"]

    def test_initial_identity_and_shared_frozen_weights(self):
        expected_features, expected_logits = self.reference.forward(self.sequences)
        features, logits, reference_features, reference_logits = (
            self.policy.forward_with_reference(self.sequences)
        )
        torch.testing.assert_close(features, expected_features, atol=0, rtol=0)
        torch.testing.assert_close(logits, expected_logits, atol=0, rtol=0)
        torch.testing.assert_close(reference_logits, expected_logits, atol=0, rtol=0)
        torch.testing.assert_close(
            reference_features, expected_features, atol=0, rtol=0
        )
        self.assertTrue(features.requires_grad)
        self.assertTrue(logits.requires_grad)
        self.assertFalse(reference_logits.requires_grad)
        self.assertFalse(reference_features.requires_grad)
        self.assertIs(self.policy.model, self.reference.model)
        self.assertIs(self.policy.reference, self.reference)
        original_parameters = dict(
            self.reference.model.transformer.h[-1].named_parameters()
        )
        for name, value in self.policy.last_block.named_parameters():
            self.assertNotEqual(value.data_ptr(), original_parameters[name].data_ptr())
        self.assertEqual(
            {id(parameter) for parameter in self.policy.parameters()},
            {id(parameter) for parameter in self.policy.last_block.parameters()},
        )
        tokens = sum(len(sequence) + 1 for sequence in self.sequences)
        self.assertEqual(self.policy.work["input_tokens"], tokens)
        self.assertEqual(self.policy.work["adapted_block_token_evaluations"], tokens)
        self.assertEqual(self.policy.work["reference_block_token_evaluations"], tokens)

    def test_incremental_generation_matches_full_policy_after_adaptation(self):
        # Simulate a changed policy without making the reference follow it.
        with torch.no_grad():
            self.policy.last_block.mlp.fc_out.bias.add_(
                torch.linspace(-0.01, 0.01, self.reference.hidden_size)
            )
        expected_features, expected_logits = self.policy.forward(self.sequences)
        ids = self.policy.encode(self.sequences)
        cache = None
        for position in range(ids.shape[1]):
            features, logits, cache = self.policy.step(
                ids[:, position : position + 1], cache
            )
            torch.testing.assert_close(
                features, expected_features[:, position], atol=1e-6, rtol=1e-5
            )
            torch.testing.assert_close(
                logits, expected_logits[:, position], atol=1e-6, rtol=1e-5
            )
            self.assertFalse(features.requires_grad)
            self.assertFalse(logits.requires_grad)
            self.assertEqual(len(cache), len(self.reference.model.transformer.h))
            self.assertEqual(cache[-1][0].shape[-2], position + 1)
        # A multi-token initial prefix and subsequent token use the same cache.
        _, _, cache = self.policy.step(ids[:, :4])
        features, logits, _ = self.policy.step(ids[:, 4:5], cache)
        torch.testing.assert_close(
            features, expected_features[:, 4], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(logits, expected_logits[:, 4], atol=1e-6, rtol=1e-5)

    def test_only_copied_block_changes_and_reference_remains_exact(self):
        original_state = {
            name: value.clone()
            for name, value in self.reference.model.state_dict().items()
        }
        _, expected_reference = self.reference.forward(self.sequences)
        _, before, _, _ = self.policy.forward_with_reference(self.sequences)
        optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        _, logits, _, reference_logits = self.policy.forward_with_reference(
            self.sequences
        )
        # Teacher-forced next residues exclude the final prediction position.
        labels = self.policy.encode(self.sequences)[:, 1:]
        loss = -logits[:, :-1].log_softmax(-1).gather(-1, labels[..., None]).mean()
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in self.policy.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad.abs().sum() > 0 for parameter in self.policy.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None and not parameter.requires_grad
                for parameter in self.reference.model.parameters()
            )
        )
        self.assertFalse(reference_logits.requires_grad)
        optimizer.step()
        _, after, _, actual_reference = self.policy.forward_with_reference(
            self.sequences
        )
        self.assertFalse(torch.equal(before, after))
        torch.testing.assert_close(actual_reference, expected_reference, atol=0, rtol=0)
        for name, value in self.reference.model.state_dict().items():
            torch.testing.assert_close(value, original_state[name], atol=0, rtol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_causal_features_and_dropout_policy_are_consistent(self):
        self.policy.last_block.train()
        features, logits = self.policy.forward(["AGLS", "AGAA"])
        self.assertFalse(self.policy.last_block.training)
        torch.testing.assert_close(features[0, :3], features[1, :3], atol=0, rtol=0)
        torch.testing.assert_close(logits[0, :3], logits[1, :3], atol=0, rtol=0)
        self.assertTrue(logits.requires_grad)
        with self.assertRaisesRegex(ValueError, "one entry"):
            self.policy.step(self.policy.encode(["AG"]), (None,))


if __name__ == "__main__":
    unittest.main()
