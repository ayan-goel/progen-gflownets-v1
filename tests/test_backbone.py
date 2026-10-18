"""CPU integration checks against tiny instances of the actual upstream model."""

import tempfile
import unittest

import torch

from stability_gfn.backbone import FrozenProGen, cache_tree
from stability_gfn.environment import PrefixTree, SequenceContext


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone = FrozenProGen.tiny()

    def test_direction_marker_and_cache_alignment(self):
        backbone = self.backbone
        self.assertNotEqual(backbone.boundary_id, [1])
        states, logits = backbone.forward(["AGLS"])
        token_ids = backbone.encode(["AGLS"])
        cache = None
        for position in range(token_ids.shape[1]):
            hidden, next_logits, cache = backbone.step(
                token_ids[:, position : position + 1], cache
            )
            torch.testing.assert_close(
                hidden, states[:, position], atol=1e-6, rtol=1e-5
            )
            torch.testing.assert_close(
                next_logits, logits[:, position], atol=1e-6, rtol=1e-5
            )
        self.assertFalse(
            any(parameter.requires_grad for parameter in backbone.model.parameters())
        )

    def test_local_checkpoint_loader_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            self.backbone.model.save_pretrained(directory)
            restored = FrozenProGen.from_checkpoint(directory, "cpu")
            _, expected = self.backbone.forward(["AGL"])
            _, actual = restored.forward(["AGL"])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertFalse(
                any(
                    parameter.requires_grad for parameter in restored.model.parameters()
                )
            )
            with self.assertRaises(ValueError):
                FrozenProGen.from_checkpoint(directory, "cpu", precision="float16")

    def test_prefix_features_have_no_future_leakage(self):
        backbone = self.backbone
        states, logits = backbone.forward(["AGLS", "AGAA"])
        torch.testing.assert_close(states[0, :3], states[1, :3], atol=0, rtol=0)
        tree = PrefixTree(SequenceContext("AAAA", 4, "AGLS"), ["AGLS", "AGAA"])
        features, cached_logits = cache_tree(backbone, tree, 2)
        for leaf, sequence in enumerate(tree.sequences):
            full_states, full_logits = backbone.forward([sequence])
            torch.testing.assert_close(features[tree.paths[leaf]], full_states[0, :-1])
            torch.testing.assert_close(
                cached_logits[tree.paths[leaf]],
                full_logits[0, :-1][:, backbone.residue_ids("AGLS")],
            )
        self.assertFalse(features.requires_grad)
        torch.testing.assert_close(
            backbone.sequence_features(["AGLS"]), states[:1, 1:].mean(1)
        )


if __name__ == "__main__":
    unittest.main()
