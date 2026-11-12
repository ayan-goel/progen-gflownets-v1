"""CPU-only checks for reference provenance, sampling and prefix-safe evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.baseline import (
    check_inputs,
    generate_batch,
    reevaluate,
    run_baseline,
)
from stability_gfn.idr.data import ALPHABET, select_references
from stability_gfn.idr.scoring import descriptors, score_records


class IDRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone = FrozenProGen.tiny(7)

    def test_cached_generation_matches_teacher_forced_probability(self):
        generator = torch.Generator().manual_seed(42)
        sequences, logps, masses = generate_batch(
            self.backbone, ["GSK", "EQA"], 12, generator
        )
        self.assertTrue(sequences[0].startswith("GSK"))
        ids = self.backbone.residue_ids(ALPHABET)
        _, logits = self.backbone.forward(sequences)
        for row, sequence in enumerate(sequences):
            expected = sum(
                logits[row, i, ids].log_softmax(-1)[ALPHABET.index(sequence[i])].item()
                for i in range(3, 12)
            )
            self.assertAlmostEqual(logps[row], expected, places=4)
            self.assertTrue(0 < masses[row] <= 1)
            self.assertEqual(len(sequence), 12)
            self.assertFalse(set(sequence) - set(ALPHABET))
        replay = generate_batch(
            self.backbone, ["GSK", "EQA"], 12, torch.Generator().manual_seed(42)
        )
        self.assertEqual(sequences, replay[0])

    def test_reference_requires_spanning_published_solution_evidence(self):
        def entry(name, sequence, evidence="ECO:0006165", unpublished=False):
            return {
                "disprot_id": name,
                "acc": name,
                "uniref50": name,
                "name": name,
                "organism": "fixture",
                "taxonomy": ["Eukaryota"],
                "sequence": sequence,
                "disprot_consensus": {
                    "Structural state": [{"start": 1, "end": 8, "type": "D"}]
                },
                "regions": [
                    {
                        "term_id": "IDPO:0000002",
                        "ec_id": evidence,
                        "ec_go": "EXP",
                        "validated": True,
                        "reference_id": "fixture-pmid",
                        "region_id": name,
                        "start": 1,
                        "end": 8,
                        "unpublished": unpublished,
                    }
                ],
            }

        entries = [
            entry("valid", "GSKQGSKQ"),
            entry("unpublished", "ACDEACDE", unpublished=True),
            entry("missing_density", "WIVLWIVL", evidence="other"),
        ]
        chosen, count = select_references(entries, length=8, count=1)
        self.assertEqual(count, 1)
        self.assertEqual(chosen[0]["id"], "valid")
        self.assertEqual(chosen[0]["start_1based"], 1)
        entries[0]["regions"][0]["end"] = 7
        with self.assertRaises(ValueError):
            select_references(entries, length=8, count=1)

    def test_supplied_prefix_does_not_count_toward_tail_success(self):
        def predictor(sequences, version):
            return [np.array([1.0 if r == "G" else 0.0 for r in s]) for s in sequences]

        config = {
            "prefix_length": 4,
            "charge_window": 3,
            "disorder_cutoff": 0.5,
            "minimum_disordered_fraction": 0.8,
        }
        refs = [{"id": "ref", "sequence": "GGGGAAAA"}]
        scored = score_records([{"sequence": "GGGGAAAA"}], refs, config, predictor)[0]
        self.assertEqual(scored["disorder"]["V3"]["full"]["mean"], 0.5)
        self.assertEqual(scored["disorder"]["V3"]["context_tail"]["mean"], 0.0)
        self.assertFalse(scored["disorder"]["V3"]["context_tail"]["passes"])
        a, b = descriptors("KKKKEEEE", 3), descriptors("KEKEKEKE", 3)
        self.assertEqual(a["ncpr"], b["ncpr"])
        self.assertEqual(a["composition"], b["composition"])
        self.assertNotEqual(a["charge_profile"], b["charge_profile"])

    def fixture(self, root):
        refs = [
            {"id": "fixture1", "role": "prompt", "sequence": "GSKESQPG" * 3},
            {"id": "fixture2", "role": "reference_only", "sequence": "QPGSEKQS" * 3},
        ]
        data = root / "data"
        data.mkdir()
        write_json(data / "references.json", refs)
        write_json(
            data / "manifest.json",
            {
                "count": 2,
                "prompt_count": 1,
                "references_sha256": file_hash(data / "references.json"),
            },
        )
        return {
            "data": str(data),
            "manifest_sha256": file_hash(data / "manifest.json"),
            "length": 24,
            "prefix_length": 8,
            "batch_size": 2,
            "samples_per_prompt": 2,
            "seeds": [0],
            "temperature": 1.0,
            "precision": "float32",
            "cpu_threads": 1,
            "charge_window": 9,
            "disorder_cutoff": 0.5,
            "minimum_disordered_fraction": 0.8,
        }

    def test_tiny_end_to_end_real_disorder_and_saved_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.fixture(root)
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real weights"),
            ):
                run = root / "run"
                run_baseline(config, run, "cpu", backbone=self.backbone)
                status = json.loads((run / "status.json").read_text())
                self.assertEqual(status["status"], "complete")
                self.assertTrue(status["fixture"])
                metrics = json.loads((run / "metrics.json").read_text())
                self.assertEqual(metrics["arms"]["prompted"]["count"], 2)
                reevaluate(run, root / "evaluation")
                self.assertEqual(
                    json.loads((root / "evaluation/metrics.json").read_text()), metrics
                )
                with self.assertRaises(FileExistsError):
                    run_baseline(config, run, "cpu", backbone=self.backbone)
                (run / "samples.jsonl").write_text("{}\n")
                with self.assertRaises(ValueError):
                    reevaluate(run, root / "corrupt-evaluation")
            self.assertFalse(torch.cuda.is_initialized())

    def test_manifest_tampering_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.fixture(root)
            check_inputs(config, False)
            (root / "data/references.json").write_text("[]")
            with self.assertRaises(ValueError):
                check_inputs(config, False)
