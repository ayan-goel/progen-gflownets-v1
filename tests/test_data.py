"""Synthetic CSV checks for missing labels, replicate handling and split isolation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from stability_gfn.data import REQUIRED_COLUMNS, prepare_local_task


class DataTests(unittest.TestCase):
    def test_missing_labels_duplicates_and_test_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            rows = []
            for index, sequence in enumerate(
                [
                    "AAAA",
                    "GAAA",
                    "LAAA",
                    "SAAA",
                    "AGAA",
                    "ALAA",
                    "ASAA",
                    "AAGA",
                    "AALA",
                    "AASA",
                    "AAAG",
                ]
            ):
                rows.append(
                    {
                        "name": str(index),
                        "aa_seq": sequence,
                        "WT_name": "parent",
                        "WT_cluster": "cluster",
                        "mut_type": "wt" if index == 0 else "mut",
                        "dG_ML": str(index),
                        "ddG_ML": str(index),
                    }
                )
            rows.extend(
                [dict(rows[1]), dict(rows[2], dG_ML="-"), dict(rows[2], dG_ML="inf")]
            )
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            output = Path(directory) / "task"
            manifest = prepare_local_task(source, output, min_variants=5, min_length=4)
            self.assertEqual(manifest["audit_counts"]["missing_or_censored_dG_ML"], 2)
            sets = []
            for split in ["train", "validation", "test_labels"]:
                sets.append(
                    {
                        row["sequence"]
                        for row in json.loads((output / f"{split}.json").read_text())
                    }
                )
            self.assertFalse(
                sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
            )
            self.assertEqual(
                set(json.loads((output / "candidates.json").read_text())), sets[2]
            )
            self.assertEqual(manifest["unique_local_sequences"], 11)
            self.assertIn("AAAA", sets[0])
            second = Path(directory) / "task2"
            prepare_local_task(source, second, min_variants=5, min_length=4)
            self.assertEqual(
                (output / "train.json").read_bytes(),
                (second / "train.json").read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                prepare_local_task(source, output)


if __name__ == "__main__":
    unittest.main()
