"""Reconstruct saved variance-repair fits and audit pairing/provenance on CPU."""

import argparse
import copy
import json
from pathlib import Path

import torch
from condition_idr_exact_cpu import ConditionedHead, equivalence
from diagnose_idr_updates_cpu import load_task, metrics

from stability_gfn.artifacts import file_hash, write_json


def verify_run(run, trees, initial, candidate):
    results = json.loads((run / "results.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    for name, digest in provenance["source_files"].items():
        assert file_hash(run / "source" / name) == digest
        if name.endswith(".py"):
            assert file_hash(Path(__file__).with_name(name)) == digest
    for name, digest in provenance["input_hashes"].items():
        assert file_hash(Path("runs/idr-continuous-v1") / name) == digest
    for directory, files in json.loads((run / "controls.json").read_text()).items():
        for name, digest in files.items():
            assert file_hash(Path("runs") / directory / name) == digest
    initialization = json.loads((run / "initialization.json").read_text())
    assert initialization["leaf_ids"] == [[r["id"] for r in t.rows] for t in trees]
    assert initialization["fixed_weights"] == [t.fixed_weights.tolist() for t in trees]
    verified = {}
    for name, result in results.items():
        saved = torch.load(run / f"{name}.pt", map_location="cpu", weights_only=True)
        head = copy.deepcopy(candidate)
        head.load_state_dict(saved["head"])
        computed = metrics(trees, head)
        for key in ("mean_tv", "mean_joint_success"):
            assert computed[key] == result[key]
        for prompt, tree in enumerate(trees):
            key = tree.rows[0]["prompt_id"]
            for field, value in computed["by_prompt"][key].items():
                assert result["by_prompt"][key][field] == value
            error = float(saved["log_z"][prompt] - tree.log_reward.logsumexp(0))
            assert result["by_prompt"][key]["normalizer_error"] == error
        worst = max(v["tv"] for v in computed["by_prompt"].values())
        assert worst == result["max_prompt_tv"]
        gate = (
            computed["mean_tv"] < 0.05
            and worst < 0.05
            and abs(computed["mean_joint_success"] - result["target_joint_success"])
            < 0.02
        )
        assert gate == result["passes_gate"]
        folded = copy.deepcopy(initial)
        folded.load_state_dict(
            torch.load(
                run / f"{name}-folded.pt", map_location="cpu", weights_only=True
            )["head"]
        )
        fold_check = equivalence(trees, head, folded)
        assert fold_check["maximum_prompt_tv_between_policies"] < 1e-4
        records = json.loads((run / f"{name}-samples.json").read_text())
        assert len(records) == 400
        count = 0
        for step, row in enumerate(records, 1):
            assert row["step"] == step
            assert len(row["indices"]) == len(trees)
            for tree, indices in zip(trees, row["indices"]):
                assert len(indices) == 32
                assert all(0 <= i < len(tree.rows) for i in indices)
                count += len(indices)
        assert result["sampled_occurrences"] == count == 102400
        if result["method"] == "svrg":
            assert result["snapshots"] == 20
            assert result["snapshot_sample_occurrences"] == count
            assert result["full_snapshot_occurrences"] == 8160
        verified[name] = dict(
            all_metrics_match=True,
            passes_gate=gate,
            sample_indices_valid=True,
            fold_equivalence=fold_check,
        )
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    trees, initial, _, _ = load_task(Path("runs/idr-continuous-v1"))
    candidate = ConditionedHead(initial, trees)
    checks = {}
    for directory in (
        "idr-fixed-plain-v1",
        "idr-fixed-svrg-v1",
        "idr-fixed-svrg-confirm-v1",
    ):
        checks[directory] = verify_run(
            Path("runs") / directory, trees, initial, candidate
        )
    for seed in range(3):
        plain = json.loads(
            Path(f"runs/idr-fixed-plain-v1/plain-{seed}-samples.json").read_text()
        )
        corrected = json.loads(
            Path(f"runs/idr-fixed-svrg-v1/svrg-{seed}-samples.json").read_text()
        )
        assert [r["indices"] for r in plain] == [r["indices"] for r in corrected]
    variance_run = Path("runs/idr-gradient-variance-v1")
    provenance = json.loads((variance_run / "provenance.json").read_text())
    for name, digest in provenance["inputs"].items():
        assert file_hash(Path(name)) == digest
    for name, digest in provenance["source_files"].items():
        assert file_hash(variance_run / "source" / name) == digest
    assert not torch.cuda.is_initialized()
    write_json(
        args.output,
        dict(
            fits=checks,
            paired_indices_identical=True,
            variance_audit_hashes_valid=True,
            cuda_initialized=False,
        ),
    )
    print(
        "All nine fits, raw-variance inputs, paired sample indices, folded heads and source hashes verify."
    )


if __name__ == "__main__":
    main()
