"""CPU-only penalty calibration on unselected historical TRAINING proposals.

    PYTHONPATH=src .venv/bin/python scripts/audit_idr_continuous_reward.py

Seed 1 calibrates; seed 2 is a fixed diagnostic, never used to adjust the penalty.
Samples came from adaptive policies: use p0/behavior importance corrections.
These finite-sample estimates do not establish attainable generation success.
"""

import json
from pathlib import Path

import numpy as np

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.idr.acceptance_experiment import verify_manifest


def target_summary(rows, penalty=None):
    """Self-normalized importance estimates per prompt, without deduplication."""
    groups = {}
    for prompt in sorted({row["prompt_id"] for row in rows}):
        subset = [row for row in rows if row["prompt_id"] == prompt]
        passed = np.array([row["joint_pass"] for row in subset])
        violations = np.array([sum(row["violations"]) for row in subset])
        log_factor = (
            np.array([row["log_factor"] for row in subset])
            if penalty is None
            else -penalty * violations
        )
        log_weight = log_factor + np.array(
            [row["log_prior"] - row["behavior_log_policy"] for row in subset]
        )
        weights = np.exp(log_weight - log_weight.max())
        weights /= weights.sum()
        groups[prompt] = {
            "count": len(subset),
            "observed_passes": int(passed.sum()),
            "estimated_joint_mass": float(weights @ passed),
            "effective_sample_size": float(1 / (weights @ weights)),
            "largest_weight_fraction": float(weights.max()),
            "log_factor_range": [float(log_factor.min()), float(log_factor.max())],
        }
    return {
        "by_prompt": groups,
        "mean_prompt_joint_mass": float(
            np.mean([group["estimated_joint_mass"] for group in groups.values()])
        ),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source = root / "runs/idr-acceptance-v1"
    verify_manifest(source)
    bank_path = root / "runs/idr-replay-v1/bank.json"
    bank = json.loads(bank_path.read_text())
    diagnostic_tails = {row["tail"] for row in bank if row["split"] == "holdout"}
    rows = {}
    paths = {}
    for seed in (1, 2):
        path = source / f"seed-{seed}/reward_inputs.jsonl"
        paths[str(path.relative_to(root))] = file_hash(path)
        rows[seed] = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows[seed]) == 6400
        # Do not fit reward strength on variants reserved for bank diagnostics.
        assert not any(row["tail"] in diagnostic_tails for row in rows[seed])
    target = target_summary(rows[1])["mean_prompt_joint_mass"]
    lower, upper = 0.0, 100.0
    for _ in range(60):
        midpoint = (lower + upper) / 2
        if target_summary(rows[1], midpoint)["mean_prompt_joint_mass"] < target:
            lower = midpoint
        else:
            upper = midpoint
    calibrated = (lower + upper) / 2
    selected = round(calibrated, 1)
    result = {
        "analysis_source_sha256": file_hash(Path(__file__)),
        "source_generation_sha256": file_hash(source / "generation.json"),
        "training_proposal_hashes": paths,
        "diagnostic_bank_sha256": file_hash(bank_path),
        "diagnostic_bank_tail_overlap": 0,
        "selection_rule": "Match equal-prompt current target joint mass on seed 1; round penalty to one decimal. Seed 2 is diagnostic only.",
        "calibrated_penalty": calibrated,
        "selected_penalty": selected,
        "importance_correction": "exp(log_prior - behavior_log_policy + log_factor), normalized within prompt; keep every trajectory, including repeats",
        "caveat": "Development estimates from adaptive training proposals, not independent test outcomes or exact target probabilities. One prompt has only 2/1 observed successes in calibration/diagnostic.",
        "calibration": {
            "current": target_summary(rows[1]),
            "continuous": target_summary(rows[1], selected),
        },
        "diagnostic": {
            "current": target_summary(rows[2]),
            "continuous": target_summary(rows[2], selected),
        },
    }
    destination = root / "docs/verification/idr-continuous-reward-audit-2026-11-15.json"
    write_json(destination, result)
    print(destination)
    print(json.dumps({"selected_penalty": selected, "calibrated_penalty": calibrated}))
