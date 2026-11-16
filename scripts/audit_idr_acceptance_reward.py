"""CPU-only target audit using inspected prior proposals and saved disorder scores.

No model loading, fitting, generation, filtering, or held-out outcome access.
Run with the project Python; the source hash pins the development pool.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.idr.acceptance_reward import (
    AcceptanceRewardConfig,
    joint_acceptance_reward,
)
from stability_gfn.idr.charge import charge_reward

SOURCE_HASH = "345dbd48f1173f14fb2b7668303f92ffd8ab9e29d83772b4ea082dea5de1503c"
ROOT = Path(__file__).resolve().parents[1]


def audit(source, output):
    """Compare one selected design and a no-acceptance-gap diagnostic ablation."""
    if file_hash(source) != SOURCE_HASH:
        raise ValueError("Source differs from the declared inspected development pool")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows = [r for r in json.loads(source.read_text()) if r["arm"] == "prior_pool"]
    groups = sorted({(r["seed"], r["prompt_id"]) for r in rows})
    if (
        len(rows) != 6144
        or len(groups) != 24
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError("Unexpected proposal panel or duplicate sample IDs")
    settings = AcceptanceRewardConfig()
    config_path = source.parent.parent / "pilot_config.json"
    historical_config = json.loads(config_path.read_text())
    tails = [r["sequence"][r["evaluation_start"] :] for r in rows]
    disorder_scores = [r["disorder"]["V3"]["isolated_tail_scores"] for r in rows]
    result = {
        k: v.numpy()
        for k, v in joint_acceptance_reward(tails, disorder_scores, settings).items()
    }
    old_score, _, _, old_pass = charge_reward(tails, historical_config["reward"])
    np.testing.assert_allclose(old_score.numpy(), [r["charge"]["score"] for r in rows])
    np.testing.assert_array_equal(old_pass.numpy(), result["charge_pass"])
    np.testing.assert_array_equal(
        result["charge_pass"], [r["charge"]["passes"] for r in rows]
    )
    np.testing.assert_array_equal(
        result["disorder_pass"],
        [r["disorder"]["V3"]["isolated_tail"]["passes"] for r in rows],
    )
    log_weights = {
        "prior": np.zeros(len(rows)),
        "historical_beta2": historical_config["beta"] * old_score.numpy(),
        "proximity_only_ablation": np.log(result["proximity_factor"]),
        "joint_acceptance_v1": result["log_factor"],
    }
    metrics = {
        "charge": result["charge_pass"],
        "joint": result["joint_pass"],
        "disorder_v3": result["disorder_pass"],
        "disorder_v2": np.array(
            [r["disorder"]["V2"]["isolated_tail"]["passes"] for r in rows]
        ),
        "repeated_trimer_fraction": np.array(
            [r["tail_descriptors"]["repeated_trimer_fraction"] for r in rows]
        ),
    }
    # Paired, within-group proposal bootstrap. It conditions on these prompts and
    # observed proposal coverage, and cannot account for unseen rare sequences.
    generator = np.random.default_rng(20261113)
    bootstrap_repeats = 2000
    bootstrap_joint = {name: np.zeros(bootstrap_repeats) for name in log_weights}
    by_group = {name: [] for name in log_weights}
    for seed, prompt in groups:
        indices = np.array(
            [
                i
                for i, r in enumerate(rows)
                if (r["seed"], r["prompt_id"]) == (seed, prompt)
            ]
        )
        if len(indices) != 256:
            raise ValueError("Unexpected group size")
        counts = generator.multinomial(
            256, np.full(256, 1 / 256), size=bootstrap_repeats
        )
        for name, log_weight in log_weights.items():
            weight = np.exp(log_weight[indices] - log_weight[indices].max())
            normalized = weight / weight.sum()
            row = {
                "seed": seed,
                "prompt_id": prompt,
                "prior_joint_count": int(result["joint_pass"][indices].sum()),
                "ess": float(1 / np.square(normalized).sum()),
                "max_weight": float(normalized.max()),
                "prior_log_factor_std": float(log_weight[indices].std()),
            }
            row.update(
                {
                    key: float(normalized @ values[indices])
                    for key, values in metrics.items()
                }
            )
            by_group[name].append(row)
            bootstrap_joint[name] += (
                (counts @ (weight * result["joint_pass"][indices])) / (counts @ weight)
            ) / len(groups)
    summary = {}
    for name, group_rows in by_group.items():
        summary[name] = {
            **{key: float(np.mean([r[key] for r in group_rows])) for key in metrics},
            "mean_ess": float(np.mean([r["ess"] for r in group_rows])),
            "median_ess": float(np.median([r["ess"] for r in group_rows])),
            "minimum_ess": min(r["ess"] for r in group_rows),
            "mean_prior_log_factor_std": float(
                np.mean([r["prior_log_factor_std"] for r in group_rows])
            ),
            "maximum_single_weight": max(r["max_weight"] for r in group_rows),
            "joint_bootstrap_95_interval": np.quantile(
                bootstrap_joint[name], [0.025, 0.975]
            ).tolist(),
            "observed_log_factor_range": [
                float(log_weights[name].min()),
                float(log_weights[name].max()),
            ],
        }
    passes = result["log_factor"][result["joint_pass"]]
    failures = result["log_factor"][~result["joint_pass"]]
    if not passes.min() > failures.max():
        raise AssertionError("Joint pass/fail property-factor ranking is inverted")
    evidence = {
        "stage": "post-hoc development analysis; no new training or model inference",
        "source": str(source.resolve()),
        "source_sha256": SOURCE_HASH,
        "config_sha256": file_hash(config_path),
        "code_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/stability_gfn/idr/acceptance_reward.py",
            )
        },
        "reward_version": "joint_acceptance_v1",
        "reward_settings": asdict(settings),
        "scorer": "saved metapredict 3.0.2 V3 isolated-tail scores; V2 sensitivity only",
        "summary": summary,
        "by_group": by_group,
        "bootstrap": {
            "seed": 20261113,
            "repeats": bootstrap_repeats,
            "method": "paired within-group proposal resampling; fixed 24-group average",
        },
        "joint_improvement_over_old_bootstrap_95_interval": np.quantile(
            bootstrap_joint["joint_acceptance_v1"]
            - bootstrap_joint["historical_beta2"],
            [0.025, 0.975],
        ).tolist(),
        "validation": {
            "saved_charge_and_disorder_flags_match": True,
            "all_joint_passes_outrank_failures_in_property_factor": True,
            "prior_joint_counts_per_group": [
                r["prior_joint_count"] for r in by_group["prior"]
            ],
        },
        "limits": "SNIS from p0 proposals: no extra p0 factor. No trained-policy forecast, exact target, independent validation, or novelty claim. Bootstrap omits unobserved modes and prompt selection uncertainty.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, evidence)
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "runs/idr-subtb-v1/tb/sequence_metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/verification/idr-acceptance-reward-2026-11-13.json",
    )
    args = parser.parse_args()
    audit(args.source, args.output)
