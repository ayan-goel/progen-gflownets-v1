#!/usr/bin/env python3
"""Read saved charge checkpoints and run tiny exact CPU learning controls.

No pretrained backbone is loaded. The synthetic task is two actions long with
three nominal-charge actions; it tests gradient plumbing/optimization, not proteins.
Run through .venv with PYTHONPATH=src and CUDA_VISIBLE_DEVICES=''.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from stability_gfn.artifacts import file_hash, new_run, write_json
from stability_gfn.guidance import GuidanceHead
from stability_gfn.idr.charge import charge_reward, context_features
from stability_gfn.idr.data import ALPHABET


def checkpoint_audit(run):
    """Compare real trained heads with their seeded initialization, without ProGen."""
    results = []
    for seed in range(3):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            head = GuidanceHead(1536, 13, 20, 128)
        initial = {key: value.clone() for key, value in head.state_dict().items()}
        stages = []
        for step in (100, 200, 300, 400):
            saved = torch.load(
                run / f"seed-{seed}/step-{step}.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert saved["step"] == step
            stages.append(
                {
                    "step": step,
                    "learning_rates": [
                        g["lr"] for g in saved["optimizer"]["param_groups"]
                    ],
                    "update_norms": {
                        name: float((value - initial[name]).norm())
                        for name, value in saved["head"].items()
                    },
                    "output_weight_norm": float(saved["head"]["output.weight"].norm()),
                }
            )
        head.load_state_dict(saved["head"])
        features = torch.randn(32, 1536, generator=torch.Generator().manual_seed(456))
        first = context_features(
            [""] * 32, [i % 8 for i in range(32)], 8, 64, [0.125, -0.125], "cpu"
        )
        second = context_features(
            ["A" * 32] * 32, [i % 8 for i in range(32)], 8, 64, [0.125, -0.125], "cpu"
        )
        with torch.no_grad():
            first_logits, second_logits = head(features, first), head(features, second)
            difference = first_logits - second_logits
            charges = torch.tensor(
                [1 if a in "KR" else -1 if a in "DE" else 0 for a in ALPHABET]
            )
            contrast = float(
                ((first_logits.softmax(-1) - second_logits.softmax(-1)) * charges)
                .sum(-1)
                .mean()
            )
        logs = [
            json.loads(line)
            for line in (run / f"seed-{seed}/training.jsonl").read_text().splitlines()
        ]
        results.append(
            {
                "seed": seed,
                "checkpoints": stages,
                "synthetic_fixed_feature_charge_contrast": contrast,
                "synthetic_fixed_feature_max_centered_logit_contrast": float(
                    (difference - difference.mean(-1, keepdim=True)).abs().max()
                ),
                "clip_fraction": float(
                    np.mean([row["gradient_norm_before_clip"] > 1 for row in logs])
                ),
                "median_clip_multiplier": float(
                    np.median(
                        [min(1, 1 / row["gradient_norm_before_clip"]) for row in logs]
                    )
                ),
                "final_log_z": saved["log_z"].tolist(),
            }
        )
    return results


def tiny_learning(seed, condition, steps=300):
    """Exactly enumerable tree: K/E/A, two residues, four nonterminal states.

    Frozen features are four fixed Gaussian vectors, not pretrained representations.
    Target charge is +1 then -1, FCR=1; adjusted for a realizable two-token fixture.
    All conditions share features, initialization, target, duration and sampling RNG.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = GuidanceHead(1536, 6, 20, 128)
    features = torch.randn(4, 1536, generator=torch.Generator().manual_seed(123))
    action_ids = [ALPHABET.index(a) for a in "KEA"]
    contexts = torch.cat(
        [
            context_features([""], [0], 1, 2, [1.0, -1.0], "cpu"),
            context_features(list("KEA"), [0] * 3, 1, 2, [1.0, -1.0], "cpu"),
        ]
    )
    actions = torch.tensor(list(itertools.product(range(3), repeat=2)))
    paths = torch.stack([torch.zeros(9, dtype=torch.long), 1 + actions[:, 0]], dim=1)
    tails = ["".join("KEA"[a] for a in pair) for pair in actions.tolist()]
    reward = {
        "block_targets": [1.0, -1.0],
        "block_scale": 1.0,
        "fcr_target": 1.0,
        "fcr_scale": 1.0,
        "block_tolerance": 0.1,
        "fcr_tolerance": 0.1,
    }
    scores = charge_reward(tails, reward)[0]
    prior_logits = torch.tensor([0.1, 0.1, 0.8]).log()
    log_prior = prior_logits[actions].sum(-1)
    log_weights = log_prior + 2 * scores
    exact_log_z = log_weights.logsumexp(0)
    target = (log_weights - exact_log_z).exp()
    normalizer = torch.nn.Parameter(torch.zeros(()))
    learning_rate = 1e-3 if condition == "sample8_higher_lr" else 1e-4
    optimizer = torch.optim.Adam(
        [
            {"params": list(head.parameters()), "lr": learning_rate},
            {"params": [normalizer], "lr": 0.01},
        ]
    )
    rng = torch.Generator().manual_seed(1000 + seed)
    clips = 0
    history = []
    for step in range(1, steps + 1):
        log_conditionals = (
            prior_logits + head(features, contexts)[:, action_ids]
        ).log_softmax(-1)
        log_q = log_conditionals[paths, actions].sum(-1)
        residual = normalizer + log_q - log_weights
        if condition == "exact_expectation":
            # Expected sampled TB gradient: trajectory probabilities are detached.
            loss = (log_q.detach().exp() * residual.square()).sum()
        else:
            sampled = torch.multinomial(
                log_q.detach().exp(), 8, replacement=True, generator=rng
            )
            loss = residual[sampled].square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        maximum = float("inf") if condition == "sample8_no_clip" else 1.0
        norm = torch.nn.utils.clip_grad_norm_(
            [*head.parameters(), normalizer], maximum, error_if_nonfinite=True
        )
        clips += float(norm) > 1
        optimizer.step()
        if step in (1, 100, 200, steps):
            history.append(
                {
                    "step": step,
                    "pre_update_tv": float(
                        (log_q.detach().exp() - target).abs().sum() / 2
                    ),
                    "loss": float(loss.detach()),
                }
            )
    with torch.no_grad():
        conditionals = (
            prior_logits + head(features, contexts)[:, action_ids]
        ).log_softmax(-1)
        final_log_q = conditionals[paths, actions].sum(-1)
    assert abs(float(final_log_q.exp().sum()) - 1) < 1e-5
    return {
        "seed": seed,
        "condition": condition,
        "steps": steps,
        "prior_tv": float((log_prior.exp() - target).abs().sum() / 2),
        "final_tv": float((final_log_q.exp() - target).abs().sum() / 2),
        "exact_log_z": float(exact_log_z),
        "learned_log_z": float(normalizer.detach()),
        "fraction_norm_above_one": clips / steps,
        "history": history,
        "target_probabilities": dict(zip(tails, target.tolist())),
        "learned_probabilities": dict(zip(tails, final_log_q.exp().tolist())),
    }


def gradient_check():
    """Compare autograd with central differences, fixing the sampling weights."""
    logits = torch.tensor(
        [[0.1, -0.1, 0.2], [0.3, -0.2, 0.1]], dtype=torch.float64, requires_grad=True
    )
    actions = torch.tensor(list(itertools.product(range(3), repeat=2)))
    log_prior = torch.tensor([0.1, 0.1, 0.8], dtype=torch.float64).log()[actions].sum(1)
    log_weights = log_prior - torch.arange(9, dtype=torch.float64) / 5
    detached_weights = torch.full((9,), 1 / 9, dtype=torch.float64)

    def loss(value):
        log_q = value.log_softmax(-1)[torch.arange(2)[None, :], actions].sum(1)
        return (detached_weights * (log_q - log_weights - 1).square()).sum()

    analytic = torch.autograd.grad(loss(logits), logits)[0]
    numeric = torch.empty_like(logits)
    epsilon = 1e-5
    for row in range(2):
        for col in range(3):
            plus, minus = logits.detach().clone(), logits.detach().clone()
            plus[row, col] += epsilon
            minus[row, col] -= epsilon
            numeric[row, col] = (loss(plus) - loss(minus)) / (2 * epsilon)
    error = float((analytic - numeric).abs().max())
    assert error < 1e-8
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/idr-charge-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = new_run(
        args.output,
        {
            "source_run": str(args.run.resolve()),
            "synthetic_steps": 300,
            "synthetic_seeds": [0, 1, 2],
        },
    )
    provenance = json.loads((args.run / "provenance.json").read_text())
    for filename, digest in provenance["files"].items():
        if "/src/" in filename:
            assert file_hash(Path(filename)) == digest, filename
    files = [
        Path(__file__),
        *args.run.glob("seed-*/step-*.pt"),
        *args.run.glob("seed-*/training.jsonl"),
        args.run / "config.json",
    ]
    write_json(
        output / "provenance.json", {str(p.resolve()): file_hash(p) for p in files}
    )
    write_json(output / "checkpoint_audit.json", checkpoint_audit(args.run))
    results = []
    for condition in (
        "exact_expectation",
        "sample8",
        "sample8_higher_lr",
        "sample8_no_clip",
    ):
        for seed in range(3):
            result = tiny_learning(seed, condition)
            results.append(result)
            print(
                condition,
                seed,
                "TV",
                round(result["prior_tv"], 4),
                "->",
                round(result["final_tv"], 4),
                flush=True,
            )
    write_json(output / "synthetic_learning.json", results)
    write_json(
        output / "checks.json",
        {
            "finite_difference_max_error": gradient_check(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "pretrained_backbone_executed": False,
        },
    )
    assert not torch.cuda.is_initialized()
    write_json(output / "status.json", {"status": "complete", "synthetic_only": True})


if __name__ == "__main__":
    main()
