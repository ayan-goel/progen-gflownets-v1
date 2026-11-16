"""Inspect optimization on the cached finite IDR task without pretrained execution."""

import argparse
import copy
import json
from pathlib import Path

import torch
from diagnose_idr_exact_cpu import FiniteBankTree, distribution_metrics

from stability_gfn.artifacts import file_hash, new_run, write_json
from stability_gfn.idr.acceptance_experiment import (
    verify_implementation,
    verify_manifest,
)
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.branch_rate import branch_optimizer
from stability_gfn.idr.context_experiment import DirectContextHead
from stability_gfn.idr.continuous_reward import continuous_reward


def load_task(run):
    """Read training-bank states only, validating historical artifacts and code."""
    verify_manifest(run)
    verify_implementation(json.loads((run / "provenance.json").read_text()))
    bank = json.loads((run / "bank.json").read_text())
    cache = torch.load(run / "bank_cache.pt", map_location="cpu", weights_only=True)
    saved = torch.load(run / "initial.pt", map_location="cpu", weights_only=True)
    config = json.loads((run / "pilot_config.json").read_text())
    reward = AcceptanceRewardConfig(**json.loads((run / "reward.json").read_text()))
    head = DirectContextHead(
        cache["features"].shape[-1],
        cache["context"].shape[-1],
        20,
        config["hidden_size"],
    )
    head.load_state_dict(saved["head"])
    trees, offsets = [], []
    for prompt in sorted({r["prompt_index"] for r in bank}):
        indices = [
            i
            for i, row in enumerate(bank)
            if row["prompt_index"] == prompt and row["split"] == "train"
        ]
        rows = [bank[i] for i in indices]
        tree = FiniteBankTree(rows, {k: v[indices] for k, v in cache.items()})
        values = continuous_reward(
            [r["tail"] for r in rows],
            [r["isolated_tail_v3_scores"] for r in rows],
            reward,
            2.5,
        )
        assert torch.equal(values["joint_pass"], tree.passed)
        tree.log_reward = tree.log_prior + values["log_factor"]
        tree.target = tree.log_reward.softmax(0)
        initial_log_q = tree.log_probabilities(head).detach().double()
        delta = tree.log_reward - initial_log_q
        offsets.append((delta[tree.passed].mean() + delta[~tree.passed].mean()) / 2)
        tree.fixed_weights = (
            0.5 * initial_log_q.exp()
            + 0.25 * tree.passed.double() / tree.passed.sum()
            + 0.25 * (~tree.passed).double() / (~tree.passed).sum()
        )
        trees.append(tree)
    return trees, head, torch.nn.Parameter(torch.stack(offsets).float()), config


def objective(trees, head, normalizer, kind="fixed_tb", parity=0):
    """The fixed weighting makes a scalar differentiable objective for line probes."""
    losses = []
    for prompt, tree in enumerate(trees):
        log_q = tree.log_probabilities(head).double()
        if kind == "teacher":
            loss = -(tree.target * log_q).sum()
        else:
            residual = normalizer[prompt] + log_q - tree.log_reward
            if kind == "fixed_tb":
                weights = tree.fixed_weights
            elif kind == "alternating_tb":
                stratum = tree.passed == ((prompt + parity) % 2 == 0)
                weights = (
                    0.5 * log_q.detach().exp() + 0.5 * stratum.double() / stratum.sum()
                )
            else:
                raise ValueError(kind)
            loss = (weights * residual.square()).sum()
        losses.append(loss)
    return torch.stack(losses).mean()


def metrics(trees, head):
    with torch.no_grad():
        rows = {
            tree.rows[0]["prompt_id"]: distribution_metrics(
                tree.log_probabilities(head), tree.target, tree.passed
            )
            for tree in trees
        }
    return {
        "by_prompt": rows,
        "mean_tv": sum(r["tv"] for r in rows.values()) / len(rows),
        "mean_joint_success": sum(r["joint_success"] for r in rows.values())
        / len(rows),
    }


def concentration(values):
    """Descriptive geometry only: energy shared across states, not causal evidence."""
    values = values.detach().double()
    normed = values / values.norm(dim=1, keepdim=True)
    cosine = normed @ normed.T
    off_diagonal = cosine[~torch.eye(len(values), dtype=torch.bool)]
    singular = torch.linalg.svdvals(values)
    return {
        "mean_component_energy_fraction": float(
            len(values) * values.mean(0).square().sum() / values.square().sum()
        ),
        "top_singular_energy_fraction": float(
            singular[0].square() / singular.square().sum()
        ),
        "median_pairwise_cosine": float(off_diagonal.median()),
    }


def audit(trees, initial, initial_normalizer, config):
    """Gradients and a single-step line probe; original weights never change."""
    head = copy.deepcopy(initial)
    normalizer = torch.nn.Parameter(initial_normalizer.detach().clone())
    parameters = list(head.parameters()) + [normalizer]
    names = [name for name, _ in head.named_parameters()] + ["normalizer"]
    geometry = {}
    with torch.no_grad():
        combined = torch.cat(
            [
                torch.cat([head.normalization(t.features), t.context], dim=1)
                for t in trees
            ]
        )
        hidden = torch.nn.functional.silu(head.hidden(combined))
        geometry["normalized_feature_and_context"] = concentration(combined)
        geometry["hidden_activations"] = concentration(hidden)
    gradients = {}
    for name, kind, parity in [
        ("teacher", "teacher", 0),
        ("fixed_tb", "fixed_tb", 0),
        ("even_tb", "alternating_tb", 0),
        ("odd_tb", "alternating_tb", 1),
    ]:
        grads = torch.autograd.grad(
            objective(trees, head, normalizer, kind, parity),
            parameters,
            allow_unused=True,
        )
        gradients[name] = [
            torch.zeros_like(p) if g is None else g.detach()
            for p, g in zip(parameters, grads)
        ]
    flat = {
        name: torch.cat([g.flatten() for g in groups])
        for name, groups in gradients.items()
    }
    cosine = torch.nn.functional.cosine_similarity(
        flat["even_tb"], flat["odd_tb"], dim=0
    )
    gradient_summary = {
        name: {key: float(g.norm()) for key, g in zip(names, groups)}
        for name, groups in gradients.items()
    }
    optimizer = branch_optimizer(head, normalizer, config, 10)
    optimizer.zero_grad(set_to_none=True)
    objective(trees, head, normalizer).backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
    before = [p.detach().clone() for p in parameters]
    optimizer.step()
    updates = [p.detach() - old for p, old in zip(parameters, before)]
    line = []
    for scale in [0, 0.25, 1, 4, 16]:
        with torch.no_grad():
            for parameter, start, update in zip(parameters, before, updates):
                parameter.copy_(start + scale * update)
            result = metrics(trees, head)
            result.update(
                scale=scale, fixed_tb_loss=float(objective(trees, head, normalizer))
            )
            line.append(result)
    return {
        "geometry": geometry,
        "gradient_norms": gradient_summary,
        "even_odd_gradient_cosine": float(cosine),
        "initial_global_gradient_norm": float(norm),
        "adam_update_norms": {
            name: float(change.norm()) for name, change in zip(names, updates)
        },
        "line_probe": line,
        "scope": "Post-hoc initial-state measurements; no production or GPU training. Probe scales are not selected training rates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/idr-continuous-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    trees, head, normalizer, config = load_task(args.run)
    output = new_run(
        args.output, {"source": str(args.run.resolve()), "kind": "update_audit"}
    )
    result = audit(trees, head, normalizer, config)
    write_json(output / "audit.json", result)
    verify_manifest(args.run)
    assert not torch.cuda.is_initialized()
    write_json(
        output / "provenance.json",
        {
            "source_manifest_sha256": file_hash(args.run / "generation.json"),
            "script_sha256": file_hash(Path(__file__)),
            "tree_script_sha256": file_hash(
                Path(__file__).with_name("diagnose_idr_exact_cpu.py")
            ),
            "torch": torch.__version__,
            "cuda_initialized": False,
        },
    )
    write_json(output / "status.json", {"status": "complete"})
    print(json.dumps({k: v for k, v in result.items() if k != "line_probe"}, indent=2))
    print(
        "line",
        [(r["scale"], r["fixed_tb_loss"], r["mean_tv"]) for r in result["line_probe"]],
    )


if __name__ == "__main__":
    main()
