"""Privileged exact-target fitting and read-only audits of saved finite tasks."""

import json
import math
import shutil
from pathlib import Path

import torch

from .artifacts import file_hash, new_run, write_json
from .evaluation import load_head, load_task
from .metrics import distribution_metrics


def exact_conditionals(task, beta):
    """Sum terminal target mass backwards through the tree, in float64 on CPU."""
    log_weights = task.base_log_probabilities.double() + beta * task.scores.double()
    node_log_mass = torch.empty(len(task.features), dtype=torch.float64)
    edge_log_mass = torch.full(task.logits.shape, -torch.inf, dtype=torch.float64)
    for node in reversed(range(len(task.features))):
        for action in torch.where(task.legal[node])[0].tolist():
            leaf = task.leaves[node, action].item()
            child = task.children[node, action].item()
            edge_log_mass[node, action] = (
                log_weights[leaf] if leaf >= 0 else node_log_mass[child]
            )
        node_log_mass[node] = edge_log_mass[node].logsumexp(0)
    return (
        edge_log_mass - node_log_mass[:, None],
        (node_log_mass - log_weights.logsumexp(0)).exp(),
    )


@torch.no_grad()
def conditional_errors(task, head, target_log_conditionals, visitation):
    """Separate uniform-prefix fit error from target-weighted whole-sequence KL.

    Chain rule: KL(target || policy) equals the sum of conditional KLs weighted
    by the target probability of visiting each prefix. Uniform-prefix CE is the
    training objective; its irreducible floor is the conditional target entropy.
    """
    nodes = torch.arange(len(task.features))
    log_q = task.conditionals(head, nodes).double()
    target = target_log_conditionals.exp()
    entropy_terms = torch.zeros_like(target)
    kl_terms = torch.zeros_like(target)
    legal = task.legal
    entropy_terms[legal] = -target[legal] * target_log_conditionals[legal]
    kl_terms[legal] = target[legal] * (target_log_conditionals[legal] - log_q[legal])
    conditional_kl = kl_terms.sum(-1)
    entropy = entropy_terms.sum(-1)
    branching = task.legal.sum(-1) > 1
    if not branching.any():
        raise ValueError("No nontrivial choices to diagnose")
    rows = []
    for node, prefix in enumerate(task.tree.prefixes):
        rows.append(
            {
                "prefix": prefix,
                "depth": len(prefix),
                "target_visitation": visitation[node].item(),
                "conditional_kl_target_to_q": conditional_kl[node].item(),
                "sequence_kl_contribution": (
                    visitation[node] * conditional_kl[node]
                ).item(),
            }
        )
    by_depth = []
    for depth in sorted({row["depth"] for row in rows}):
        selected = [row for row in rows if row["depth"] == depth]
        by_depth.append(
            {
                "depth": depth,
                "prefix_count": len(selected),
                "sequence_kl_contribution": sum(
                    row["sequence_kl_contribution"] for row in selected
                ),
            }
        )
    return {
        "conditional_entropy_floor": entropy[branching].mean().item(),
        "excess_conditional_cross_entropy": conditional_kl[branching].mean().item(),
        "cross_entropy": (entropy + conditional_kl)[branching].mean().item(),
        "chain_rule_kl_target_to_q": (visitation * conditional_kl).sum().item(),
        "by_depth": by_depth,
        "prefixes": rows,
    }


def _record_inputs(output, run, fit=None):
    """Save identities and current diagnostic source without altering old runs."""
    paths = [run / "config.json", run / "task.pt"]
    if fit is not None:
        paths.extend([fit / "config.json", fit / "head.pt", fit / "diagnostics.json"])
    write_json(
        output / "provenance.json",
        {
            "input_sha256": {str(path.resolve()): file_hash(path) for path in paths},
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "cpu_threads": torch.get_num_threads(),
        },
    )
    shutil.copytree(
        Path(__file__).parent,
        output / "source/stability_gfn",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def fit_exact_conditionals(run, output, steps=2000, learning_rate=0.001, seed=0):
    """Fit the unchanged head/objective; report convergence using exact labels.

    This user-run experiment is privileged, not TB or a fair baseline. It loads
    only the saved finite CPU cache. Failure to fit does not prove incapacity:
    finite optimization, input conditioning and implementation are alternatives.
    """
    if steps <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Steps and learning rate must be finite and positive")
    run = Path(run)
    original_config = json.loads((run / "config.json").read_text())
    settings = {
        "kind": "privileged_exact_conditional_fit",
        "run": str(run.resolve()),
        "steps": steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "smoke": original_config["smoke"],
        "hidden_size": original_config["training"]["hidden_size"],
        "beta": original_config["training"]["beta"],
        "objective": "uniform_branching_prefix_cross_entropy",
        "gradient_clip": 1.0,
        "log_every": 100,
        "cpu_threads": 1,
        "checkpoint_selection": "final; best observed TV is diagnostic only",
    }
    output = new_run(output, settings)
    try:
        return _fit_exact_conditionals(run, output, settings)
    except BaseException as error:
        write_json(
            output / "status.json",
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "error": str(error),
                "privileged": True,
                "smoke": settings["smoke"],
            },
        )
        raise


def _fit_exact_conditionals(run, output, settings):
    """Keep the fitting loop identical while logging post-update distribution error."""
    torch.set_num_threads(settings["cpu_threads"])
    _record_inputs(output, run)
    task, _ = load_task(run)
    beta = settings["beta"]
    target_log_conditionals, visitation = exact_conditionals(task, beta)
    # Match the original float32 targets and uniform-prefix loss exactly.
    target = target_log_conditionals.exp().float()
    nodes = torch.where(task.legal.sum(-1) > 1)[0]
    if len(nodes) == 0:
        raise ValueError("No nontrivial choices to diagnose")
    torch.manual_seed(settings["seed"])
    head = task.head(settings["hidden_size"])
    optimizer = torch.optim.Adam(head.parameters(), lr=settings["learning_rate"])
    history = []
    best = None
    with (output / "progress.jsonl").open("w") as progress:
        for step in range(settings["steps"] + 1):
            gradient_norm = None
            if step > 0:
                log_q = task.conditionals(head, nodes)
                legal = task.legal[nodes]
                loss = -(target[nodes][legal] * log_q[legal]).sum() / len(nodes)
                optimizer.zero_grad()
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0, error_if_nonfinite=True
                ).item()
                optimizer.step()
            if step in (0, 1, settings["steps"]) or step % 100 == 0:
                errors = conditional_errors(
                    task, head, target_log_conditionals, visitation
                )
                metrics = distribution_metrics(
                    task.all_log_probabilities(head),
                    task.base_log_probabilities,
                    task.scores,
                    beta,
                )
                row = {
                    "step": step,
                    **{
                        key: value
                        for key, value in errors.items()
                        if key not in ("prefixes", "by_depth")
                    },
                    **metrics,
                    "gradient_norm_before_clip": gradient_norm,
                }
                history.append(row)
                progress.write(json.dumps(row, allow_nan=False) + "\n")
                progress.flush()
                if best is None or metrics["total_variation"] < best["total_variation"]:
                    best = {"step": step, "total_variation": metrics["total_variation"]}
    torch.save(
        {"state_dict": head.state_dict(), "hidden_size": settings["hidden_size"]},
        output / "head.pt",
    )
    write_json(
        output / "diagnostics.json",
        {
            "privileged": True,
            "history": history,
            "best_observed_tv": best,
            "conditional_fit": errors,
            **metrics,
        },
    )
    write_json(
        output / "status.json",
        {"status": "complete", "privileged": True, "smoke": settings["smoke"]},
    )
    return metrics


def inspect_saved_fit(run, fit, output):
    """Audit cached exact-task results without optimizer steps or backbone calls."""
    run, fit = Path(run), Path(fit)
    config = json.loads((run / "config.json").read_text())
    fit_config = json.loads((fit / "config.json").read_text())
    if config["experiment"] != "exact":
        raise ValueError("This audit is for the exact mechanism benchmark only")
    if Path(fit_config["run"]).resolve() != run.resolve():
        raise ValueError("Fit was created for a different run")
    output = new_run(
        output,
        {
            "kind": "read_only_fit_audit",
            "run": str(run.resolve()),
            "fit": str(fit.resolve()),
        },
    )
    try:
        torch.set_num_threads(1)
        _record_inputs(output, run, fit)
        task, _ = load_task(run)
        head, _ = load_head(task, fit / "head.pt")
        beta = config["training"]["beta"]
        target_log_conditionals, visitation = exact_conditionals(task, beta)
        log_target = (
            task.base_log_probabilities.double() + beta * task.scores.double()
        ).log_softmax(0)
        # Independent of backward DP: add each leaf's mass to its visited edges.
        edge_mass = torch.zeros_like(task.logits, dtype=torch.float64)
        for leaf, mass in enumerate(log_target.exp()):
            edge_mass[task.paths[leaf], task.actions[leaf]] += mass
        independent_conditionals = (
            edge_mass.log() - edge_mass.sum(-1, keepdim=True).log()
        )
        oracle_log_q = target_log_conditionals[task.paths, task.actions].sum(-1)
        errors = conditional_errors(task, head, target_log_conditionals, visitation)
        metrics = distribution_metrics(
            task.all_log_probabilities(head),
            task.base_log_probabilities,
            task.scores,
            beta,
        )
        # Inspect the initial LayerNorm representation, without learned affine terms.
        with torch.no_grad():
            normalized = torch.nn.functional.layer_norm(
                task.features, (task.features.shape[1],), eps=head.normalization.eps
            ).double()
        pairs = []
        # Limit pair comparisons to the ten prefixes contributing most sequence KL.
        worst = sorted(
            errors["prefixes"], key=lambda row: -row["sequence_kl_contribution"]
        )[:10]
        for row in worst:
            node = task.tree.prefixes.index(row["prefix"])
            peers = [
                index
                for index, prefix in enumerate(task.tree.prefixes)
                if len(prefix) == row["depth"] and index != node
            ]
            if not peers:
                continue
            distance = (normalized[peers] - normalized[node]).square().mean(-1).sqrt()
            nearest = peers[distance.argmin().item()]
            # Compare required residuals, not just targets: ProGen already supplies
            # different logits. Remove softmax's arbitrary per-prefix constant.
            common = task.legal[node] & task.legal[nearest]
            correction_difference = None
            if common.sum() >= 2:
                left = (
                    target_log_conditionals[node, common]
                    - task.base_conditionals[node, common].double()
                )
                right = (
                    target_log_conditionals[nearest, common]
                    - task.base_conditionals[nearest, common].double()
                )
                centered_difference = (left - left.mean()) - (right - right.mean())
                correction_difference = (
                    centered_difference.square().mean().sqrt().item()
                )
            pairs.append(
                {
                    "prefix": row["prefix"],
                    "nearest_same_depth_prefix": task.tree.prefixes[nearest],
                    "initial_normalized_feature_rms_distance": distance.min().item(),
                    "context_l2_distance": torch.linalg.vector_norm(
                        task.context[node] - task.context[nearest]
                    ).item(),
                    "required_centered_correction_rms_difference": correction_difference,
                    "target_conditional_tv": (
                        0.5
                        * (
                            target_log_conditionals[node].exp()
                            - target_log_conditionals[nearest].exp()
                        )
                        .abs()
                        .sum()
                    ).item(),
                }
            )
        saved = json.loads((fit / "diagnostics.json").read_text())
        report = {
            "training_performed": False,
            "backbone_loaded": False,
            "device": "cpu",
            "prefix_count": len(task.features),
            "unique_raw_feature_rows": len(torch.unique(task.features, dim=0)),
            "unique_initial_normalized_rows": len(torch.unique(normalized, dim=0)),
            "oracle_reconstruction_tv": (
                0.5 * (oracle_log_q.exp() - log_target.exp()).abs().sum()
            ).item(),
            "independent_oracle_max_log_conditional_difference": (
                independent_conditionals[task.legal]
                - target_log_conditionals[task.legal]
            )
            .abs()
            .max()
            .item(),
            "chain_rule_kl_difference": abs(
                errors["chain_rule_kl_target_to_q"] - metrics["kl_target_to_q"]
            ),
            "saved_tv_reproduction_difference": abs(
                saved["total_variation"] - metrics["total_variation"]
            ),
            "metrics": metrics,
            "conditional_fit": errors,
            "worst_prefix_nearest_neighbors": pairs,
        }
        write_json(output / "audit.json", report)
        lines = [
            "# Saved exact-fit audit",
            "",
            "Read-only CPU analysis; no training or backbone execution. This is privileged synthetic-task evidence, not biological validation.",
            "",
            f"- Final total variation: {metrics['total_variation']:.9f} (required <0.05).",
            f"- Exact-oracle reconstruction TV: {report['oracle_reconstruction_tv']:.3g}.",
            f"- Independent oracle maximum log-probability difference: {report['independent_oracle_max_log_conditional_difference']:.3g}.",
            f"- Uniform-prefix cross-entropy: {errors['cross_entropy']:.9f}; entropy floor: {errors['conditional_entropy_floor']:.9f}.",
            f"- Excess uniform-prefix cross-entropy: {errors['excess_conditional_cross_entropy']:.9f}.",
            f"- Whole-sequence KL(target || policy): {metrics['kl_target_to_q']:.9f}.",
            f"- Unique raw / initially normalized feature rows: {report['unique_raw_feature_rows']} / {report['unique_initial_normalized_rows']} of {report['prefix_count']}.",
            "",
            "| Prefix length | Prefixes | Contribution to sequence KL |",
            "|---|---:|---:|",
        ]
        lines.extend(
            f"| {row['depth']} | {row['prefix_count']} | {row['sequence_kl_contribution']:.9f} |"
            for row in errors["by_depth"]
        )
        lines.extend(
            [
                "",
                "The uniform-prefix objective has the correct optimum, but weights prefixes differently from sequence KL. A small loss excess need not imply passing TV. Distinct feature rows rule out exact collisions in this cache; they do not establish easy optimization or generalization. Nearby features requiring different corrections motivate further checks, not a proven cause.",
                "",
                "See audit.json for all prefix errors, nearest-neighbor measurements and reproduction checks; provenance.json records immutable input hashes.",
            ]
        )
        (output / "report.md").write_text("\n".join(lines) + "\n")
        write_json(
            output / "status.json", {"status": "complete", "training_performed": False}
        )
        return report
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
