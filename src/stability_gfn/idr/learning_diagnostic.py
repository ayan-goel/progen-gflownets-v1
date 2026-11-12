"""Capture real frozen states and diagnose charge learning without sustained fitting."""

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..guidance import GuidanceHead
from ..pipeline import project_path
from .charge import charge_reward, path_log_probabilities, rollout
from .charge_experiment import check_charge_inputs
from .data import ALPHABET


def check_source(settings, check_checkpoint=True):
    """Pin the existing run and heads; never execute a backbone during preflight."""
    source = project_path(settings["source_run"])
    if file_hash(source / "config.json") != settings["source_config_sha256"]:
        raise ValueError("Source charge configuration changed")
    config = json.loads((source / "config.json").read_text())
    references, _ = check_charge_inputs(config, check_checkpoint)
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("Need a completed source experiment")
    if json.loads((source / "references.json").read_text()) != references:
        raise ValueError("Saved and prepared references differ")
    batches = settings["batches_per_case"]
    groups = settings["gradient_group_sizes"]
    if not isinstance(batches, int) or batches < 4 or batches % 2:
        raise ValueError("Need an even number of batches, at least four")
    if not groups or groups != sorted(set(groups)) or groups[0] != 1:
        raise ValueError("Gradient group sizes must start at one and increase")
    if any(not isinstance(k, int) or k < 1 or (batches // 2) % k for k in groups):
        raise ValueError("Group sizes must divide the first half of the batches")
    if max(groups) >= batches // 2:
        raise ValueError("Keep at least two gradient groups per averaging size")
    if not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("Need distinct seeds")
    if not set(settings["seeds"]).issubset(config["seeds"]):
        raise ValueError("Diagnostic seed is absent from the pilot")
    prompts = [r for r in references if r["role"] == "prompt"]
    for seed in settings["seeds"]:
        path = source / f"seed-{seed}/step-{settings['checkpoint_step']}.pt"
        if file_hash(path) != settings["checkpoint_sha256"][str(seed)]:
            raise ValueError("Source head checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["step"] != settings["checkpoint_step"]:
            raise ValueError("Checkpoint does not match the pinned configuration")
        if saved["prompt_ids"] != [p["id"] for p in prompts]:
            raise ValueError("Checkpoint prompt order differs")
    # Existing source implementation must still represent the saved run.
    provenance = json.loads((source / "provenance.json").read_text())
    for filename, digest in provenance["files"].items():
        if "/src/" in filename and file_hash(Path(filename)) != digest:
            raise ValueError(f"Historical implementation changed: {filename}")
    return source, config, references


def load_head(snapshot, device):
    """Construct a head from a CPU snapshot without disturbing global CPU RNG."""
    state = snapshot["head"]
    feature_size = state["normalization.weight"].numel()
    context_size = state["hidden.weight"].shape[1] - feature_size
    with torch.random.fork_rng(devices=[]):
        head = GuidanceHead(
            feature_size, context_size, 20, state["hidden.weight"].shape[0]
        )
    head.load_state_dict(state)
    return head.to(device).eval(), torch.nn.Parameter(
        snapshot["log_z"].to(device).clone()
    )


def optimizer_for(head, log_z, config):
    return torch.optim.Adam(
        [
            {"params": list(head.parameters()), "lr": config["learning_rate"]},
            {"params": [log_z], "lr": config["normalizer_learning_rate"]},
        ]
    )


def initial_snapshot(final_snapshot, config, seed):
    """Reconstruct exactly the pilot's zero-output initialized head, on CPU."""
    state = final_snapshot["head"]
    feature_size = state["normalization.weight"].numel()
    context_size = state["hidden.weight"].shape[1] - feature_size
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = GuidanceHead(feature_size, context_size, 20, config["hidden_size"])
    log_z = torch.nn.Parameter(torch.zeros(len(final_snapshot["prompt_ids"])))
    return {
        "head": head.state_dict(),
        "log_z": log_z.detach(),
        "optimizer": optimizer_for(head, log_z, config).state_dict(),
        "step": 0,
        "prompt_ids": final_snapshot["prompt_ids"],
        "config": config,
    }


def load_batch(path, device):
    batch = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def batch_residual(head, log_z, batch, config):
    scores = charge_reward(
        [s[config["prefix_length"] :] for s in batch["sequences"]],
        config["reward"],
        log_z.device,
    )[0]
    return (
        log_z
        + path_log_probabilities(head, batch)
        - batch["log_prior"]
        - config["beta"] * scores
    )


def batch_gradients(head, log_z, batch, config):
    """Raw per-batch head/normalizer gradients, before any averaging or clipping."""
    residual = batch_residual(head, log_z, batch, config)
    parameters = [*head.parameters(), log_z]
    gradients = torch.autograd.grad(residual.square().mean(), parameters)
    vector = torch.cat([g.detach().flatten() for g in gradients[:-1]])
    normalizer = gradients[-1].detach()
    norm = torch.cat([vector, normalizer]).norm()
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite diagnostic gradient")
    scale = min(1.0, config["gradient_clip"] / (float(norm) + 1e-6))
    names = [name for name, _ in head.named_parameters()]
    named = dict(zip(names, gradients[:-1]))
    feature_size = head.normalization.weight.numel()
    hidden_gradient = named["hidden.weight"]
    stats = {
        "tb_mse": float(residual.detach().square().mean()),
        "head_norm": float(vector.norm()),
        "normalizer_norm": float(normalizer.norm()),
        "joint_norm": float(norm),
        "clip_multiplier": scale,
        "post_clip_head_norm": float(vector.norm()) * scale,
        "post_clip_normalizer_norm": float(normalizer.norm()) * scale,
        "hidden_feature_columns_norm": float(hidden_gradient[:, :feature_size].norm()),
        "hidden_task_columns_norm": float(
            hidden_gradient[:, feature_size : feature_size + 5].norm()
        ),
        "hidden_prompt_columns_norm": float(
            hidden_gradient[:, feature_size + 5 :].norm()
        ),
    }
    return vector.cpu(), normalizer.cpu(), stats


@torch.no_grad()
def behavior_probe(head, batch):
    """Observed token changes and fixed-feature context interventions on real states."""
    features, context, logits = batch["features"], batch["context"], batch["logits"]
    correction = head(features, context)
    prior = logits.softmax(-1)
    policy = (logits + correction).softmax(-1)
    charge = logits.new_tensor(
        [1 if a in "KR" else -1 if a in "DE" else 0 for a in ALPHABET]
    )
    changes = ((policy - prior) * charge).sum(-1)
    first, second = context.clone(), context.clone()
    first[..., 0], second[..., 0] = 0.0, 0.5
    # Counterfactuals keep features, accrued charges, and prompt identity fixed.
    magnitude = context[..., 4].abs()
    first[..., 4], second[..., 4] = magnitude, -magnitude
    first_correction, second_correction = head(features, first), head(features, second)
    difference = first_correction - second_correction
    center = difference - difference.mean(-1, keepdim=True)
    counterfactual_charge = (
        (
            (logits + first_correction).softmax(-1)
            - (logits + second_correction).softmax(-1)
        )
        * charge
    ).sum(-1)
    feature_size = features.shape[-1]
    normalized = head.normalization(features)
    feature_part = torch.nn.functional.linear(
        normalized, head.hidden.weight[:, :feature_size]
    )
    task_part = torch.nn.functional.linear(
        context[..., :5], head.hidden.weight[:, feature_size : feature_size + 5]
    )
    prompt_part = torch.nn.functional.linear(
        context[..., 5:], head.hidden.weight[:, feature_size + 5 :]
    )
    midpoint = logits.shape[1] // 2
    log_ratio = (logits + correction).log_softmax(-1) - logits.log_softmax(-1)
    return {
        "first_half_expected_charge_change": changes[:, :midpoint]
        .mean(1)
        .cpu()
        .tolist(),
        "second_half_expected_charge_change": changes[:, midpoint:]
        .mean(1)
        .cpu()
        .tolist(),
        "mean_token_kl_to_prior": float((policy * log_ratio).sum(-1).mean()),
        "counterfactual_mean_charge_contrast": float(counterfactual_charge.mean()),
        "counterfactual_max_centered_logit_contrast": float(center.abs().max()),
        "mean_hidden_input_norms": {
            "frozen_features": float(feature_part.norm(dim=-1).mean()),
            "task_context": float(task_part.norm(dim=-1).mean()),
            "prompt_identity": float(prompt_part.norm(dim=-1).mean()),
        },
    }


def cosine(left, right):
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator) if denominator > 1e-12 else None


def gradient_geometry(vectors, groups):
    """Compare disjoint proposal groups with a separate, finite reference half."""
    midpoint = len(vectors) // 2
    proposal, reference = vectors[:midpoint], vectors[midpoint:].mean(0)
    result = {"reference_mean_norm": float(reference.norm()), "groups": {}}
    for size in groups:
        means = proposal.reshape(midpoint // size, size, -1).mean(1)
        result["groups"][str(size)] = {
            "count": len(means),
            "cosines_to_reference": [cosine(row, reference) for row in means],
            "mean_squared_distance_to_reference": float(
                (means - reference).square().sum(1).mean()
            ),
            "mean_norm": float(means.norm(dim=1).mean()),
        }
    return result


def one_step_probe(
    snapshot, config, head_gradients, z_gradients, size, reference_paths, device
):
    """One disposable Adam step using inherited moments; no source head is mutated."""
    head, log_z = load_head(snapshot, device)
    optimizer = optimizer_for(head, log_z, config)
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    old_head = copy.deepcopy(head)
    old_z = log_z.detach().clone()
    mean_head = head_gradients[:size].mean(0).to(device)
    mean_z = z_gradients[:size].mean(0).to(device)
    offset = 0
    for parameter in head.parameters():
        count = parameter.numel()
        parameter.grad = (
            mean_head[offset : offset + count].reshape_as(parameter).clone()
        )
        offset += count
    log_z.grad = mean_z.clone()
    norm = torch.nn.utils.clip_grad_norm_(
        [*head.parameters(), log_z], config["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    measures = []
    with torch.no_grad():
        for path in reference_paths:
            batch = load_batch(path, device)
            old_q = path_log_probabilities(old_head, batch)
            new_q = path_log_probabilities(head, batch)
            old_residual = batch_residual(old_head, old_z, batch, config)
            delta_q, delta_z = new_q - old_q, log_z - old_z
            measures.append(
                {
                    "before_tb_mse": float(old_residual.square().mean()),
                    "after_joint_tb_mse": float(
                        (old_residual + delta_q + delta_z).square().mean()
                    ),
                    "after_head_only_tb_mse": float(
                        (old_residual + delta_q).square().mean()
                    ),
                    "after_normalizer_only_tb_mse": float(
                        (old_residual + delta_z).square().mean()
                    ),
                    "mean_absolute_path_log_probability_change": float(
                        delta_q.abs().mean()
                    ),
                }
            )
    return {
        "microbatches_averaged": size,
        "joint_gradient_norm": float(norm),
        "clip_multiplier": min(1.0, config["gradient_clip"] / (float(norm) + 1e-6)),
        "learning_rates": [g["lr"] for g in optimizer.param_groups],
        "head_update_norm": float(
            torch.cat(
                [
                    (p - q).detach().flatten()
                    for p, q in zip(head.parameters(), old_head.parameters())
                ]
            ).norm()
        ),
        "normalizer_update_norm": float((log_z - old_z).detach().norm()),
        **{key: float(np.mean([row[key] for row in measures])) for key in measures[0]},
    }


def analyze_case(case, config, settings, output, device):
    snapshot = torch.load(case / "head.pt", map_location="cpu", weights_only=True)
    head, log_z = load_head(snapshot, device)
    paths = [
        case / f"batch-{index:02d}.pt" for index in range(settings["batches_per_case"])
    ]
    head_vectors, z_vectors, batch_stats, behavior = [], [], [], []
    for index, path in enumerate(paths):
        batch = load_batch(path, device)
        with torch.no_grad():
            recomputed = path_log_probabilities(head, batch)
            max_error = float((recomputed - batch["log_policy"]).abs().max())
            if max_error > 0.002:
                raise ValueError("Captured and recomputed path likelihoods disagree")
        head_gradient, z_gradient, stats = batch_gradients(head, log_z, batch, config)
        stats.update(batch=index, maximum_path_log_probability_error=max_error)
        scores, blocks, fcr, success = charge_reward(
            [s[config["prefix_length"] :] for s in batch["sequences"]], config["reward"]
        )
        stats.update(
            scores=scores.tolist(),
            block_charges=blocks.tolist(),
            charged_fractions=fcr.tolist(),
            charge_pass=success.tolist(),
        )
        head_vectors.append(head_gradient)
        z_vectors.append(z_gradient)
        batch_stats.append(stats)
        behavior.append(behavior_probe(head, batch))
    head_vectors, z_vectors = torch.stack(head_vectors), torch.stack(z_vectors)
    torch.save(
        {
            "head": head_vectors,
            "normalizer": z_vectors,
            "parameter_shapes": {
                name: list(value.shape) for name, value in head.named_parameters()
            },
        },
        output / f"{case.name}-gradients.pt",
    )
    sizes = [settings["gradient_group_sizes"][0], settings["gradient_group_sizes"][-1]]
    probes = [
        one_step_probe(
            snapshot,
            config,
            head_vectors,
            z_vectors,
            size,
            paths[len(paths) // 2 :],
            device,
        )
        for size in dict.fromkeys(sizes)
    ]
    # All observations and trial updates must leave the captured policy unchanged.
    for name, value in head.state_dict().items():
        if not torch.equal(value.cpu(), snapshot["head"][name]):
            raise RuntimeError("Diagnostic mutated the source head")
    if not torch.equal(log_z.detach().cpu(), snapshot["log_z"]):
        raise RuntimeError("Diagnostic mutated the source normalizer")
    return {
        "prompt_ids": snapshot["prompt_ids"],
        "checkpoint_step": snapshot["step"],
        "batch_statistics": batch_stats,
        "behavior": behavior,
        "gradient_geometry": {
            "head": gradient_geometry(head_vectors, settings["gradient_group_sizes"]),
            "normalizer": gradient_geometry(
                z_vectors, settings["gradient_group_sizes"]
            ),
        },
        "one_step_probes": probes,
    }


def analyze_capture(run, output, device="cpu"):
    """Analysis can be replayed from captures without accessing ProGen weights."""
    manifest = json.loads((run / "capture.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete capture")
    for filename, digest in manifest["files"].items():
        if file_hash(run / filename) != digest:
            raise ValueError(f"Captured input changed: {filename}")
    for filename, digest in manifest["analysis_source"].items():
        if file_hash(Path(filename)) != digest:
            raise ValueError("Diagnostic implementation changed")
    settings = json.loads((run / "config.json").read_text())
    config = json.loads((run / "pilot_config.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    results = {}
    for name in manifest["cases"]:
        print(
            f"Analyzing {name}: gradients, context response and disposable updates",
            flush=True,
        )
        results[name] = analyze_case(run / name, config, settings, output, device)
    write_json(output / "metrics.json", results)
    lines = [
        "# Charge-learning diagnostic",
        "",
        "No sustained training or source-model updates. Metrics use captured real prefix states.",
        "",
        "## Gradient estimates",
        "",
        "Cosines compare proposal-batch gradients with a separate reference half. The reference is a finite estimate, not a true gradient.",
        "",
        "| Case | Averaged microbatches | Head cosine to reference (mean) | Mean head norm |",
        "|---|---:|---:|---:|",
    ]
    for name, result in results.items():
        for size, group in result["gradient_geometry"]["head"]["groups"].items():
            values = [x for x in group["cosines_to_reference"] if x is not None]
            mean = f"{np.mean(values):.4f}" if values else "undefined"
            lines.append(f"| {name} | {size} | {mean} | {group['mean_norm']:.3f} |")
    lines += [
        "",
        "## Token changes on captured states",
        "",
        "| Case | First-half charge change | Second-half charge change | Fixed-feature context contrast |",
        "|---|---:|---:|---:|",
    ]
    for name, result in results.items():
        behavior = result["behavior"]
        first = np.mean([r["first_half_expected_charge_change"] for r in behavior])
        second = np.mean([r["second_half_expected_charge_change"] for r in behavior])
        contrast = np.mean([r["counterfactual_mean_charge_contrast"] for r in behavior])
        lines.append(f"| {name} | {first:.6f} | {second:.6f} | {contrast:.6f} |")
    lines += [
        "",
        "## One-step probes on disjoint reference trajectories",
        "",
        "Each candidate starts from an identical captured head and optimizer state; gradients use the first 1 or largest requested group of proposal batches.",
        "",
        "| Case | Microbatches | Before TB MSE | After joint | Head contribution only | Normalizer contribution only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        for probe in result["one_step_probes"]:
            lines.append(
                f"| {name} | {probe['microbatches_averaged']} | {probe['before_tb_mse']:.4f} | {probe['after_joint_tb_mse']:.4f} | {probe['after_head_only_tb_mse']:.4f} | {probe['after_normalizer_only_tb_mse']:.4f} |"
            )
    lines += [
        "",
        "Trial updates are discarded. These are fixed-trajectory diagnostics, not new generation outcomes or evidence of improved biological properties. Context counterfactuals intentionally hold frozen features fixed and need not describe reachable histories. Initial/final cases use different on-policy draws. Inspect batch and prompt arrays in metrics.json; no setting is automatically selected.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return output / "report.md"


def run_learning_diagnostic(settings, output, device="cuda", backbone=None):
    source, config, references = check_source(
        settings, check_checkpoint=backbone is None
    )
    output = new_run(output, settings)
    fixture = backbone is not None
    started = time.monotonic()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prompts = [r for r in references if r["role"] == "prompt"]
        write_json(output / "pilot_config.json", config)
        write_json(output / "references.json", references)
        source_provenance = json.loads((source / "provenance.json").read_text())
        if not fixture:
            for name in ("config.json", "pytorch_model.bin"):
                path = project_path(config["checkpoint"]) / name
                if file_hash(path) != source_provenance["files"].get(
                    str(path.resolve())
                ):
                    raise ValueError("Frozen backbone files changed")
        write_json(
            output / "provenance.json",
            {
                "source_provenance": source_provenance,
                "diagnostic_settings": settings,
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "fixture": fixture,
            },
        )
        if backbone is None:
            print("Loading frozen ProGen for diagnostic capture only", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        cases = []
        for seed in settings["seeds"]:
            saved = torch.load(
                source / f"seed-{seed}/step-{settings['checkpoint_step']}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for stage_index, stage in enumerate(("initial", "final")):
                case = output / f"seed-{seed}-{stage}"
                case.mkdir()
                cases.append(case.name)
                snapshot = (
                    initial_snapshot(saved, config, seed)
                    if stage == "initial"
                    else saved
                )
                torch.save(snapshot, case / "head.pt")
                head, _ = load_head(snapshot, backbone.device)
                generator = torch.Generator(device=backbone.device).manual_seed(
                    settings["capture_seed_base"] + 100 * seed + stage_index
                )
                for index in range(settings["batches_per_case"]):
                    batch = rollout(
                        backbone,
                        head,
                        prompts,
                        list(range(len(prompts))),
                        config,
                        generator,
                        retain=True,
                    )
                    cpu_batch = {
                        key: value.cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    torch.save(cpu_batch, case / f"batch-{index:02d}.pt")
                    del batch, cpu_batch
                    print(
                        f"{case.name}: saved batch {index + 1}/{settings['batches_per_case']}",
                        flush=True,
                    )
        paths = [
            output / "config.json",
            output / "pilot_config.json",
            output / "references.json",
            output / "provenance.json",
            *output.glob("seed-*/*.pt"),
        ]
        analysis_files = [
            Path(__file__),
            Path(__file__).with_name("charge.py"),
            Path(__file__).with_name("data.py"),
            Path(__file__).parent.parent / "guidance.py",
        ]
        write_json(
            output / "capture.json",
            {
                "complete": True,
                "cases": cases,
                "files": {str(p.relative_to(output)): file_hash(p) for p in paths},
                "analysis_source": {
                    str(p.resolve()): file_hash(p) for p in analysis_files
                },
                "seconds_before_analysis": time.monotonic() - started,
                "backbone_work": backbone.work,
                "trajectories": len(cases)
                * settings["batches_per_case"]
                * len(prompts),
            },
        )
        analyze_capture(output, output, device)
        write_json(
            output / "status.json",
            {
                "status": "complete",
                "fixture": fixture,
                "seconds": time.monotonic() - started,
            },
        )
    except Exception as error:
        write_json(
            output / "status.json",
            {"status": "failed", "fixture": fixture, "error": repr(error)},
        )
        raise
    return output / "report.md"


def replay_learning_diagnostic(run, output):
    """CPU-only replay; a failed analysis never forces new backbone sampling."""
    run = Path(run)
    output = new_run(output, {"source_capture": str(run.resolve())})
    try:
        report = analyze_capture(run, output, "cpu")
        write_json(output / "status.json", {"status": "complete"})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    return report
