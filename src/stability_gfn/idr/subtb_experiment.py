"""Matched fresh TB/SubTB charge fits and recoverable CPU evaluation.

Historical modules remain unchanged to preserve their run provenance. Both arms
use the direct-context token head, a tenfold task-branch rate, identical budgets
and shared fresh prior controls. Only SubTB adds an auxiliary flow network.
"""

import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..pipeline import project_path
from . import subtb
from .baseline import scorer_files
from .branch_rate import (
    branch_optimizer,
    check_rate_inputs,
    check_shared_controls,
    set_rates,
)
from .charge import rollout, update_head
from .charge_experiment import evaluate_charge, generate_records, work_delta
from .context_experiment import make_context_head
from .learning_diagnostic import behavior_probe
from .scoring import predict_disorder

METHODS = ("tb", "subtb")


def check_subtb_inputs(settings, check_checkpoint=True):
    """Validate pinned historical protocol and declared overrides without execution."""
    source, config, references, provenance = check_rate_inputs(
        settings, check_checkpoint
    )
    for key in ("training_steps", "evaluation_samples_per_prompt"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key in ("subtb_lambda", "flow_learning_rate"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Invalid {key}")
    config.update(
        steps=settings["training_steps"],
        samples_per_prompt=settings["evaluation_samples_per_prompt"],
    )
    return source, config, references, provenance


def train_method(backbone, prompts, config, settings, seed, method, output):
    """Fresh matched policy initialization and on-policy data for a fixed budget."""
    if method not in METHODS:
        raise ValueError(f"Unknown training method: {method}")
    head = make_context_head(backbone, len(prompts), config, seed, "direct_context")
    log_z = torch.nn.Parameter(torch.zeros(len(prompts), device=backbone.device))
    flow = (
        subtb.make_flow_head(backbone, len(prompts), config, seed)
        if method == "subtb"
        else None
    )
    multiplier = settings["branch_rate_multiplier"]
    optimizer = branch_optimizer(head, log_z, config, multiplier)
    if flow is not None:
        optimizer.add_param_group(
            {
                "params": list(flow.parameters()),
                "lr": settings["flow_learning_rate"],
                "name": "flow",
            }
        )
    # Fresh streams; same starting draws in both methods, trajectories diverge as
    # policies learn. Neither evaluation nor auxiliary initialization consumes this RNG.
    generator = torch.Generator(device=backbone.device).manual_seed(30000 + seed)
    indices = list(range(len(prompts)))
    started, before = time.monotonic(), dict(backbone.work)
    with (output / "training.jsonl").open("w") as log:
        for step in range(1, config["steps"] + 1):
            set_rates(optimizer, config, multiplier, step)
            if flow is not None:
                factor = optimizer.param_groups[0]["lr"] / config["learning_rate"]
                optimizer.param_groups[3]["lr"] = (
                    settings["flow_learning_rate"] * factor
                )
            batch = rollout(
                backbone, head, prompts, indices, config, generator, retain=True
            )
            if flow is None:
                stats = update_head(head, log_z, optimizer, batch, indices, config)
                stats["loss"] = stats["tb_loss"]
            else:
                stats = subtb.update_subtb(
                    head,
                    flow,
                    log_z,
                    optimizer,
                    batch,
                    indices,
                    config,
                    settings["subtb_lambda"],
                )
            stats.update(
                step=step,
                method=method,
                rates={group["name"]: group["lr"] for group in optimizer.param_groups},
                branch_weight_norm=float(head.task_weight.detach().norm()),
                log_z=log_z.detach().cpu().tolist(),
                seconds=time.monotonic() - started,
            )
            log.write(json.dumps(stats, allow_nan=False) + "\n")
            log.flush()
            del batch
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                print(
                    f"{method} seed {seed} step {step}/{config['steps']}: "
                    f"objective {stats['loss']:.4g}, full TB {stats['tb_loss']:.4g}, "
                    f"charge {stats['charge_success']:.1%}",
                    flush=True,
                )
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                torch.save(
                    {
                        "method": method,
                        "architecture": "direct_context",
                        "head": head.state_dict(),
                        "flow": None if flow is None else flow.state_dict(),
                        "log_z": log_z.detach(),
                        "optimizer": optimizer.state_dict(),
                        "generator_state": generator.get_state(),
                        "step": step,
                        "prompt_ids": [r["id"] for r in prompts],
                        "config": config,
                        "settings": settings,
                    },
                    output / f"step-{step}.pt",
                )
    return (
        head,
        log_z,
        {
            "seconds": time.monotonic() - started,
            "trajectories": config["steps"] * len(prompts),
            "backbone_work": work_delta(backbone, before),
            "policy_parameters": sum(p.numel() for p in head.parameters()),
            "flow_parameters": 0
            if flow is None
            else sum(p.numel() for p in flow.parameters()),
        },
    )


@torch.no_grad()
def sample_method(backbone, head, log_z, prompts, config, seed, method):
    """Raw fresh final-policy draws; probes use the same states, no extra rollouts."""
    generator = torch.Generator(device=backbone.device).manual_seed(50000 + seed)
    records, probes = [], []
    for repeat in range(config["samples_per_prompt"]):
        batch = rollout(
            backbone,
            head,
            prompts,
            list(range(len(prompts))),
            config,
            generator,
            retain=True,
        )
        probes.append(behavior_probe(head, batch))
        for index, sequence in enumerate(batch["sequences"]):
            records.append(
                {
                    "id": f"{method}-{seed}-{index}-{repeat}",
                    "arm": "gfn",
                    "method": method,
                    "seed": seed,
                    "prompt_id": prompts[index]["id"],
                    "prefix": sequence[: config["prefix_length"]],
                    "sequence": sequence,
                    "evaluation_start": config["prefix_length"],
                    "log_prior": float(batch["log_prior"][index]),
                    "log_policy": float(batch["log_policy"][index]),
                    "log_z": float(log_z[index]),
                    "mean_canonical_probability_mass": float(
                        batch["canonical_mass"][index]
                    ),
                }
            )
        if (repeat + 1) % 16 == 0 or repeat + 1 == config["samples_per_prompt"]:
            print(
                f"{method} seed {seed} evaluation: {repeat + 1}/{config['samples_per_prompt']} per prompt",
                flush=True,
            )
    return records, probes


def evaluate_subtb(run, output, predictor=predict_disorder):
    """Identical charge/disorder criteria; loss magnitudes are not method rankings."""
    config = json.loads((run / "pilot_config.json").read_text())
    references = json.loads((run / "references.json").read_text())
    samples = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    costs = json.loads((run / "costs.json").read_text())
    prior = [r for r in samples if r["arm"] == "prior_pool"]
    results = {}
    for method in METHODS:
        destination = output / method
        destination.mkdir(exist_ok=True)
        records = prior + [r for r in samples if r.get("method") == method]
        results[method] = evaluate_charge(
            records, references, config, destination, costs[method], predictor
        )
    check_shared_controls(results["tb"], results["subtb"])
    write_json(output / "comparison.json", results)
    # Training windows describe different on-policy samples. They are diagnostic
    # curves, not an off-policy validation set or checkpoint selection rule.
    curves = {}
    for method in METHODS:
        for seed in config["seeds"]:
            path = run / method / f"seed-{seed}" / "training.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            windows = []
            for start in range(0, len(rows), 100):
                window = rows[start : start + 100]
                fields = (
                    "loss",
                    "tb_loss",
                    "mean_score",
                    "charge_success",
                    "gradient_norm_before_clip",
                )
                windows.append(
                    {
                        "first_step": window[0]["step"],
                        "last_step": window[-1]["step"],
                        **{
                            field: sum(r[field] for r in window) / len(window)
                            for field in fields
                        },
                    }
                )
            curves[f"{method}/seed-{seed}"] = windows
    write_json(output / "training_curves.json", curves)
    lines = [
        "# Matched TB/SubTB charge-guidance comparison",
        "",
        "Same frozen prior, charge target, direct-context token head and training budget. Final checkpoints only.",
        "",
        "| Method | Charge pass | Joint charge / predicted disorder | Isolated disorder | Unique tails | Mean score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in [("prior", results["tb"]["arms"]["prior"])] + [
        (m, results[m]["arms"]["gfn"]) for m in METHODS
    ]:
        lines.append(
            f"| {name} | {row['charge_success_rate']:.1%} | {row['joint_charge_and_isolated_disorder_rate']:.1%} | "
            f"{row['v3_isolated_tail_success_rate']:.1%} | {row['unique_tail_fraction']:.1%} | {row['mean_score']:.4f} |"
        )
    lines += [
        "",
        "## Paired seeds",
        "",
        "| Seed | TB charge | SubTB charge | TB joint | SubTB joint |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed in config["seeds"]:
        a, b = [results[m]["by_seed"][f"gfn/seed-{seed}"] for m in METHODS]
        lines.append(
            f"| {seed} | {a['charge_success_rate']:.1%} | {b['charge_success_rate']:.1%} | "
            f"{a['joint_charge_and_isolated_disorder_rate']:.1%} | {b['joint_charge_and_isolated_disorder_rate']:.1%} |"
        )
    lines += [
        "",
        "Read each method/report.md for per-prompt results, selection/editing controls and costs.",
        "training_curves.json summarizes 100-update windows; raw logs and checkpoints are retained.",
        "TB and SubTB objective magnitudes differ. Compare generated outcomes and common full-trajectory residuals, not raw objective values.",
        "Shared prior generation costs are attributed once to TB and marked reused for SubTB; both methods use those same controls.",
        "This is a development panel. Disorder is predicted, and target-distribution accuracy is not exactly enumerable here.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return output / "report.md"


def run_subtb(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    """Run six fresh fits; finish with automatic CPU scoring and immutable manifests."""
    source, config, references, provenance = check_subtb_inputs(
        settings, backbone is None
    )
    output = new_run(output, settings)
    started, fixture = time.monotonic(), backbone is not None
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write_json(output / "pilot_config.json", config)
        write_json(output / "references.json", references)
        write_json(
            output / "source_config.json",
            json.loads((source / "pilot_config.json").read_text()),
        )
        write_json(
            output / "source_generation.json",
            json.loads((source / "generation.json").read_text()),
        )
        if not fixture:
            for name in ("config.json", "pytorch_model.bin"):
                path = project_path(config["checkpoint"]) / name
                if file_hash(path) != provenance["files"].get(str(path.resolve())):
                    raise ValueError("Pretrained model differs from pinned control")
        provenance = dict(
            provenance,
            files=dict(provenance["files"]),
            device=str(device),
            fixture=fixture,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        # Include every local helper imported by this experiment as well as the
        # historical model/scorer provenance. Do not mutate any completed run.
        for path in Path(__file__).parent.glob("*.py"):
            provenance["files"][str(path.resolve())] = file_hash(path)
        write_json(output / "provenance.json", provenance)
        if backbone is None:
            print("Loading frozen ProGen for matched TB/SubTB training", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        prompts = [r for r in references if r["role"] == "prompt"]
        costs = {method: {} for method in METHODS}
        behavior = {}
        for method in METHODS:
            (output / method).mkdir()
        with (output / "samples.jsonl").open("w") as handle:
            for seed in config["seeds"]:
                stage_start, before = time.monotonic(), dict(backbone.work)
                prior = generate_records(
                    backbone,
                    None,
                    prompts,
                    config,
                    seed,
                    config["samples_per_prompt"] * config["selection_factor"],
                    "prior_pool",
                    torch.Generator(device=backbone.device).manual_seed(40000 + seed),
                )
                prior_cost = {
                    "seconds": time.monotonic() - stage_start,
                    "trajectories": len(prior),
                    "backbone_work": work_delta(backbone, before),
                }
                for row in prior:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                for method in METHODS:
                    seed_dir = output / method / f"seed-{seed}"
                    seed_dir.mkdir()
                    head, log_z, training = train_method(
                        backbone, prompts, config, settings, seed, method, seed_dir
                    )
                    stage_start, before = time.monotonic(), dict(backbone.work)
                    rows, probes = sample_method(
                        backbone, head, log_z, prompts, config, seed, method
                    )
                    behavior[f"{method}/seed-{seed}"] = probes
                    costs[method][str(seed)] = {
                        "training": training,
                        "gfn": {
                            "seconds": time.monotonic() - stage_start,
                            "trajectories": len(rows),
                            "backbone_work": work_delta(backbone, before),
                        },
                        "prior_pool": prior_cost
                        if method == "tb"
                        else {
                            "seconds": 0.0,
                            "trajectories": 0,
                            "reused_trajectories": len(prior),
                        },
                    }
                    for row in rows:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    write_json(output / "costs.json", costs)
                    write_json(output / "behavior.json", behavior)
                    del head, log_z
        captured = [
            "config.json",
            "pilot_config.json",
            "references.json",
            "source_config.json",
            "source_generation.json",
            "samples.jsonl",
            "behavior.json",
            "costs.json",
            "provenance.json",
        ]
        captured += [
            str(path.relative_to(output))
            for method in METHODS
            for path in (output / method).rglob("training.jsonl")
        ]
        captured += [
            str(path.relative_to(output))
            for method in METHODS
            for path in (output / method).rglob("*.pt")
        ]
        write_json(
            output / "generation.json",
            {
                "complete": True,
                "count": len(prompts)
                * len(config["seeds"])
                * config["samples_per_prompt"]
                * (config["selection_factor"] + len(METHODS)),
                "files": {name: file_hash(output / name) for name in captured},
                "seconds_before_evaluation": time.monotonic() - started,
            },
        )
        evaluate_subtb(output, output, predictor)
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


def reevaluate_subtb(run, output, predictor=predict_disorder):
    """Replay after generation, without old runs, pretrained weights or GPU calls."""
    run = Path(run)
    generation = json.loads((run / "generation.json").read_text())
    if not generation["complete"]:
        raise ValueError("Generation incomplete")
    for name, digest in generation["files"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Captured input changed: {name}")
    provenance = json.loads((run / "provenance.json").read_text())
    for name, version in provenance["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Package differs from generation: {name}")
    for filename, digest in provenance["files"].items():
        path = Path(filename)
        if path.suffix == ".py" or path in scorer_files():
            if file_hash(path) != digest:
                raise ValueError(f"Evaluation source changed: {path}")
    if len((run / "samples.jsonl").read_text().splitlines()) != generation["count"]:
        raise ValueError("Sample count differs")
    output = new_run(output, {"source_run": str(run.resolve())})
    config = json.loads((run / "pilot_config.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    try:
        report = evaluate_subtb(run, output, predictor)
        write_json(
            output / "status.json",
            {"status": "complete", "source_fixture": provenance["fixture"]},
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    return report
