"""Matched TB fits testing a direct task-context path, with frozen ProGen.

Keep historical charge code unchanged so its saved runs remain replayable. Reuse
its rollout, optimizer update, reward and evaluation functions verbatim.
"""

import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..artifacts import file_hash, new_run, write_json
from ..backbone import TOKENIZER_PATH, FrozenProGen
from ..guidance import GuidanceHead
from ..pipeline import project_path
from .baseline import scorer_files
from .charge import rollout, update_head
from .charge_experiment import (
    check_charge_inputs,
    evaluate_charge,
    generate_records,
    work_delta,
)
from .learning_diagnostic import behavior_probe
from .scoring import predict_disorder


class DirectContextHead(GuidanceHead):
    """Add 100 zero-initialized parameters, without new inputs or random draws."""

    def __init__(self, feature_size, context_size, actions, hidden_size=128):
        super().__init__(feature_size, context_size, actions, hidden_size)
        # A Parameter avoids a Linear constructor consuming extra random numbers.
        # No bias: the original head already has an output bias. No prompt shortcut.
        self.task_weight = torch.nn.Parameter(torch.zeros(actions, 5))

    def forward(self, features, context):
        return super().forward(features, context) + F.linear(
            context[..., :5], self.task_weight
        )


def make_context_head(backbone, prompt_count, config, seed, arm):
    """Identical shared initialization in both arms, before moving to the device."""
    if arm not in ("standard", "direct_context"):
        raise ValueError(f"Unknown context arm: {arm}")
    head_class = GuidanceHead if arm == "standard" else DirectContextHead
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = head_class(
            backbone.hidden_size, prompt_count + 5, 20, config["hidden_size"]
        )
    return head.to(backbone.device)


def check_context_inputs(settings, check_checkpoint=True):
    """Pin the pilot protocol; do not access CUDA or execute a model here."""
    source = project_path(settings["source_run"])
    if file_hash(source / "config.json") != settings["source_config_sha256"]:
        raise ValueError("Source pilot configuration changed")
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("Source pilot is incomplete")
    if settings["arms"] != ["standard", "direct_context"]:
        raise ValueError("Use the fixed standard/direct_context comparison")
    config = json.loads((source / "config.json").read_text())
    references, manifest = check_charge_inputs(config, check_checkpoint)
    if references != json.loads((source / "references.json").read_text()):
        raise ValueError("Prepared prompts differ from the pilot")
    return config, references, manifest


def train_head(backbone, prompts, config, seed, arm, output):
    """Original 400-step training algorithm, fresh initialization for each arm."""
    head = make_context_head(backbone, len(prompts), config, seed, arm)
    log_z = torch.nn.Parameter(torch.zeros(len(prompts), device=backbone.device))
    rates = (config["learning_rate"], config["normalizer_learning_rate"])
    optimizer = torch.optim.Adam(
        [
            {"params": list(head.parameters()), "lr": rates[0]},
            {"params": [log_z], "lr": rates[1]},
        ]
    )
    generator = torch.Generator(device=backbone.device).manual_seed(1000 + 10 * seed)
    indices = list(range(len(prompts)))
    started, before = time.monotonic(), dict(backbone.work)
    with (output / "training.jsonl").open("w") as log:
        for step in range(1, config["steps"] + 1):
            factor = (
                config["final_lr_factor"]
                if step > int(config["steps"] * config["decay_fraction"])
                else 1.0
            )
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = factor * rate
            batch = rollout(
                backbone, head, prompts, indices, config, generator, retain=True
            )
            stats = update_head(head, log_z, optimizer, batch, indices, config)
            stats.update(
                step=step,
                learning_rate=optimizer.param_groups[0]["lr"],
                log_z=log_z.detach().cpu().tolist(),
                seconds=time.monotonic() - started,
            )
            log.write(json.dumps(stats, allow_nan=False) + "\n")
            log.flush()
            del batch
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                print(
                    f"{arm} seed {seed} step {step}/{config['steps']}: "
                    f"TB {stats['tb_loss']:.4g}, charge {stats['charge_success']:.1%}",
                    flush=True,
                )
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                torch.save(
                    {
                        "architecture": arm,
                        "head": head.state_dict(),
                        "log_z": log_z.detach(),
                        "optimizer": optimizer.state_dict(),
                        "generator_state": generator.get_state(),
                        "step": step,
                        "prompt_ids": [r["id"] for r in prompts],
                        "config": config,
                    },
                    output / f"step-{step}.pt",
                )
    cost = {
        "seconds": time.monotonic() - started,
        "trajectories": config["steps"] * len(prompts),
        "backbone_work": work_delta(backbone, before),
        "head_parameters": sum(p.numel() for p in head.parameters()),
    }
    return head, log_z, cost


def generate_with_behavior(backbone, head, log_z, prompts, config, seed):
    """Raw final-policy samples with token effects on those same sampled states."""
    generator = torch.Generator(device=backbone.device).manual_seed(1002 + 10 * seed)
    records, behavior = [], []
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
        behavior.append(behavior_probe(head, batch))
        for index, sequence in enumerate(batch["sequences"]):
            records.append(
                {
                    "id": f"gfn-{seed}-{index}-{repeat}",
                    "arm": "gfn",
                    "seed": seed,
                    "prompt_id": prompts[index]["id"],
                    "prefix": prompts[index]["sequence"][: config["prefix_length"]],
                    "sequence": sequence,
                    "evaluation_start": config["prefix_length"],
                    "log_prior": float(batch["log_prior"][index]),
                    "log_policy": float(batch["log_policy"][index]),
                    "log_z": float(log_z[index].detach()),
                    "mean_canonical_probability_mass": float(
                        batch["canonical_mass"][index]
                    ),
                }
            )
    return records, behavior


def evaluate_context(run, output, predictor=predict_disorder):
    """Reuse pilot scorers and controls; report paired seed/prompt comparisons."""
    config = json.loads((run / "pilot_config.json").read_text())
    references = json.loads((run / "references.json").read_text())
    costs = json.loads((run / "costs.json").read_text())
    all_samples = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    prior = [r for r in all_samples if r["arm"] == "prior_pool"]
    results = {}
    for arm in ("standard", "direct_context"):
        destination = output / arm
        destination.mkdir()
        samples = prior + [r for r in all_samples if r.get("architecture") == arm]
        results[arm] = evaluate_charge(
            samples, references, config, destination, costs[arm], predictor
        )
    metrics = {"arms": {}, "by_seed": {}, "by_prompt": {}}
    for arm, result in results.items():
        metrics["arms"][arm] = result["arms"]["gfn"]
        for section in ("by_seed", "by_prompt"):
            metrics[section][arm] = {
                key.removeprefix("gfn/"): value
                for key, value in result[section].items()
                if key.startswith("gfn/")
            }
    metrics["arms"]["prior"] = results["standard"]["arms"]["prior"]
    metrics["arms"]["select"] = results["standard"]["arms"]["select"]
    behavior = json.loads((run / "behavior.json").read_text())
    write_json(output / "metrics.json", metrics)
    lines = [
        "# Direct charge-context comparison",
        "",
        "Fresh matched fits; frozen ProGen, unchanged charge reward and TB schedule.",
        "",
        "| Arm | n | Charge pass | Isolated disorder | Joint pass | Unique tails | Mean score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, stats in metrics["arms"].items():
        lines.append(
            f"| {arm} | {stats['count']} | {stats['charge_success_rate']:.1%} | "
            f"{stats['v3_isolated_tail_success_rate']:.1%} | "
            f"{stats['joint_charge_and_isolated_disorder_rate']:.1%} | "
            f"{stats['unique_tail_fraction']:.1%} | {stats['mean_score']:.3f} |"
        )
    lines += [
        "",
        "## Paired seeds",
        "",
        "| Seed | Standard charge | Direct charge | Standard joint | Direct joint |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed in config["seeds"]:
        left, right = [metrics["by_seed"][arm][f"seed-{seed}"] for arm in results]
        lines.append(
            f"| {seed} | {left['charge_success_rate']:.1%} | {right['charge_success_rate']:.1%} | "
            f"{left['joint_charge_and_isolated_disorder_rate']:.1%} | "
            f"{right['joint_charge_and_isolated_disorder_rate']:.1%} |"
        )
    lines += [
        "",
        "## Token corrections on final evaluation prefixes",
        "",
        "| Arm / seed | First-half charge change | Second-half charge change | Context contrast |",
        "|---|---:|---:|---:|",
    ]
    for key, batches in behavior.items():
        fields = (
            "first_half_expected_charge_change",
            "second_half_expected_charge_change",
            "counterfactual_mean_charge_contrast",
        )
        values = [float(np.mean([b[field] for b in batches])) for field in fields]
        lines.append(f"| {key} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    lines += [
        "",
        "The prior pool is shared, not independent replication. Selection uses four times the proposals.",
        "Inspect each arm's report/metrics for per-prompt results, diversity, repetition and TB residuals.",
        "Disorder is predicted. Lower residual is not exact target matching. No best checkpoint is selected.",
        "The branch adds capacity as well as changing routing; a gain would not isolate feature scaling as the cause.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return output / "report.md"


def run_context(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    """One user command runs both arms, shared controls and CPU scoring."""
    config, references, manifest = check_context_inputs(settings, backbone is None)
    output = new_run(output, settings)
    fixture = backbone is not None
    started = time.monotonic()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for filename, value in (
            ("pilot_config.json", config),
            ("references.json", references),
            ("reference_manifest.json", manifest),
        ):
            write_json(output / filename, value)
        source_root = Path(__file__).parent.parent
        files = [
            TOKENIZER_PATH,
            *Path(__file__).parent.glob("*.py"),
            source_root / "guidance.py",
            source_root / "backbone.py",
            source_root / "artifacts.py",
            *(source_root / "vendor/progen").glob("*.py"),
            *scorer_files(),
        ]
        if not fixture:
            model_files = [
                project_path(config["checkpoint"]) / name
                for name in ("config.json", "pytorch_model.bin")
            ]
            source_provenance = json.loads(
                (project_path(settings["source_run"]) / "provenance.json").read_text()
            )
            for path in model_files:
                if file_hash(path) != source_provenance["files"].get(
                    str(path.resolve())
                ):
                    raise ValueError("Pretrained checkpoint differs from pilot")
            files += model_files
        write_json(
            output / "provenance.json",
            {
                "files": {str(p.resolve()): file_hash(p) for p in files if p.is_file()},
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "numpy", "transformers", "metapredict")
                },
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "fixture": fixture,
            },
        )
        if backbone is None:
            print("Loading frozen ProGen for the context comparison", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        prompts = [r for r in references if r["role"] == "prompt"]
        costs = {arm: {} for arm in settings["arms"]}
        behaviors, count = {}, 0
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
                    torch.Generator(device=backbone.device).manual_seed(
                        1001 + 10 * seed
                    ),
                )
                prior_cost = {
                    "seconds": time.monotonic() - stage_start,
                    "trajectories": len(prior),
                    "backbone_work": work_delta(backbone, before),
                    "shared_between_arms": True,
                }
                for row in prior:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                count += len(prior)
                handle.flush()
                for arm in settings["arms"]:
                    seed_dir = output / f"{arm}-seed-{seed}"
                    seed_dir.mkdir()
                    head, log_z, training_cost = train_head(
                        backbone, prompts, config, seed, arm, seed_dir
                    )
                    stage_start, before = time.monotonic(), dict(backbone.work)
                    rows, behavior = generate_with_behavior(
                        backbone, head, log_z, prompts, config, seed
                    )
                    behaviors[f"{arm}/seed-{seed}"] = behavior
                    costs[arm][str(seed)] = {
                        "training": training_cost,
                        "prior_pool": prior_cost,
                        "gfn": {
                            "seconds": time.monotonic() - stage_start,
                            "trajectories": len(rows),
                            "backbone_work": work_delta(backbone, before),
                        },
                    }
                    for row in rows:
                        row["architecture"] = arm
                        row["id"] = f"{arm}-{row['id']}"
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                    count += len(rows)
                    handle.flush()
                    write_json(output / "costs.json", costs)
                    write_json(output / "behavior.json", behaviors)
                    del head, log_z
        captured = (
            "config.json",
            "pilot_config.json",
            "references.json",
            "reference_manifest.json",
            "samples.jsonl",
            "costs.json",
            "behavior.json",
            "provenance.json",
        )
        write_json(
            output / "generation.json",
            {
                "complete": True,
                "count": count,
                "files": {name: file_hash(output / name) for name in captured},
                "seconds_before_evaluation": time.monotonic() - started,
            },
        )
        evaluate_context(output, output, predictor)
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


def reevaluate_context(run, output, predictor=predict_disorder):
    """Replay complete samples on CPU without loading any generator checkpoint."""
    run = Path(run)
    generation = json.loads((run / "generation.json").read_text())
    if not generation["complete"]:
        raise ValueError("Context generation is incomplete")
    for name, digest in generation["files"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Saved context artifact changed: {name}")
    provenance = json.loads((run / "provenance.json").read_text())
    for filename, digest in provenance["files"].items():
        # Scorers and Python sources affect replay; ProGen weights are unnecessary.
        path = Path(filename)
        if path.suffix == ".py" or path in scorer_files():
            if file_hash(path) != digest:
                raise ValueError(f"Evaluation source changed: {path}")
    if len((run / "samples.jsonl").read_text().splitlines()) != generation["count"]:
        raise ValueError("Context sample count mismatch")
    output = new_run(output, {"source_run": str(run.resolve())})
    config = json.loads((run / "pilot_config.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    try:
        report = evaluate_context(run, output, predictor)
        write_json(
            output / "status.json",
            {"status": "complete", "source_fixture": provenance["fixture"]},
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    return report
