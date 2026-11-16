"""One bounded branch-rate follow-up, reusing pinned completed controls."""

import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..pipeline import project_path
from .baseline import scorer_files
from .charge import rollout, update_head
from .charge_experiment import check_charge_inputs, evaluate_charge, work_delta
from .context_experiment import generate_with_behavior, make_context_head
from .scoring import predict_disorder


def check_rate_inputs(settings, check_checkpoint=True):
    """Verify saved controls and current implementation without executing models."""
    source = project_path(settings["source_run"])
    required = {
        "generation.json",
        "standard/metrics.json",
        "direct_context/metrics.json",
    }
    if set(settings["source_files"]) != required:
        raise ValueError("Require the fixed control manifests and metrics")
    if settings["branch_rate_multiplier"] != 10:
        raise ValueError("This follow-up fixes a tenfold branch-rate change")
    for name, digest in settings["source_files"].items():
        if file_hash(source / name) != digest:
            raise ValueError(f"Pinned control changed: {name}")
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("Control comparison is incomplete")
    generation = json.loads((source / "generation.json").read_text())
    if not generation["complete"]:
        raise ValueError("Control generation is incomplete")
    for name, digest in generation["files"].items():
        if file_hash(source / name) != digest:
            raise ValueError(f"Control artifact changed: {name}")
    provenance = json.loads((source / "provenance.json").read_text())
    for name, version in provenance["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Package differs from control: {name}")
    for filename, digest in provenance["files"].items():
        path = Path(filename)
        if path.suffix == ".py" or path in scorer_files():
            if file_hash(path) != digest:
                raise ValueError(f"Control implementation changed: {path}")
    config = json.loads((source / "pilot_config.json").read_text())
    references, _ = check_charge_inputs(config, check_checkpoint)
    if references != json.loads((source / "references.json").read_text()):
        raise ValueError("References differ from the control")
    if check_checkpoint and provenance["fixture"]:
        raise ValueError("Real training cannot use fixture controls")
    return source, config, references, provenance


def branch_optimizer(head, log_z, config, multiplier):
    """Separate Adam rate for task_weight; all other parameters retain old rates."""
    base_parameters = [
        p for name, p in head.named_parameters() if name != "task_weight"
    ]
    return torch.optim.Adam(
        [
            {
                "params": base_parameters,
                "lr": config["learning_rate"],
                "name": "original_head",
            },
            {
                "params": [head.task_weight],
                "lr": config["learning_rate"] * multiplier,
                "name": "task_branch",
            },
            {
                "params": [log_z],
                "lr": config["normalizer_learning_rate"],
                "name": "normalizer",
            },
        ]
    )


def set_rates(optimizer, config, multiplier, step):
    """Same boundary as the control: decay starts after step 300 of 400."""
    factor = (
        config["final_lr_factor"]
        if step > int(config["steps"] * config["decay_fraction"])
        else 1.0
    )
    rates = (
        config["learning_rate"],
        config["learning_rate"] * multiplier,
        config["normalizer_learning_rate"],
    )
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate * factor


def train_rate_head(backbone, prompts, config, seed, multiplier, output):
    """Fresh zero-output initialization; never resume or rescale a completed head."""
    head = make_context_head(backbone, len(prompts), config, seed, "direct_context")
    log_z = torch.nn.Parameter(torch.zeros(len(prompts), device=backbone.device))
    optimizer = branch_optimizer(head, log_z, config, multiplier)
    generator = torch.Generator(device=backbone.device).manual_seed(1000 + 10 * seed)
    indices = list(range(len(prompts)))
    started, before = time.monotonic(), dict(backbone.work)
    with (output / "training.jsonl").open("w") as log:
        for step in range(1, config["steps"] + 1):
            set_rates(optimizer, config, multiplier, step)
            batch = rollout(
                backbone, head, prompts, indices, config, generator, retain=True
            )
            stats = update_head(head, log_z, optimizer, batch, indices, config)
            stats.update(
                step=step,
                learning_rate=optimizer.param_groups[0]["lr"],
                branch_learning_rate=optimizer.param_groups[1]["lr"],
                normalizer_learning_rate=optimizer.param_groups[2]["lr"],
                branch_weight_norm=float(head.task_weight.detach().norm()),
                log_z=log_z.detach().cpu().tolist(),
                seconds=time.monotonic() - started,
            )
            log.write(json.dumps(stats, allow_nan=False) + "\n")
            log.flush()
            del batch
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                print(
                    f"branch-rate seed {seed} step {step}/{config['steps']}: TB {stats['tb_loss']:.4g}, "
                    f"charge {stats['charge_success']:.1%}, branch lr {stats['branch_learning_rate']:.4g}",
                    flush=True,
                )
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                torch.save(
                    {
                        "architecture": "direct_context",
                        "branch_rate_multiplier": multiplier,
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
    return (
        head,
        log_z,
        {
            "seconds": time.monotonic() - started,
            "trajectories": config["steps"] * len(prompts),
            "backbone_work": work_delta(backbone, before),
        },
    )


def check_shared_controls(current, saved):
    """Counts/rates stay exact; continuous floats allow scorer batch roundoff."""
    for arm in ("prior", "prior_pool", "select", "swap"):
        if current["arms"][arm].keys() != saved["arms"][arm].keys():
            raise ValueError(f"Control metric fields differ: {arm}")
        for key, expected in saved["arms"][arm].items():
            actual = current["arms"][arm][key]
            if isinstance(expected, float) and not key.endswith("_rate"):
                matches = np.isclose(actual, expected, rtol=1e-6, atol=1e-7)
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"Reused control evaluation differs: {arm}/{key}")


def evaluate_rate(run, output, predictor=predict_disorder):
    """Score fresh samples; compare against copied, checksum-pinned control metrics."""
    config = json.loads((run / "pilot_config.json").read_text())
    references = json.loads((run / "references.json").read_text())
    samples = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    costs = json.loads((run / "costs.json").read_text())
    destination = output / "high_rate"
    destination.mkdir()
    high = evaluate_charge(samples, references, config, destination, costs, predictor)
    low = json.loads((run / "source/direct_context/metrics.json").read_text())
    standard = json.loads((run / "source/standard/metrics.json").read_text())
    # A different set of GFN outputs can change the scorer's batching. Preserve
    # exact pass rates while allowing negligible roundoff in continuous scores.
    check_shared_controls(high, low)
    results = {"low_rate": low, "high_rate": high, "standard": standard}
    write_json(output / "comparison.json", results)
    lines = [
        "# Charge-context branch-rate follow-up",
        "",
        "Only the new branch uses a tenfold rate. Low-rate/standard controls are reused, not new fits.",
        "",
        "| Arm | Charge pass | Isolated disorder | Joint pass | Unique tails | Mean score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, result in results.items():
        row = result["arms"]["gfn"]
        lines.append(
            f"| {arm} | {row['charge_success_rate']:.1%} | {row['v3_isolated_tail_success_rate']:.1%} | "
            f"{row['joint_charge_and_isolated_disorder_rate']:.1%} | {row['unique_tail_fraction']:.1%} | {row['mean_score']:.3f} |"
        )
    lines += [
        "",
        "## Paired seeds",
        "",
        "| Seed | Low charge | High charge | Low joint | High joint |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed in config["seeds"]:
        a, b = [r["by_seed"][f"gfn/seed-{seed}"] for r in (low, high)]
        lines.append(
            f"| {seed} | {a['charge_success_rate']:.1%} | {b['charge_success_rate']:.1%} | "
            f"{a['joint_charge_and_isolated_disorder_rate']:.1%} | {b['joint_charge_and_isolated_disorder_rate']:.1%} |"
        )
    low_behavior = json.loads((run / "source/behavior.json").read_text())
    high_behavior = json.loads((run / "behavior.json").read_text())
    lines += [
        "",
        "## Token response",
        "",
        "| Rate / seed | First-half change | Second-half change | Context contrast |",
        "|---|---:|---:|---:|",
    ]
    for rate, batches in (("low", low_behavior), ("high", high_behavior)):
        for seed in config["seeds"]:
            values = batches[f"direct_context/seed-{seed}"]
            fields = (
                "first_half_expected_charge_change",
                "second_half_expected_charge_change",
                "counterfactual_mean_charge_contrast",
            )
            mean = [np.mean([b[field] for b in values]) for field in fields]
            lines.append(
                f"| {rate} / {seed} | {mean[0]:.6f} | {mean[1]:.6f} | {mean[2]:.6f} |"
            )
    lines += [
        "",
        "Read high_rate/report.md and comparison.json for per-prompt, seed/prompt, control and diversity results.",
        "No seed/checkpoint selection. A loss decrease or larger internal response alone is not success.",
        "Disorder remains predicted; this inspected prompt panel is development evidence, not new biological validation.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return output / "report.md"


def run_rate(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    source, config, references, provenance = check_rate_inputs(
        settings, backbone is None
    )
    output = new_run(output, settings)
    started, fixture = time.monotonic(), backbone is not None
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        copied = [
            "generation.json",
            "pilot_config.json",
            "references.json",
            "reference_manifest.json",
            "provenance.json",
            "costs.json",
            "behavior.json",
            "standard/metrics.json",
            "direct_context/metrics.json",
        ]
        for name in copied:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        write_json(output / "pilot_config.json", config)
        write_json(output / "references.json", references)
        # Existing code/scorer hashes were checked above. Verify the large model
        # hashes only for the actual user run, before any pretrained execution.
        if not fixture:
            for name in ("config.json", "pytorch_model.bin"):
                path = project_path(config["checkpoint"]) / name
                if file_hash(path) != provenance["files"].get(str(path.resolve())):
                    raise ValueError("Pretrained model differs from control")
        provenance = dict(
            provenance,
            files=dict(provenance["files"]),
            device=str(device),
            fixture=fixture,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        provenance["files"][str(Path(__file__).resolve())] = file_hash(Path(__file__))
        write_json(output / "provenance.json", provenance)
        if backbone is None:
            print("Loading frozen ProGen for the branch-rate follow-up", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        prompts = [r for r in references if r["role"] == "prompt"]
        original_samples = [
            json.loads(line)
            for line in (source / "samples.jsonl").read_text().splitlines()
        ]
        prior = [r for r in original_samples if r["arm"] == "prior_pool"]
        original_costs = json.loads((source / "costs.json").read_text())
        costs, behavior = {}, {}
        with (output / "samples.jsonl").open("w") as handle:
            for row in prior:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            for seed in config["seeds"]:
                seed_dir = output / f"seed-{seed}"
                seed_dir.mkdir()
                head, log_z, training = train_rate_head(
                    backbone,
                    prompts,
                    config,
                    seed,
                    settings["branch_rate_multiplier"],
                    seed_dir,
                )
                stage_start, before = time.monotonic(), dict(backbone.work)
                rows, probes = generate_with_behavior(
                    backbone, head, log_z, prompts, config, seed
                )
                behavior[f"direct_context/seed-{seed}"] = probes
                prior_count = sum(r["seed"] == seed for r in prior)
                costs[str(seed)] = {
                    "training": training,
                    "gfn": {
                        "seconds": time.monotonic() - stage_start,
                        "trajectories": len(rows),
                        "backbone_work": work_delta(backbone, before),
                    },
                    "prior_pool": {
                        "seconds": 0.0,
                        "trajectories": 0,
                        "reused_trajectories": prior_count,
                        "original_cost": original_costs["direct_context"][str(seed)][
                            "prior_pool"
                        ],
                    },
                }
                for row in rows:
                    row.update(
                        id=f"high-rate-{row['id']}",
                        architecture="direct_context",
                        branch_rate_multiplier=10,
                    )
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                write_json(output / "costs.json", costs)
                write_json(output / "behavior.json", behavior)
                del head, log_z
        captured = [
            "config.json",
            "pilot_config.json",
            "references.json",
            "samples.jsonl",
            "behavior.json",
            "costs.json",
            "provenance.json",
        ]
        captured += [f"source/{name}" for name in copied]
        write_json(
            output / "generation.json",
            {
                "complete": True,
                "count": len(prior)
                + len(prompts) * config["samples_per_prompt"] * len(config["seeds"]),
                "files": {name: file_hash(output / name) for name in captured},
                "seconds_before_evaluation": time.monotonic() - started,
            },
        )
        evaluate_rate(output, output, predictor)
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


def reevaluate_rate(run, output, predictor=predict_disorder):
    """CPU recovery is self-contained in the new run; no old run/model required."""
    run = Path(run)
    generation = json.loads((run / "generation.json").read_text())
    if not generation["complete"]:
        raise ValueError("Generation incomplete")
    for name, digest in generation["files"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Captured input changed: {name}")
    provenance = json.loads((run / "provenance.json").read_text())
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
        report = evaluate_rate(run, output, predictor)
        write_json(
            output / "status.json",
            {"status": "complete", "source_fixture": provenance["fixture"]},
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    return report
