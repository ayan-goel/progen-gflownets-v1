"""Charge-guidance training, matched comparisons, and recoverable CPU evaluation."""

import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import TOKENIZER_PATH, FrozenProGen
from ..pipeline import project_path
from .baseline import check_inputs, scorer_files
from .charge import charge_reward, make_head, rollout, update_head
from .scoring import predict_disorder, score_records, summarize


def check_charge_inputs(config, check_checkpoint=True):
    """Validate the complete protocol without loading models or querying CUDA."""
    references, manifest = check_inputs(config, check_checkpoint)
    tail_length = config["length"] - config["prefix_length"]
    if tail_length % 2:
        raise ValueError("Charge experiment needs equal-sized blocks")
    prompts = [r for r in references if r["role"] == "prompt"]
    if any(not isinstance(seed, int) or seed < 0 for seed in config["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if config["batch_size"] != len(prompts):
        raise ValueError("Use exactly one trajectory per prompt in each training batch")
    for key in (
        "steps",
        "hidden_size",
        "selection_factor",
        "swap_proposals",
        "log_every",
        "checkpoint_every",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key in ("learning_rate", "normalizer_learning_rate", "gradient_clip"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(config["beta"]) or config["beta"] < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not 0 < config["decay_fraction"] <= 1 or not 0 < config["final_lr_factor"] <= 1:
        raise ValueError("Invalid fixed learning-rate schedule")
    reward = config["reward"]
    targets = reward["block_targets"]
    if len(targets) != 2 or not all(np.isfinite(t) and abs(t) <= 1 for t in targets):
        raise ValueError("Need two feasible nominal charge targets")
    for key in ("block_scale", "fcr_scale", "block_tolerance", "fcr_tolerance"):
        if not np.isfinite(reward[key]) or reward[key] <= 0:
            raise ValueError(f"Invalid reward {key}")
    if not np.isfinite(reward["fcr_target"]) or not 0 <= reward["fcr_target"] <= 1:
        raise ValueError("Invalid target charged fraction")
    if reward["fcr_target"] < sum(abs(t) for t in targets) / 2:
        raise ValueError("Charged fraction cannot realize the requested signed means")
    return references, manifest


def work_delta(backbone, before):
    return {key: value - before[key] for key, value in backbone.work.items()}


def generate_records(
    backbone, head, prompts, config, seed, count, arm, generator, log_z=None
):
    """Round-robin prompts give exactly the same count to each context."""
    records = []
    for repeat in range(count):
        indices = list(range(len(prompts)))
        batch = rollout(backbone, head, prompts, indices, config, generator)
        for index, sequence in enumerate(batch["sequences"]):
            row = {
                "id": f"{arm}-{seed}-{index}-{repeat}",
                "arm": arm,
                "seed": seed,
                "prompt_id": prompts[index]["id"],
                "prefix": prompts[index]["sequence"][: config["prefix_length"]],
                "sequence": sequence,
                "evaluation_start": config["prefix_length"],
                "log_prior": float(batch["log_prior"][index]),
                "log_policy": float(batch["log_policy"][index]),
                "mean_canonical_probability_mass": float(
                    batch["canonical_mass"][index]
                ),
            }
            if log_z is not None:
                row["log_z"] = float(log_z[index].detach())
            records.append(row)
        if (repeat + 1) % 8 == 0 or repeat + 1 == count:
            print(f"seed {seed} {arm}: {repeat + 1}/{count} per prompt", flush=True)
    return records


def derive_comparators(samples, config):
    """Selection is within prompt/seed; editing and shuffling preserve composition.

    Selection uses only the charge score, never disorder. The prior's first k
    draws form the ordinary-sampling arm before the whole 4k pool is ranked.
    """
    records = list(samples)
    prefix_length = config["prefix_length"]
    count = config["samples_per_prompt"]
    for seed in config["seeds"]:
        rng = np.random.default_rng(100000 + seed)
        pool = [r for r in samples if r["seed"] == seed and r["arm"] == "prior_pool"]
        for prompt_id in dict.fromkeys(r["prompt_id"] for r in pool):
            group = [r for r in pool if r["prompt_id"] == prompt_id]
            scores = charge_reward(
                [r["sequence"][prefix_length:] for r in group], config["reward"]
            )[0]
            # Stable tie handling keeps original sample order; no hidden deduplication.
            order = sorted(range(len(group)), key=lambda i: (-float(scores[i]), i))
            for arm, indices in (("prior", range(count)), ("select", order[:count])):
                for index in indices:
                    row = dict(
                        group[index],
                        arm=arm,
                        id=f"{arm}-{group[index]['id']}",
                        source_id=group[index]["id"],
                    )
                    if arm == "select":
                        # Original path density is not the selected-output density.
                        row.pop("log_policy")
                    records.append(row)
            for source in group[:count]:
                tail = list(source["sequence"][prefix_length:])
                best = float(charge_reward(["".join(tail)], config["reward"])[0][0])
                for _ in range(config["swap_proposals"]):
                    left, right = rng.choice(len(tail), 2, replace=False)
                    tail[left], tail[right] = tail[right], tail[left]
                    score = float(
                        charge_reward(["".join(tail)], config["reward"])[0][0]
                    )
                    if score > best:
                        best = score
                    else:
                        tail[left], tail[right] = tail[right], tail[left]
                row = dict(
                    source,
                    id=f"swap-{source['id']}",
                    arm="swap",
                    source_id=source["id"],
                )
                row["sequence"] = source["prefix"] + "".join(tail)
                # These transformations have no evaluated ProGen path likelihood.
                for key in (
                    "log_prior",
                    "log_policy",
                    "mean_canonical_probability_mass",
                ):
                    row.pop(key)
                records.append(row)
        for source in [r for r in samples if r["seed"] == seed and r["arm"] == "gfn"]:
            tail = list(source["sequence"][prefix_length:])
            rng.shuffle(tail)
            row = dict(
                source,
                id=f"shuffle-{source['id']}",
                arm="shuffle",
                source_id=source["id"],
            )
            row["sequence"] = source["prefix"] + "".join(tail)
            for key in (
                "log_prior",
                "log_policy",
                "log_z",
                "mean_canonical_probability_mass",
            ):
                row.pop(key)
            records.append(row)
    return records


def summarize_charge(rows, config):
    stats = summarize(rows)
    scores, blocks, fcr, passes = charge_reward(
        [r["sequence"][config["prefix_length"] :] for r in rows], config["reward"]
    )
    disorder = torch.tensor(
        [r["disorder"]["V3"]["isolated_tail"]["passes"] for r in rows]
    )
    stats.update(
        mean_score=float(scores.mean()),
        block_charge_means=blocks.mean(0).tolist(),
        charge_success_rate=float(passes.float().mean()),
        joint_charge_and_isolated_disorder_rate=float(
            (passes & disorder).float().mean()
        ),
        mean_repeated_trimer_fraction=float(
            np.mean([r["tail_descriptors"]["repeated_trimer_fraction"] for r in rows])
        ),
    )
    if all("log_prior" in r for r in rows):
        stats["mean_log_prior_per_generated_residue"] = float(
            np.mean([r["log_prior"] for r in rows])
            / (config["length"] - config["prefix_length"])
        )
    if all("log_z" in r for r in rows):
        residual = (
            np.array([r["log_z"] + r["log_policy"] - r["log_prior"] for r in rows])
            - config["beta"] * scores.numpy()
        )
        stats["heldout_tb_residual_rmse"] = float(np.sqrt(np.mean(residual**2)))
        stats["heldout_tb_residual_mean"] = float(residual.mean())
    return stats


def evaluate_charge(
    samples, references, config, output, costs, predictor=predict_disorder
):
    """Score all raw and derived outputs on CPU; no disorder-based filtering."""
    started = time.monotonic()
    records = derive_comparators(samples, config)
    scored = score_records(records, references, config, predictor)
    for row in scored:
        score, blocks, fcr, passes = charge_reward(
            [row["sequence"][config["prefix_length"] :]], config["reward"]
        )
        row["charge"] = {
            "score": float(score[0]),
            "blocks": blocks[0].tolist(),
            "fcr": float(fcr[0]),
            "passes": bool(passes[0]),
        }
    write_json(output / "sequence_metrics.json", scored)
    groups = {"arms": {}, "by_seed": {}, "by_prompt": {}, "by_seed_prompt": {}}
    for row in scored:
        keys = (
            row["arm"],
            f"{row['arm']}/seed-{row['seed']}",
            f"{row['arm']}/{row['prompt_id']}",
            f"{row['arm']}/seed-{row['seed']}/{row['prompt_id']}",
        )
        for group, key in zip(groups.values(), keys):
            group.setdefault(key, []).append(row)
    metrics = {
        name: {key: summarize_charge(rows, config) for key, rows in group.items()}
        for name, group in groups.items()
    }
    metrics["notes"] = [
        "Selection uses more prior proposals per output; training cost is additional for GFN.",
        "Prior and selection share a pool; derived arms are not independent samples.",
        "Disorder is predicted, V2 is related, and reference similarity is not global novelty.",
        "TB residual on fresh samples is a diagnostic, not exact distribution distance.",
        "Swap and shuffle are composition-preserving controls, not prior-target samplers.",
    ]
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "evaluation_cost.json",
        {
            "seconds": time.monotonic() - started,
            "records_including_derived": len(scored),
            "swap_score_evaluations": sum(r["arm"] == "swap" for r in scored)
            * (config["swap_proposals"] + 1),
        },
    )
    lines = [
        "# IDR charge-guidance pilot",
        "",
        "All main comparisons use the same prefixes and generated length. No disorder filtering.",
        "",
        "| Method | n | Charge pass | Isolated disorder pass | Joint pass | Unique tails | Mean charge score | Repeated trimers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("prior", "select", "gfn", "swap", "shuffle", "prior_pool"):
        s = metrics["arms"][arm]
        lines.append(
            f"| {arm} | {s['count']} | {s['charge_success_rate']:.1%} | {s['v3_isolated_tail_success_rate']:.1%} | {s['joint_charge_and_isolated_disorder_rate']:.1%} | {s['unique_tail_fraction']:.1%} | {s['mean_score']:.3f} | {s['mean_repeated_trimer_fraction']:.3f} |"
        )
    lines += [
        "",
        f"select keeps the best 1/{config['selection_factor']} by charge score within each prompt and seed. prior is the first unselected subset of that pool. gfn is raw final-head sampling.",
        "",
        "swap greedily edits prior tails without changing composition; shuffle permutes GFN tails. Neither respects the original sampling distribution.",
        "",
        "Inspect metrics.json by seed and prompt, including RPB1 repetition and joint disorder/charge success. Do not infer distribution accuracy from charge improvement or low TB residual alone.",
        "",
        "## Joint charge and disorder success",
        "",
        "| Seed | Prior | Select | GFN | GFN TB residual RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed in config["seeds"]:
        groups = [
            metrics["by_seed"][f"{arm}/seed-{seed}"]
            for arm in ("prior", "select", "gfn")
        ]
        rates = " | ".join(
            f"{group['joint_charge_and_isolated_disorder_rate']:.1%}"
            for group in groups
        )
        lines.append(
            f"| {seed} | {rates} | {groups[-1]['heldout_tb_residual_rmse']:.3f} |"
        )
    lines += [
        "",
        "| Prompt | Prior | Select | GFN | GFN unique tails |",
        "|---|---:|---:|---:|---:|",
    ]
    for ref in references:
        if ref["role"] != "prompt":
            continue
        groups = [
            metrics["by_prompt"][f"{arm}/{ref['id']}"]
            for arm in ("prior", "select", "gfn")
        ]
        rates = " | ".join(
            f"{group['joint_charge_and_isolated_disorder_rate']:.1%}"
            for group in groups
        )
        lines.append(
            f"| {ref['id']} | {rates} | {groups[-1]['unique_tail_fraction']:.1%} |"
        )
    lines += [
        "",
        "## Measured generation/training costs",
        "",
        "| Seed | Stage | Seconds | Generated trajectories |",
        "|---|---|---:|---:|",
    ]
    for seed, stages in costs.items():
        for stage, cost in stages.items():
            lines.append(
                f"| {seed} | {stage} | {cost['seconds']:.1f} | {cost['trajectories']} |"
            )
    lines += [
        "",
        "Source model loading/hashing and CPU evaluation are separate from these stage times. Selection consumes the complete saved pool, including rejected proposals. No end-to-end efficiency advantage is assumed.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return metrics


def run_charge(
    config, output, device="cuda", backbone=None, predictor=predict_disorder
):
    references, manifest = check_charge_inputs(config, backbone is None)
    output = new_run(output, config)
    fixture = backbone is not None
    started = time.monotonic()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write_json(output / "references.json", references)
        write_json(output / "reference_manifest.json", manifest)
        source_root = Path(__file__).parent.parent
        files = [
            TOKENIZER_PATH,
            *Path(__file__).parent.glob("*.py"),
            source_root / "guidance.py",
            source_root / "backbone.py",
            source_root / "vendor/progen/modeling_progen.py",
            source_root / "vendor/progen/configuration_progen.py",
        ]
        files.extend(p for p in scorer_files() if p.is_file())
        if not fixture:
            files += [
                project_path(config["checkpoint"]) / name
                for name in ("config.json", "pytorch_model.bin")
            ]
        write_json(
            output / "provenance.json",
            {
                "files": {str(p.resolve()): file_hash(p) for p in files},
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
            print("Loading frozen ProGen for charge guidance", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        prompts = [r for r in references if r["role"] == "prompt"]
        samples, costs = [], {}
        with (output / "samples.jsonl").open("w") as sample_file:
            for seed in config["seeds"]:
                costs[str(seed)] = {}
                seed_dir = output / f"seed-{seed}"
                seed_dir.mkdir()
                # Initialize on CPU, then move: CPU/GPU use the same initialization seed.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    head = make_head(backbone, len(prompts), config["hidden_size"])
                log_z = torch.nn.Parameter(
                    torch.zeros(len(prompts), device=backbone.device)
                )
                optimizer = torch.optim.Adam(
                    [
                        {
                            "params": list(head.parameters()),
                            "lr": config["learning_rate"],
                        },
                        {"params": [log_z], "lr": config["normalizer_learning_rate"]},
                    ]
                )
                generator = torch.Generator(device=backbone.device).manual_seed(
                    1000 + 10 * seed
                )
                stage_start, before = time.monotonic(), dict(backbone.work)
                with (seed_dir / "training.jsonl").open("w") as log:
                    for step in range(1, config["steps"] + 1):
                        factor = (
                            config["final_lr_factor"]
                            if step > int(config["steps"] * config["decay_fraction"])
                            else 1.0
                        )
                        for group, rate in zip(
                            optimizer.param_groups,
                            (
                                config["learning_rate"],
                                config["normalizer_learning_rate"],
                            ),
                        ):
                            group["lr"] = rate * factor
                        batch = rollout(
                            backbone,
                            head,
                            prompts,
                            list(range(len(prompts))),
                            config,
                            generator,
                            retain=True,
                        )
                        stats = update_head(
                            head,
                            log_z,
                            optimizer,
                            batch,
                            list(range(len(prompts))),
                            config,
                        )
                        stats.update(
                            step=step,
                            learning_rate=optimizer.param_groups[0]["lr"],
                            log_z=log_z.detach().cpu().tolist(),
                            seconds=time.monotonic() - stage_start,
                        )
                        log.write(json.dumps(stats, allow_nan=False) + "\n")
                        log.flush()
                        del batch
                        if (
                            step == 1
                            or step % config["log_every"] == 0
                            or step == config["steps"]
                        ):
                            print(
                                f"seed {seed} step {step}/{config['steps']}: TB {stats['tb_loss']:.4g}, charge pass {stats['charge_success']:.1%}, elapsed {stats['seconds']:.0f}s",
                                flush=True,
                            )
                        if (
                            step % config["checkpoint_every"] == 0
                            or step == config["steps"]
                        ):
                            torch.save(
                                {
                                    "head": head.state_dict(),
                                    "log_z": log_z.detach(),
                                    "optimizer": optimizer.state_dict(),
                                    "generator_state": generator.get_state(),
                                    "step": step,
                                    "prompt_ids": [r["id"] for r in prompts],
                                    "config": config,
                                },
                                seed_dir / f"step-{step}.pt",
                            )
                costs[str(seed)]["training"] = {
                    "seconds": time.monotonic() - stage_start,
                    "trajectories": config["steps"] * len(prompts),
                    "backbone_work": work_delta(backbone, before),
                }
                for arm, model, count, offset in (
                    (
                        "prior_pool",
                        None,
                        config["samples_per_prompt"] * config["selection_factor"],
                        1,
                    ),
                    ("gfn", head, config["samples_per_prompt"], 2),
                ):
                    stage_start, before = time.monotonic(), dict(backbone.work)
                    rows = generate_records(
                        backbone,
                        model,
                        prompts,
                        config,
                        seed,
                        count,
                        arm,
                        torch.Generator(device=backbone.device).manual_seed(
                            1000 + 10 * seed + offset
                        ),
                        log_z if arm == "gfn" else None,
                    )
                    for row in rows:
                        sample_file.write(json.dumps(row, allow_nan=False) + "\n")
                    sample_file.flush()
                    samples.extend(rows)
                    costs[str(seed)][arm] = {
                        "seconds": time.monotonic() - stage_start,
                        "trajectories": len(rows),
                        "backbone_work": work_delta(backbone, before),
                    }
                write_json(output / "costs.json", costs)
        (output / "samples.fasta").write_text(
            "".join(f">{r['id']}\n{r['sequence']}\n" for r in samples)
        )
        write_json(
            output / "generation.json",
            {
                "complete": True,
                "count": len(samples),
                "samples_sha256": file_hash(output / "samples.jsonl"),
                "config_sha256": file_hash(output / "config.json"),
                "costs_sha256": file_hash(output / "costs.json"),
                "seconds_before_evaluation": time.monotonic() - started,
            },
        )
        evaluate_charge(samples, references, config, output, costs, predictor)
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


def reevaluate_charge(run, output, predictor=predict_disorder):
    """Recover evaluation from complete saved samples without loading ProGen."""
    run = Path(run)
    generation = json.loads((run / "generation.json").read_text())
    if not generation["complete"]:
        raise ValueError("Generation is incomplete")
    for name, key in (
        ("samples.jsonl", "samples_sha256"),
        ("config.json", "config_sha256"),
        ("costs.json", "costs_sha256"),
    ):
        if file_hash(run / name) != generation[key]:
            raise ValueError(f"Saved {name} changed")
    manifest = json.loads((run / "reference_manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    if (
        file_hash(run / "reference_manifest.json") != config["manifest_sha256"]
        or file_hash(run / "references.json") != manifest["references_sha256"]
    ):
        raise ValueError("Saved references changed")
    provenance = json.loads((run / "provenance.json").read_text())
    for path in scorer_files() + [
        Path(__file__),
        Path(__file__).with_name("charge.py"),
        Path(__file__).with_name("scoring.py"),
        Path(__file__).with_name("data.py"),
    ]:
        if path.is_file() and file_hash(path) != provenance["files"].get(
            str(path.resolve())
        ):
            raise ValueError("Evaluation implementation changed")
    samples = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    if len(samples) != generation["count"]:
        raise ValueError("Sample count mismatch")
    references = json.loads((run / "references.json").read_text())
    costs = json.loads((run / "costs.json").read_text())
    output = new_run(
        output, {"source_run": str(run.resolve()), "source_config": config}
    )
    try:
        torch.set_num_threads(config["cpu_threads"])
        evaluate_charge(samples, references, config, output, costs, predictor)
        write_json(
            output / "status.json",
            {"status": "complete", "source_fixture": provenance["fixture"]},
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    return output / "report.md"
