"""User-run development prior ablation, reusing fixed scorer predictions."""

import copy
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from .artifacts import file_hash, new_run, write_json
from .backbone import TOKENIZER_PATH, FrozenProGen, cache_tree
from .diversity import (
    descendant_log_counts,
    policy_report,
    read_development_run,
)
from .environment import PrefixTree, SequenceContext
from .evaluation import load_head, load_task
from .metrics import distribution_metrics
from .pipeline import (
    ROOT,
    progress_recorder,
    project_path,
    save_head,
    validate_backbone,
    validate_config,
)
from .target_diagnostics import candidate_priors
from .training import train_supervised, train_tb

ARMS = ["original", "count_corrected"]


def development_source_hashes(run):
    """Freeze consumed source artifacts; do not touch confirmation or test files."""
    run = Path(run)
    manifest = json.loads((run / "diagnostic_manifest.json").read_text())
    names = [
        "config.json",
        "status.json",
        "diagnostic_manifest.json",
        "panel_manifest.json",
        "provenance.json",
        "costs.json",
    ]
    for entry in manifest["parents"]:
        names += [
            f"{entry['id']}/{name}"
            for name in ("inputs.pt", "scorer_fits.json", "mlp-0.pt")
        ]
    return {name: file_hash(run / name) for name in names}


def check_diversity_inputs(config, smoke=False):
    """Validate on CPU before model loading; hashes fix the development data/scorer."""
    if config["experiment"] != "diversity" or config["arms"] != ARMS:
        raise ValueError("Expected original and count-corrected diversity arms")
    equivalent = copy.deepcopy(config)
    equivalent["experiment"] = "biological"
    equivalent["evaluation"]["filter_pool_count"] = config["evaluation"]["raw_count"]
    validate_config(equivalent)
    if config["precision"] != "float32" or any(
        config[m]["beta"] != 1 for m in ("training", "supervised")
    ):
        raise ValueError("Keep float32 and beta=1 for both learners")
    settings = config["evaluation"]
    for key in ("raw_count", "selected_count", "repeats", "seed_offset"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"evaluation.{key} must be a positive integer")
    if settings["selected_count"] > settings["raw_count"] or settings["repeats"] > 1000:
        raise ValueError("Invalid selection/repeat budget")
    if (
        config["training"]["normalizer_learning_rate"] <= 0
        or config["supervised"]["rollouts"] <= 0
    ):
        raise ValueError("Invalid learning rate or rollout count")
    source = project_path(config["development_run"])
    if development_source_hashes(source) != config["source_sha256"]:
        raise ValueError(
            "Development source artifacts changed from frozen configuration"
        )
    source_config, loaded = read_development_run(source)
    if bool(source_config["smoke"]) != smoke:
        raise ValueError(
            "Smoke requires synthetic source; real runs require real development data"
        )
    if not smoke and (len(loaded) != 3 or config["seeds"] != [0, 1, 2]):
        raise ValueError(
            "Real ablation retains all three development parents and seeds"
        )
    provenance = json.loads((source / "provenance.json").read_text())
    if file_hash(TOKENIZER_PATH) != provenance["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from saved scorer inputs")
    if not smoke:
        checkpoint = project_path(config["checkpoint"])
        hashes = {
            name: file_hash(checkpoint / name)
            for name in ("config.json", "pytorch_model.bin")
        }
        if hashes != provenance["checkpoint_sha256"]:
            raise ValueError("Backbone checkpoint differs from saved scorer inputs")
    return loaded, provenance


def fit_file_names(run, entries, config):
    """Exact required set for replay integrity, including failed-screen records."""
    names = ["config.json", "parents.json", "setup_costs.json"]
    for entry in entries:
        parent = entry["id"]
        names += [f"{parent}/inputs.pt", f"{parent}/scorer_validation.json"]
        for arm in config["arms"]:
            prefix = f"{parent}/{arm}"
            names += [
                f"{prefix}/task.pt",
                f"{prefix}/costs.json",
                f"{prefix}/fit_status.json",
            ]
            status = json.loads((Path(run) / prefix / "fit_status.json").read_text())
            if status["status"] == "complete":
                names += [
                    f"{prefix}/seed-{seed}/{method}.pt"
                    for seed in config["seeds"]
                    for method in ("gfn", "supervised")
                ]
            elif status["status"] != "scorer_gate_failed":
                raise ValueError("Unfinished development fit")
    return names


def evaluate_diversity(run, output):
    """CPU replay of complete development fits; no backbone or scorer fitting."""
    torch.set_num_threads(1)
    run, output = Path(run), Path(output)
    config = json.loads((run / "config.json").read_text())
    entries = json.loads((run / "parents.json").read_text())
    seal = json.loads((run / "fits_complete.json").read_text())
    required = fit_file_names(run, entries, config)
    if set(required) != set(seal) or any(
        file_hash(run / name) != digest for name, digest in seal.items()
    ):
        raise ValueError("Missing or changed saved development fit")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {
        "smoke": config["smoke"],
        "development_only": True,
        "parents": [],
        "settings": config["evaluation"],
        "setup_costs": json.loads((run / "setup_costs.json").read_text()),
    }
    timing = []
    lines = [
        "# Development diversity ablation",
        "",
        "Development evidence, not fresh confirmation. Means are over seeds, then sampling repeats where applicable.",
        f"Unique selection deduplicates {config['evaluation']['raw_count']} proposals before ranking; no refill. Quality and count must be read together.",
        "",
        "| Parent | Arm | Method | Measured raw | TV | Expected unique | Unique selected count | Unique selected measured | Fit seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        parent_path = run / entry["id"]
        inputs = torch.load(
            parent_path / "inputs.pt", map_location="cpu", weights_only=True
        )
        parent = {"id": entry["id"], "parent": entry["parent"], "arms": {}}
        original_task, _ = load_task(parent_path / "original")
        original_mean = (
            original_task.base_log_probabilities.double().softmax(0)
            @ inputs["labels"].double()
        ).item()
        for arm in config["arms"]:
            folder = parent_path / arm
            task, _ = load_task(folder)
            screen = json.loads((folder / "fit_status.json").read_text())
            costs = json.loads((folder / "costs.json").read_text())
            methods = ["prior", "exact_target", "uniform_candidates"]
            if screen["status"] == "complete":
                methods += ["gfn", "supervised"]
            rows = []
            matched_mean = (
                task.base_log_probabilities.double().softmax(0)
                @ inputs["labels"].double()
            ).item()
            for seed in config["seeds"]:
                for method in methods:
                    before = time.perf_counter()
                    head = None
                    checkpoint = {}
                    if method in ("gfn", "supervised"):
                        head, checkpoint = load_head(
                            task, folder / f"seed-{seed}/{method}.pt"
                        )
                    if method == "exact_target":
                        log_probabilities = (
                            task.base_log_probabilities.double() + task.scores.double()
                        ).log_softmax(0)
                    elif method == "uniform_candidates":
                        log_probabilities = torch.zeros(
                            len(task.scores), dtype=torch.float64
                        ).log_softmax(0)
                    else:
                        log_probabilities = task.all_log_probabilities(head).double()
                    if abs(log_probabilities.exp().sum().item() - 1) > 1e-4:
                        raise ValueError("Policy fails normalization")
                    distribution = distribution_metrics(
                        log_probabilities,
                        task.base_log_probabilities,
                        task.scores,
                        1,
                        checkpoint.get("log_z"),
                    )
                    distribution_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    report = policy_report(
                        log_probabilities.softmax(0),
                        inputs,
                        config["evaluation"],
                        config["evaluation"]["seed_offset"] + 1000 * seed,
                    )
                    categorical_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    if method in ("prior", "gfn", "supervised"):
                        task.sample(
                            head,
                            config["evaluation"]["raw_count"],
                            torch.Generator().manual_seed(900000 + seed),
                        )
                        trajectory_seconds = time.perf_counter() - before
                    else:
                        trajectory_seconds = None
                    file = f"{entry['id']}-{arm}-seed-{seed}-{method}.json"
                    write_json(output / file, report)
                    fit_cost = costs.get(str(seed), {}).get(method, {})
                    row = {
                        "seed": seed,
                        "method": method,
                        "exact": report["exact"],
                        "selection": report["selection"],
                        "distribution": distribution,
                        "gain_vs_matched_prior": report["exact"]["measured_mean"]
                        - matched_mean,
                        "gain_vs_original_prior": report["exact"]["measured_mean"]
                        - original_mean,
                        "fit_seconds": fit_cost.get("training_seconds", 0),
                        "fit_cost": fit_cost,
                        "gfn_tv_check": distribution["total_variation"] <= 0.01
                        if method == "gfn"
                        else None,
                    }
                    rows.append(row)
                    timing.append(
                        {
                            "id": entry["id"],
                            "arm": arm,
                            "seed": seed,
                            "method": method,
                            "distribution_seconds": distribution_seconds,
                            "categorical_reference_seconds": categorical_seconds,
                            "trajectory_seconds": trajectory_seconds,
                            "trajectory_count": config["evaluation"]["raw_count"]
                            if trajectory_seconds is not None
                            else 0,
                        }
                    )
            parent["arms"][arm] = {"fit_status": screen["status"], "rows": rows}
            for method in methods:
                selected = [row for row in rows if row["method"] == method]
                raw = mean([row["exact"]["measured_mean"] for row in selected])
                tv = mean([row["distribution"]["total_variation"] for row in selected])
                unique = mean([row["exact"]["expected_unique_raw"] for row in selected])
                count = mean([row["selection"]["unique"]["count"] for row in selected])
                quality = mean(
                    [row["selection"]["unique"]["measured_mean"] for row in selected]
                )
                cost = mean([row["fit_seconds"] for row in selected])
                lines.append(
                    f"| {entry['parent']} | {arm} | {method} | {raw:.5f} | {tv:.6f} | {unique:.2f} | {count:.2f} | {quality:.5f} | {cost:.2f} |"
                )
        summary["parents"].append(parent)
    aggregates = []
    for arm in config["arms"]:
        for method in (
            "prior",
            "exact_target",
            "uniform_candidates",
            "gfn",
            "supervised",
        ):
            parent_values = []
            for parent in summary["parents"]:
                rows = [
                    row
                    for row in parent["arms"][arm]["rows"]
                    if row["method"] == method
                ]
                if not rows:
                    continue
                parent_values.append(
                    {
                        "parent": parent["parent"],
                        "raw_gain_vs_matched_prior": mean(
                            row["gain_vs_matched_prior"] for row in rows
                        ),
                        "unique_selected_count": mean(
                            row["selection"]["unique"]["count"] for row in rows
                        ),
                        "unique_selected_measured": mean(
                            row["selection"]["unique"]["measured_mean"] for row in rows
                        ),
                        "mean_tv": mean(
                            row["distribution"]["total_variation"] for row in rows
                        ),
                        "fit_seconds": mean(row["fit_seconds"] for row in rows),
                    }
                )
            aggregates.append(
                {
                    "arm": arm,
                    "method": method,
                    "declared_parents": len(entries),
                    "reported_parents": len(parent_values),
                    "per_parent": parent_values,
                    "means": {
                        key: mean(row[key] for row in parent_values)
                        for key in parent_values[0]
                        if key != "parent"
                    }
                    if parent_values
                    else {},
                }
            )
    summary["equal_parent_aggregates"] = aggregates
    write_json(output / "summary.json", summary)
    write_json(
        output / "evaluation_costs.json",
        {"seconds": time.perf_counter() - started, "policies": timing},
    )
    lines += [
        "",
        "All fitted seeds and failed screens are retained in summary.json. Fit time is per seed; setup and original reused scorer/cache costs are reported separately.",
        "Categorical repeats are finite-law references, not an online speed measurement. Actual trajectory timing is separate. Equal updates are not equal compute.",
        "No automatic biological pass: assess distinct count, stability and cost jointly; use fresh data before any confirmation claim.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def run_diversity(config, output, device="cpu", smoke=False):
    """Reuse frozen development rewards; train the two fixed priors only on user GPU."""
    if (smoke and device != "cpu") or (not smoke and not device.startswith("cuda")):
        raise ValueError("Real fits require user CUDA; smoke fits require CPU")
    config = copy.deepcopy(config)
    loaded, provenance = check_diversity_inputs(config, smoke)
    if smoke:
        config["seeds"] = [0]
        for method in ("training", "supervised"):
            config[method].update(hidden_size=8, steps=4, batch_size=4, log_every=2)
        config["supervised"]["rollouts"] = 8
        config["evaluation"].update(raw_count=8, selected_count=4, repeats=2)
    config.update(smoke=smoke, device=device)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    output = new_run(output, config)
    started = time.perf_counter()
    try:
        shutil.copytree(
            ROOT / "src",
            output / "source/src",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(ROOT / "spec.md", output / "source/spec.md")
        write_json(
            output / "provenance.json",
            {
                "torch": torch.__version__,
                "device": device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "source_sha256": {
                    str(path.relative_to(output / "source")): file_hash(path)
                    for path in (output / "source").rglob("*")
                    if path.is_file()
                },
                "upstream": provenance,
                "development_source_sha256": config["source_sha256"],
            },
        )
        entries = [entry for entry, _, _ in loaded]
        write_json(output / "parents.json", entries)
        if not smoke:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        print(
            "Caching development prefixes; frozen scorer predictions reused", flush=True
        )
        backbone = (
            FrozenProGen.tiny()
            if smoke
            else FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        )
        write_json(
            output / "backbone_checks.json", validate_backbone(backbone, "float32")
        )
        setup = {
            "reused_source_costs": json.loads(
                (project_path(config["development_run"]) / "costs.json").read_text()
            ),
            "parents": {},
        }
        for entry, inputs, screen in loaded:
            print(f"{entry['parent']}: prefix cache", flush=True)
            folder = output / entry["id"]
            folder.mkdir()
            torch.save(inputs, folder / "inputs.pt")
            write_json(folder / "scorer_validation.json", screen)
            before = time.perf_counter()
            tree = PrefixTree(
                SequenceContext(inputs["reference"], 1), inputs["sequences"]
            )
            features, logits = cache_tree(backbone, tree, config["feature_batch_size"])
            setup["parents"][entry["id"]] = {
                "prefix_cache_seconds": time.perf_counter() - before,
                "prefix_count": len(tree.prefixes),
            }
            scorer = inputs["scorers"]["mlp"]
            scores = (scorer["predictions"] - scorer["reference_prediction"]) / inputs[
                "label_scale"
            ]
            for arm in config["arms"]:
                destination = folder / arm
                destination.mkdir()
                adjusted = (
                    logits
                    if arm == "original"
                    else logits + descendant_log_counts(tree).float()
                )
                torch.save(
                    {
                        "context": asdict(tree.context),
                        "sequences": tree.sequences,
                        "features": features,
                        "logits": adjusted,
                        "scores": scores,
                    },
                    destination / "task.pt",
                )
            # Check prefix-vs-teacher-forced consistency before spending real fit time.
            task, _ = load_task(folder / "original")
            expected = candidate_priors(
                inputs["sequences"], inputs["reference"], inputs["path_logits"]
            )["local"]
            torch.testing.assert_close(
                task.base_log_probabilities.double().log_softmax(0),
                expected,
                atol=2e-4,
                rtol=2e-4,
            )
        setup["new_backbone_work"] = backbone.work
        if any(parameter.requires_grad for parameter in backbone.model.parameters()):
            raise RuntimeError("Backbone must remain frozen")
        del backbone
        if not smoke:
            torch.cuda.empty_cache()
        write_json(output / "setup_costs.json", setup)
        for entry, _, screen in loaded:
            for arm in config["arms"]:
                destination = output / entry["id"] / arm
                costs = {}
                if screen["gate_passed"]:
                    task, _ = load_task(destination, device)
                    for seed in config["seeds"]:
                        print(
                            f"{entry['parent']} {arm}: guidance seed {seed}", flush=True
                        )
                        folder = destination / f"seed-{seed}"
                        folder.mkdir()
                        head, log_z, history, gfn_cost = train_tb(
                            task,
                            config["training"],
                            seed,
                            progress_recorder(folder, "gfn"),
                        )
                        save_head(
                            folder / "gfn.pt",
                            head,
                            config["training"]["hidden_size"],
                            log_z,
                        )
                        write_json(folder / "gfn_history.json", history)
                        head, history, supervised_cost = train_supervised(
                            task,
                            config["supervised"],
                            seed,
                            progress_recorder(folder, "supervised"),
                        )
                        save_head(
                            folder / "supervised.pt",
                            head,
                            config["supervised"]["hidden_size"],
                        )
                        write_json(folder / "supervised_history.json", history)
                        costs[str(seed)] = {
                            "gfn": gfn_cost,
                            "supervised": supervised_cost,
                        }
                        write_json(destination / "costs.json", costs)
                write_json(destination / "costs.json", costs)
                write_json(
                    destination / "fit_status.json",
                    {
                        "status": "complete"
                        if screen["gate_passed"]
                        else "scorer_gate_failed"
                    },
                )
        write_json(
            output / "fits_complete.json",
            {
                name: file_hash(output / name)
                for name in fit_file_names(output, entries, config)
            },
        )
        evaluate_diversity(output, output / "evaluation")
        write_json(
            output / "status.json",
            {
                "status": "smoke_complete" if smoke else "complete",
                "smoke": smoke,
                "development_only": True,
                "total_seconds": time.perf_counter() - started,
            },
        )
        print(
            f"Development ablation complete: {output / 'evaluation/report.md'}",
            flush=True,
        )
    except (Exception, KeyboardInterrupt) as error:
        write_json(
            output / "status.json",
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "smoke": smoke,
                "error": str(error),
            },
        )
        raise
