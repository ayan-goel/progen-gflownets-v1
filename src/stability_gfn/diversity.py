"""Development-only concentration diagnostics and fixed-budget unique selection."""

import json
import shutil
import statistics
import time
from pathlib import Path

import torch

from .artifacts import file_hash, write_json
from .environment import PrefixTree, SequenceContext
from .target_diagnostics import (
    candidate_priors,
    probability_metrics,
    support_statistics,
)


def descendant_log_counts(tree):
    """Log number of terminal sequences below each legal edge, without any labels.

    With uniform model logits these counts telescope to a uniform leaf law. This
    changes the prior: it is not conditioning the original language-model law.
    Illegal entries are zero here and remain excluded by the task's legal mask.
    """
    paths = torch.as_tensor(tree.paths)
    actions = torch.as_tensor(tree.actions)
    width = len(tree.context.alphabet)
    counts = torch.zeros(len(tree.prefixes) * width, dtype=torch.float64)
    counts.scatter_add_(
        0, (paths * width + actions).flatten(), torch.ones(paths.numel()).double()
    )
    return counts.reshape(len(tree.prefixes), width).clamp_min(1).log()


def corrected_path_prior(tree, path_logits):
    """Compute the descendant-count prior from existing causal path logits on CPU."""
    paths, actions = torch.as_tensor(tree.paths), torch.as_tensor(tree.actions)
    legal = torch.as_tensor(tree.legal)[paths]
    corrected = path_logits.double() + descendant_log_counts(tree)[paths]
    conditional = corrected.masked_fill(~legal, -torch.inf).log_softmax(-1)
    log_probabilities = conditional.gather(-1, actions[..., None]).squeeze(-1).sum(-1)
    if abs(log_probabilities.exp().sum().item() - 1) > 1e-4:
        raise ValueError("Corrected path prior does not normalize")
    return log_probabilities.log_softmax(0)


def select_unique(proposals, predictions, count):
    """Deduplicate before ranking; ties retain first proposal order; never refill.

    Only identities and predictions enter this rule, never measured outcomes.
    Return fewer than count designs if the fixed proposal batch lacks diversity.
    """
    distinct = list(dict.fromkeys(proposals.tolist()))
    ranked = sorted(distinct, key=lambda index: -float(predictions[index]))
    return torch.tensor(ranked[:count], dtype=torch.long)


def selection_metrics(proposals, predictions, labels, distances, selected_count):
    """Compare historical repeated slots with a separately declared unique protocol."""
    repeated = proposals[
        torch.argsort(predictions[proposals], descending=True, stable=True)[
            :selected_count
        ]
    ]
    unique = select_unique(proposals, predictions, selected_count)
    metrics = {}
    for name, chosen in (("slots", repeated), ("unique", unique)):
        count = len(chosen)
        metrics[name] = {
            "selected_indices": chosen.tolist(),
            "count": count,
            "distinct_count": len(chosen.unique()),
            "fill_fraction": count / selected_count,
            "measured_mean": labels[chosen].mean().item(),
            "predicted_mean": predictions[chosen].mean().item(),
            "pairwise_hamming": distances[chosen][:, chosen].sum().item()
            / (count * (count - 1))
            if count > 1
            else 0.0,
        }
    return metrics


def policy_report(probabilities, inputs, settings, sampling_seed):
    """Exact law and repeated finite-budget batches; repeats measure sampling noise.

    Categorical draws use an enumerated law, including for learned policies. Their
    timings do not represent online trajectory decoding. That cost is timed
    separately by the experiment evaluator. Repeats are not biological replicates.
    """
    probabilities = probabilities.double()
    if (
        not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or abs(probabilities.sum().item() - 1) > 1e-4
    ):
        raise ValueError("Need a normalized finite policy")
    predictions = inputs["scorers"]["mlp"]["predictions"].double()
    labels = inputs["labels"].double()
    positions, distances = support_statistics(inputs["sequences"], inputs["reference"])
    exact = probability_metrics(
        probabilities,
        labels,
        inputs["reference_label"],
        positions,
        distances,
        settings["raw_count"],
    )
    exact["predicted_mean"] = (probabilities @ predictions).item()
    exact["position_mass"] = {
        str(position): probabilities[positions == position].sum().item()
        for position in range(1, len(inputs["reference"]) + 1)
    }
    samples = []
    for repeat in range(settings["repeats"]):
        seed = sampling_seed + repeat
        proposals = torch.multinomial(
            probabilities,
            settings["raw_count"],
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        samples.append(
            {
                "seed": seed,
                "raw_indices": proposals.tolist(),
                "raw_unique_count": len(proposals.unique()),
                **selection_metrics(
                    proposals,
                    predictions,
                    labels,
                    distances,
                    settings["selected_count"],
                ),
            }
        )
    aggregate = {}
    for protocol in ("slots", "unique"):
        aggregate[protocol] = {}
        for metric in (
            "count",
            "distinct_count",
            "fill_fraction",
            "measured_mean",
            "predicted_mean",
            "pairwise_hamming",
        ):
            values = [sample[protocol][metric] for sample in samples]
            aggregate[protocol][metric] = statistics.mean(values)
            aggregate[protocol][metric + "_sampling_sd"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregate[protocol]["full_batch_rate"] = statistics.mean(
            sample[protocol]["count"] == settings["selected_count"]
            for sample in samples
        )
    return {
        "exact": exact,
        "selection": aggregate,
        "samples": samples,
        "probabilities": probabilities.tolist(),
    }


def read_development_run(run):
    """Read only the original development parents; reject subsets and changed inputs."""
    run = Path(run)
    status = json.loads((run / "status.json").read_text())
    if status["status"] not in ("complete", "smoke_complete"):
        raise ValueError("Need a completed development diagnostic")
    config = json.loads((run / "config.json").read_text())
    if config["experiment"] != "target-diagnostic":
        raise ValueError("Expected a development target diagnostic, not confirmation")
    manifest = json.loads((run / "diagnostic_manifest.json").read_text())
    panel = json.loads((run / "panel_manifest.json").read_text())
    expected = [p["id"] for p in panel["parents"] if p["role"] == "development"]
    if [p["id"] for p in manifest["parents"]] != expected:
        raise ValueError("Must retain every development parent in original order")
    loaded = []
    for entry in manifest["parents"]:
        if entry["role"] != "development":
            raise ValueError("Reserved parent cannot enter development analysis")
        path = run / entry["id"] / "inputs.pt"
        if file_hash(path) != entry["inputs_sha256"]:
            raise ValueError(f"Changed development inputs: {path}")
        inputs = torch.load(path, map_location="cpu", weights_only=True)
        if inputs["sequences"] != sorted(set(inputs["sequences"])):
            raise ValueError("Inputs must be sorted unique identities")
        report = json.loads((run / entry["id"] / "scorer_fits.json").read_text())
        loaded.append((entry, inputs, report["mlp-0"]))
    return config, loaded


def diagnose_diversity(run, output):
    """CPU analysis of saved development predictions; no model load or fitting."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    config, loaded = read_development_run(run)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = {"raw_count": 256, "selected_count": 32, "repeats": 100}
    lines = [
        "# Development diversity and prior diagnostic",
        "",
        "Development evidence only; no GFN fit. Targets are exact references.",
        "100 categorical batches of 256 proposals; up to 32 unique designs, no refill.",
        "",
        "| Parent | Policy | First-three mass | Expected unique /256 | Measured raw | Unique selected count | Unique selected measured |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    result = {"smoke": config["smoke"], "settings": settings, "parents": []}
    for entry, inputs, screen in loaded:
        tree = PrefixTree(SequenceContext(inputs["reference"], 1), inputs["sequences"])
        priors = candidate_priors(
            inputs["sequences"], inputs["reference"], inputs["path_logits"]
        )
        corrected = corrected_path_prior(tree, inputs["path_logits"])
        scorer = inputs["scorers"]["mlp"]
        scores = (
            scorer["predictions"].double() - scorer["reference_prediction"]
        ) / inputs["label_scale"]
        laws = {
            "uniform_candidates": torch.full_like(scores, 1 / len(scores)),
            "uniform_legal_actions": priors["uniform_legal_actions"].exp(),
            "original_prior": priors["local"].exp(),
            "original_target": (priors["local"] + scores).softmax(0),
            "count_corrected_prior": corrected.exp(),
            "count_corrected_target": (corrected + scores).softmax(0),
        }
        parent = {
            "id": entry["id"],
            "parent": entry["parent"],
            "inputs_sha256": entry["inputs_sha256"],
            "scorer_gate_passed": screen["gate_passed"],
            "policies": {},
        }
        for name, probabilities in laws.items():
            report = policy_report(probabilities, inputs, settings, 200000)
            # Samples are kept in per-policy files; the summary stays readable.
            write_json(output / f"{entry['id']}-{name}.json", report)
            parent["policies"][name] = {
                k: v for k, v in report.items() if k not in ("samples", "probabilities")
            }
            exact, unique = report["exact"], report["selection"]["unique"]
            lines.append(
                f"| {entry['parent']} | {name} | {exact['first3_position_mass']:.4f} | {exact['expected_unique_raw']:.2f} | {exact['measured_mean']:.5f} | {unique['count']:.2f} | {unique['measured_mean']:.5f} |"
            )
        result["parents"].append(parent)
    result["seconds"] = time.perf_counter() - started
    result["source_files"] = {
        str(Path(run) / name): file_hash(Path(run) / name)
        for name in ("diagnostic_manifest.json", "config.json", "provenance.json")
    }
    source = Path(__file__).parent
    shutil.copytree(
        source,
        output / "source/stability_gfn",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    result["implementation_sha256"] = {
        str(path.relative_to(output / "source")): file_hash(path)
        for path in (output / "source").rglob("*")
        if path.is_file()
    }
    write_json(output / "summary.json", result)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def inspect_saved_unique_selection(run, output):
    """Re-select from completed confirmation proposals; historical, never fresh test.

    This post-hoc rule deduplicates all original 256 proposals before ranking.
    It differs from merely deduplicating the old selected 32 repeated slots.
    There are no new proposals, measurements, fits or hyperparameter selections.
    """
    from .confirmation_evaluation import verify_fit_seal

    run, output = Path(run), Path(output)
    status = json.loads((run / "status.json").read_text())
    if status["status"] not in ("complete", "smoke_complete"):
        raise ValueError("Historical unique review requires a completed confirmation")
    config, entries = verify_fit_seal(run)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"posthoc": True, "new_draws": False, "parents": []}
    lines = [
        "# Historical unique-selection review",
        "",
        "Post-hoc reuse of original proposals, not fresh confirmation. Deduplicate before ranking; no refill.",
        "",
        "| Parent | Method | Unique designs | Measured mean | Slots measured mean | Mean fit seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for entry in entries:
        folder = run / entry["id"]
        payload = torch.load(folder / "task.pt", map_location="cpu", weights_only=True)
        label_path = folder / "evaluation_labels.json"
        if file_hash(label_path) != entry["file_sha256"]["test_labels.json"]:
            raise ValueError("Changed saved evaluation labels")
        by_sequence = {
            row["sequence"]: row["stability"]
            for row in json.loads(label_path.read_text())
        }
        labels = torch.tensor(
            [by_sequence[sequence] for sequence in payload["sequences"]],
            dtype=torch.float64,
        )
        _, distances = support_statistics(
            payload["sequences"], payload["context"]["reference"]
        )
        predictions = payload["predictions"].double()
        costs = json.loads((folder / "costs.json").read_text())
        rows = []
        for path in sorted((run / "evaluation" / entry["id"]).glob("seed-*.json")):
            saved = json.loads(path.read_text())
            seed, method = saved["exact"]["seed"], saved["exact"]["method"]
            proposals = torch.tensor(saved["raw_leaf_ids"])
            if len(proposals) != config["evaluation"]["raw_count"]:
                raise ValueError(
                    "Historical proposal budget differs from configured budget"
                )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "source_sha256": file_hash(path),
                    **selection_metrics(
                        proposals,
                        predictions,
                        labels,
                        distances,
                        config["evaluation"]["selected_count"],
                    ),
                    "fit_seconds": costs["seeds"]
                    .get(str(seed), {})
                    .get(method, {})
                    .get("training_seconds", 0),
                }
            )
        summary["parents"].append(
            {"parent": entry["parent"], "rows": rows, "original_costs": costs}
        )
        for method in ("progen", "exact_target", "gfn", "supervised"):
            values = [row for row in rows if row["method"] == method]
            if not values:
                continue
            count = statistics.mean(row["unique"]["count"] for row in values)
            quality = statistics.mean(row["unique"]["measured_mean"] for row in values)
            old_quality = statistics.mean(
                row["slots"]["measured_mean"] for row in values
            )
            fit = statistics.mean(row["fit_seconds"] for row in values)
            lines.append(
                f"| {entry['parent']} | {method} | {count:.2f} | {quality:.5f} | {old_quality:.5f} | {fit:.2f} |"
            )
    summary["source_run"] = str(run)
    summary["fit_seal_sha256"] = file_hash(run / "fits_complete.json")
    summary["implementation_sha256"] = file_hash(Path(__file__))
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary
