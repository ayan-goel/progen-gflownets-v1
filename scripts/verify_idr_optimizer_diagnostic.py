"""Reconstruct saved optimizer results on CPU; no fitting or pretrained execution."""

import argparse
import copy
import json
from pathlib import Path

import torch
from condition_idr_exact_cpu import ConditionedHead, equivalence
from diagnose_idr_updates_cpu import load_task, metrics, objective

from stability_gfn.artifacts import file_hash, write_json


def verify(run, controls, conditioned):
    """Compare saved final parameters with their reported exact finite-task metrics."""
    trees, initial, _, _ = load_task(run)
    checked = []
    maximum_metric_error = 0.0
    control_results = json.loads((controls / "results.json").read_text())
    for root in (controls, conditioned):
        provenance = json.loads((root / "provenance.json").read_text())
        if provenance["source_manifest_sha256"] != file_hash(run / "generation.json"):
            raise ValueError("Source manifest differs")
        for name, expected in provenance["source_files"].items():
            if file_hash(root / "source" / name) != expected:
                raise ValueError("Captured analysis source differs")
    for arm, result in control_results.items():
        saved = torch.load(
            controls / f"{arm}.pt", map_location="cpu", weights_only=True
        )
        head = copy.deepcopy(initial)
        head.load_state_dict(saved["head"])
        current = metrics(trees, head)
        if current["by_prompt"] != result["by_prompt"]:
            raise ValueError(f"Exact metrics disagree: {arm}")
        loss = float(objective(trees, head, saved["log_z"]).detach())
        if loss != result["fixed_tb_loss"]:
            raise ValueError(f"Exact loss disagrees: {arm}")
        checked.append(arm)
    result = json.loads((conditioned / "results.json").read_text())
    saved = torch.load(
        conditioned / "conditioned.pt", map_location="cpu", weights_only=True
    )
    head = ConditionedHead(initial, trees)
    head.load_state_dict(saved["head"])
    current = metrics(trees, head)
    if current["by_prompt"] != result["by_prompt"]:
        raise ValueError("Conditioned final metrics disagree")
    loss = float(objective(trees, head, saved["log_z"]).detach())
    if loss != result["fixed_tb_loss"]:
        raise ValueError("Conditioned final objective disagrees")
    folded = copy.deepcopy(initial)
    folded.load_state_dict(
        torch.load(conditioned / "folded.pt", map_location="cpu", weights_only=True)[
            "head"
        ]
    )
    fold_check = equivalence(trees, head, folded)
    if fold_check != result["fold_equivalence"]:
        raise ValueError("Saved folded-head equivalence disagrees")
    maximum_metric_error = max(
        abs(metrics(trees, folded)["mean_tv"] - result["mean_tv"]), maximum_metric_error
    )
    checked.append("conditioned_lbfgs")
    assert not torch.cuda.is_initialized()
    return {
        "exact_final_metrics_and_losses_reproduced": checked,
        "folded_mean_tv_difference": maximum_metric_error,
        "folded_maximum_prompt_policy_tv": fold_check[
            "maximum_prompt_tv_between_policies"
        ],
        "source_and_snapshot_hashes_verified": True,
        "cuda_initialized": False,
        "scope": "Saved-head CPU arithmetic only; no training, ProGen or scorer execution. Finite selected support, not open-generation performance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = verify(
        Path("runs/idr-continuous-v1"),
        Path("runs/idr-exact-optimizer-v1"),
        Path("runs/idr-exact-conditioned-v1"),
    )
    result["verification_script_sha256"] = file_hash(Path(__file__))
    result["artifacts"] = {
        str(p): file_hash(p)
        for p in [
            Path("runs/idr-update-audit-v1/audit.json"),
            Path("runs/idr-exact-optimizer-v1/results.json"),
            Path("runs/idr-exact-conditioned-v1/results.json"),
            Path("runs/idr-exact-conditioned-v1/initialization.json"),
        ]
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
