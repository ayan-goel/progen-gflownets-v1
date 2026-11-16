"""Construct an exact-bank head algebraically to separate capacity from fitting.

This is a post-hoc finite-state memorization witness, not a deployable model.
No gradient updates, pretrained execution, scorer inference or GPU access occur.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from diagnose_idr_exact_cpu import FiniteBankTree, distribution_metrics

from stability_gfn.artifacts import file_hash, new_run, write_json
from stability_gfn.idr.acceptance_experiment import verify_manifest
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.context_experiment import DirectContextHead
from stability_gfn.idr.continuous_reward import continuous_reward


def construct_head(initial, trees):
    """Use twenty hidden units to interpolate desired corrections on finite states.

    Positive SiLU outputs have a unique positive preimage. Invert them, then solve
    the hidden-layer linear equations on the fixed normalized features/context.
    The existing 128-hidden-unit architecture contains this construction. A failed
    numerical solve is inconclusive; a verified fit is a finite-state capacity witness.
    """
    head = copy.deepcopy(initial)
    design_parts, desired_parts = [], []
    with torch.no_grad():
        for tree in trees:
            normalized = head.normalization(tree.features).double()
            combined = torch.cat([normalized, tree.context.double()], dim=1)
            design_parts.append(
                torch.cat([combined, torch.ones(len(combined), 1)], dim=1)
            )
            desired = torch.zeros_like(tree.logits, dtype=torch.float64)
            for node, groups in enumerate(tree.edges):
                masses = {
                    action: tree.target[indices].sum()
                    for action, indices in groups.items()
                }
                total = sum(masses.values())
                legal = list(groups)
                for action, mass in masses.items():
                    desired[node, action] = (mass / total).log() - tree.logits[
                        node, action
                    ].double()
                # Softmax is unchanged by a constant shared by legal actions.
                desired[node, legal] -= desired[node, legal].mean()
            desired_parts.append(desired)
        design = torch.cat(design_parts)
        desired = torch.cat(desired_parts)
        offset = 1.0 - desired.min()
        positive = desired + offset
        lower = torch.zeros_like(positive)
        upper = positive + 2.0
        for _ in range(60):
            midpoint = (lower + upper) / 2
            too_low = torch.nn.functional.silu(midpoint) < positive
            lower = torch.where(too_low, midpoint, lower)
            upper = torch.where(too_low, upper, midpoint)
        hidden_targets = (lower + upper) / 2
        solution = torch.linalg.lstsq(
            design, hidden_targets, rcond=1e-12, driver="gelsd"
        )
        head.hidden.weight.zero_()
        head.hidden.bias.zero_()
        head.hidden.weight[:20] = solution.solution[:-1].T.float()
        head.hidden.bias[:20] = solution.solution[-1].float()
        head.output.weight.zero_()
        head.output.weight[:, :20] = torch.eye(20)
        head.output.bias.fill_(-float(offset))
        head.task_weight.zero_()
        diagnostics = {
            "states": len(design),
            "design_columns": design.shape[1],
            "numerical_rank": int(solution.rank),
            "rcond": 1e-12,
            "minimum_singular_value": float(solution.singular_values.min()),
            "condition_number": float(
                solution.singular_values.max() / solution.singular_values.min()
            ),
            "max_double_linear_residual": float(
                (design @ solution.solution - hidden_targets).abs().max()
            ),
            "hidden_weight_norm": float(head.hidden.weight.norm()),
            "initial_hidden_weight_norm": float(initial.hidden.weight.norm()),
            "silu_positive_offset": float(offset),
        }
    return head, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/idr-continuous-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_manifest(args.run)
    bank = json.loads((args.run / "bank.json").read_text())
    cache = torch.load(
        args.run / "bank_cache.pt", map_location="cpu", weights_only=True
    )
    saved = torch.load(args.run / "initial.pt", map_location="cpu", weights_only=True)
    reward = AcceptanceRewardConfig(
        **json.loads((args.run / "reward.json").read_text())
    )
    initial = DirectContextHead(
        cache["features"].shape[-1], cache["context"].shape[-1], 20, 128
    )
    initial.load_state_dict(saved["head"])
    trees = []
    for prompt in sorted({r["prompt_index"] for r in bank}):
        indices = [
            i
            for i, r in enumerate(bank)
            if r["prompt_index"] == prompt and r["split"] == "train"
        ]
        rows = [bank[i] for i in indices]
        tree = FiniteBankTree(
            rows, {key: value[indices] for key, value in cache.items()}
        )
        values = continuous_reward(
            [r["tail"] for r in rows],
            [r["isolated_tail_v3_scores"] for r in rows],
            reward,
            2.5,
        )
        tree.target = (tree.log_prior + values["log_factor"]).softmax(0)
        trees.append(tree)
    head, result = construct_head(initial, trees)
    with torch.no_grad():
        result["by_prompt"] = {
            t.rows[0]["prompt_id"]: distribution_metrics(
                t.log_probabilities(head), t.target, t.passed
            )
            for t in trees
        }
    result["mean_tv"] = sum(r["tv"] for r in result["by_prompt"].values()) / len(trees)
    result["interpretation"] = (
        "Algebraic finite-state interpolation; no generalization, open-generation or optimizer-reachability guarantee. No constructed weights are deployed or saved."
    )
    output = new_run(
        args.output,
        {
            "source": str(args.run.resolve()),
            "method": "positive_silu_inversion_and_linear_interpolation",
        },
    )
    write_json(output / "capacity.json", result)
    write_json(
        output / "provenance.json",
        {
            "source_manifest_sha256": file_hash(args.run / "generation.json"),
            "script_sha256": file_hash(Path(__file__)),
            "tree_script_sha256": file_hash(
                Path(__file__).with_name("diagnose_idr_exact_cpu.py")
            ),
            "torch": torch.__version__,
        },
    )
    verify_manifest(args.run)
    assert not torch.cuda.is_initialized()
    write_json(
        output / "status.json", {"status": "complete", "cuda_initialized": False}
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
