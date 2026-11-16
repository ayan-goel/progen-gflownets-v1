"""Exact finite-bank IDR checks using cached states only; never execute ProGen.

The locally masked tree is an explicitly different support/prior from generation.
Forced suffixes are omitted from computation because their log probability is zero.
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import torch

from stability_gfn.artifacts import file_hash, new_run, write_json
from stability_gfn.idr.acceptance_experiment import (
    verify_implementation,
    verify_manifest,
)
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.branch_rate import branch_optimizer, set_rates
from stability_gfn.idr.context_experiment import DirectContextHead
from stability_gfn.idr.continuous_reward import continuous_reward
from stability_gfn.idr.data import ALPHABET
from stability_gfn.idr.replay_data import bank_indices


class FiniteBankTree:
    """One prompt's locally masked trie, retaining only nodes with actual choices."""

    def __init__(self, rows, cache):
        self.rows = rows
        tails = [row["tail"] for row in rows]
        if len(set(tails)) != len(tails) or len({len(t) for t in tails}) != 1:
            raise ValueError("Need distinct equal-length tails per prompt")
        locations = defaultdict(list)
        for leaf, tail in enumerate(tails):
            for depth in range(len(tail)):
                locations[tail[:depth]].append((leaf, depth))
        self.edges = []
        features, contexts, logits, legal, depths = [], [], [], [], []
        leaf_ids, node_ids, actions = [], [], []
        max_feature_difference = 0.0
        for prefix, occurrences in locations.items():
            allowed = sorted({ALPHABET.index(tails[i][d]) for i, d in occurrences})
            if len(allowed) == 1:
                continue
            node = len(features)
            leaf, depth = occurrences[0]
            for name in ("features", "context", "logits"):
                reference = cache[name][leaf, depth]
                values = torch.stack([cache[name][i, d] for i, d in occurrences])
                difference = float((values - reference).abs().max())
                max_feature_difference = max(max_feature_difference, difference)
                if not torch.allclose(
                    values, reference.expand_as(values), atol=0.003, rtol=1e-5
                ):
                    raise ValueError(f"Inconsistent shared prefix {name}: {difference}")
            features.append(cache["features"][leaf, depth])
            contexts.append(cache["context"][leaf, depth])
            logits.append(cache["logits"][leaf, depth])
            mask = torch.zeros(20, dtype=torch.bool)
            mask[allowed] = True
            legal.append(mask)
            depths.append(depth)
            groups = {}
            for action in allowed:
                groups[action] = [
                    i for i, d in occurrences if ALPHABET.index(tails[i][d]) == action
                ]
            self.edges.append(groups)
            for i, d in occurrences:
                leaf_ids.append(i)
                node_ids.append(node)
                actions.append(ALPHABET.index(tails[i][d]))
        if not features:
            raise ValueError("Diagnostic requires at least two leaves")
        self.features = torch.stack(features)
        self.context = torch.stack(contexts)
        self.logits = torch.stack(logits)
        self.legal = torch.stack(legal)
        self.leaf_ids = torch.tensor(leaf_ids)
        self.node_ids = torch.tensor(node_ids)
        self.actions = torch.tensor(actions)
        self.passed = torch.tensor([r["joint_pass"] for r in rows])
        if not self.passed.any() or self.passed.all():
            raise ValueError("Need both reward strata")
        self.log_prior = self.log_probabilities(None).detach().double()
        self.summary = {
            "leaves": len(rows),
            "branching_nodes": len(features),
            "branch_depth_histogram": {
                str(d): depths.count(d) for d in sorted(set(depths))
            },
            "mean_decisions_per_leaf": len(leaf_ids) / len(rows),
            "maximum_shared_cache_difference": max_feature_difference,
        }

    def from_conditionals(self, log_conditionals):
        """Sum only branching log probabilities; all omitted edges are forced."""
        selected = log_conditionals[self.node_ids, self.actions]
        return selected.new_zeros(len(self.rows)).index_add(0, self.leaf_ids, selected)

    def log_probabilities(self, head):
        correction = 0 if head is None else head(self.features, self.context)
        conditionals = (
            (self.logits + correction)
            .masked_fill(~self.legal, -torch.inf)
            .log_softmax(-1)
        )
        return self.from_conditionals(conditionals)

    def oracle_log_probabilities(self, target):
        """Independent bottom-up target masses give exact tree conditionals."""
        conditional = torch.full(self.logits.shape, -torch.inf, dtype=torch.float64)
        for node, groups in enumerate(self.edges):
            masses = {
                action: target[indices].sum() for action, indices in groups.items()
            }
            total = sum(masses.values())
            for action, mass in masses.items():
                conditional[node, action] = (mass / total).log()
        return self.from_conditionals(conditional)


def distribution_metrics(log_q, target, passed):
    probabilities = log_q.detach().double().exp()
    if not torch.allclose(
        probabilities.sum(), torch.tensor(1.0, dtype=torch.float64), atol=2e-5
    ):
        raise ValueError("Tree probabilities do not sum to one")
    return {
        "tv": float((probabilities - target).abs().sum() / 2),
        "joint_success": float(probabilities[passed].sum()),
        "probability_sum": float(probabilities.sum()),
    }


def fit_condition(trees, initial, config, condition):
    """400 bounded CPU updates; no model/scorer execution, checkpoint or sequence generation."""
    head = copy.deepcopy(initial)
    offsets = []
    for tree in trees:
        delta = tree.log_reward - tree.log_probabilities(head).detach().double()
        offsets.append((delta[tree.passed].mean() + delta[~tree.passed].mean()) / 2)
    normalizer = torch.nn.Parameter(torch.stack(offsets).float())
    optimizer = branch_optimizer(head, normalizer, config, 10)
    fresh_rng = torch.Generator().manual_seed(92000)
    replay_rng = torch.Generator().manual_seed(95000)
    all_rows = [
        dict(row, prompt_index=p) for p, tree in enumerate(trees) for row in tree.rows
    ]
    starts = [sum(len(t.rows) for t in trees[:p]) for p in range(len(trees))]
    history, clipped = [], 0
    for step in range(1, config["steps"] + 1):
        set_rates(optimizer, config, 10, step)
        selected = bank_indices(all_rows, len(trees), replay_rng, step)
        losses = []
        for prompt, tree in enumerate(trees):
            log_q = tree.log_probabilities(head).double()
            if condition == "exact_teacher":
                loss = -(tree.target * log_q).sum()
            else:
                residual = normalizer[prompt] + log_q - tree.log_reward
                if condition == "expected_tb":
                    positive = (prompt + step) % 2 == 0
                    stratum = tree.passed == positive
                    # Differentiate residuals only, as in sampled TB updates.
                    weights = (
                        0.5 * log_q.detach().exp()
                        + 0.5 * stratum.double() / stratum.sum()
                    )
                    loss = (weights * residual.square()).sum()
                elif condition == "sampled_tb":
                    fresh = int(
                        torch.multinomial(log_q.detach().exp(), 1, generator=fresh_rng)
                    )
                    stored = selected[prompt] - starts[prompt]
                    loss = (residual[fresh].square() + residual[stored].square()) / 2
                else:
                    raise ValueError(condition)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [*head.parameters(), normalizer],
            config["gradient_clip"],
            error_if_nonfinite=True,
        )
        clipped += float(norm) > config["gradient_clip"]
        optimizer.step()
        if step == 1 or step % 100 == 0:
            metrics = [
                distribution_metrics(t.log_probabilities(head), t.target, t.passed)
                for t in trees
            ]
            history.append(
                {
                    "step": step,
                    "loss_before_update": float(loss.detach()),
                    "mean_tv_after_update": sum(m["tv"] for m in metrics) / len(trees),
                }
            )
    results = {}
    for prompt, tree in enumerate(trees):
        metrics = distribution_metrics(
            tree.log_probabilities(head), tree.target, tree.passed
        )
        if condition != "exact_teacher":
            metrics["normalizer_error"] = float(
                normalizer[prompt].detach() - tree.log_reward.logsumexp(0)
            )
        results[tree.rows[0]["prompt_id"]] = metrics
    return {
        "by_prompt": results,
        "mean_tv": sum(r["tv"] for r in results.values()) / len(results),
        "clipped_fraction": clipped / config["steps"],
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/idr-continuous-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = verify_manifest(args.run)
    provenance = json.loads((args.run / "provenance.json").read_text())
    verify_implementation(provenance)
    bank = json.loads((args.run / "bank.json").read_text())
    cache = torch.load(
        args.run / "bank_cache.pt", map_location="cpu", weights_only=True
    )
    saved = torch.load(args.run / "initial.pt", map_location="cpu", weights_only=True)
    config = json.loads((args.run / "pilot_config.json").read_text())
    reward = AcceptanceRewardConfig(
        **json.loads((args.run / "reward.json").read_text())
    )
    initial = DirectContextHead(
        cache["features"].shape[-1],
        cache["context"].shape[-1],
        20,
        config["hidden_size"],
    )
    initial.load_state_dict(saved["head"])
    trees, targets = [], {}
    for prompt in sorted({r["prompt_index"] for r in bank}):
        indices = [
            i
            for i, row in enumerate(bank)
            if row["prompt_index"] == prompt and row["split"] == "train"
        ]
        rows = [bank[i] for i in indices]
        subset = {key: value[indices] for key, value in cache.items()}
        tree = FiniteBankTree(rows, subset)
        values = continuous_reward(
            [r["tail"] for r in rows],
            [r["isolated_tail_v3_scores"] for r in rows],
            reward,
            2.5,
        )
        if not torch.equal(tree.passed, values["joint_pass"]):
            raise ValueError("Cached pass labels differ")
        tree.log_reward = tree.log_prior + values["log_factor"]
        tree.target = tree.log_reward.softmax(0)
        oracle = tree.oracle_log_probabilities(tree.target)
        oracle_error = float((oracle.exp() - tree.target).abs().max())
        if oracle_error > 1e-10:
            raise ValueError("Oracle reconstruction failed")
        original_log_prior = (
            subset["logits"]
            .double()
            .log_softmax(-1)
            .gather(-1, subset["actions"][..., None])
            .squeeze(-1)
            .sum(-1)
        )
        if not torch.allclose(
            original_log_prior, subset["log_prior"].double(), atol=0.003, rtol=1e-5
        ):
            raise ValueError("Cache likelihood mismatch")
        conditioned_prior = original_log_prior.softmax(0)
        conditioned_target = (original_log_prior + values["log_factor"]).softmax(0)
        previous_target = (tree.log_prior + values["current_log_factor"]).softmax(0)
        targets[rows[0]["prompt_id"]] = dict(
            tree.summary,
            ids=[r["id"] for r in rows],
            prior=distribution_metrics(tree.log_prior, tree.target, tree.passed),
            initial=distribution_metrics(
                tree.log_probabilities(initial), tree.target, tree.passed
            ),
            target_joint=float(tree.target[tree.passed].sum()),
            previous_target_joint=float(previous_target[tree.passed].sum()),
            oracle_max_probability_error=oracle_error,
            original_log_prior_mass=float(original_log_prior.logsumexp(0)),
            globally_conditioned_prior_joint=float(
                conditioned_prior[tree.passed].sum()
            ),
            globally_conditioned_target_joint=float(
                conditioned_target[tree.passed].sum()
            ),
            globally_conditioned_vs_local_prior_tv=float(
                (conditioned_prior - tree.log_prior.exp()).abs().sum() / 2
            ),
        )
        trees.append(tree)
    output = new_run(
        args.output,
        {
            "source": str(args.run.resolve()),
            "steps": config["steps"],
            "support": "locally_masked_training_bank",
            "conditions": ["exact_teacher", "expected_tb", "sampled_tb"],
        },
    )
    write_json(output / "target_audit.json", targets)
    results = {}
    for condition in ("exact_teacher", "expected_tb", "sampled_tb"):
        results[condition] = fit_condition(trees, initial, config, condition)
        write_json(output / "fits.json", results)
        print(condition, results[condition]["mean_tv"], flush=True)
    verify_manifest(args.run)
    assert not torch.cuda.is_initialized()
    write_json(
        output / "provenance.json",
        {
            "source_manifest_sha256": file_hash(args.run / "generation.json"),
            "source_files": manifest["files"],
            "analysis_sha256": file_hash(Path(__file__)),
            "imported_sources": {
                str(p): file_hash(p)
                for p in sorted(Path("src/stability_gfn").rglob("*.py"))
            },
            "torch": torch.__version__,
            "cuda_initialized": False,
            "scope": "Post-hoc finite-bank diagnostic; no held-out generalization or open-generation claim.",
        },
    )
    write_json(output / "status.json", {"status": "complete", "cpu_only": True})


if __name__ == "__main__":
    main()
