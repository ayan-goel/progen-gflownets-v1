# Current project status

**Updated 17 October 2026.** The [spec](../spec.md) owns the scientific contract;
the [experiment index](experiments/README.md) preserves dated history.

## What we have accomplished

We have moved from separate ProGen/GFN demonstrations to an integrated pipeline:
keep ProGen2 frozen, fit a stability predictor on Mega-scale measurements, and train
an additional GFN head to steer sequence choices using completed-sequence scores.
Exact toy checks established that the training mechanics work. The first biological
pilot exposed a useful distinction: learning the target accurately does not ensure
that the target improves measured stability. Development diagnostics informed a
fixed five-parent confirmation, which now shows both useful targets and accurate
GFN approximation on the tested supports. See the dated evidence below.

## What remains weak

The current task chooses among roughly 200 measured single-mutant variants per
protein, rather than searching unrestricted protein space. Each stability predictor
uses local labels from its own protein background; this is not zero-shot guidance
on arbitrary proteins. No new proteins were synthesized or experimentally tested.

The guidance repeats a small number of variants. Its intended distribution is
already concentrated, including substantial weight near the beginning of the
sequence. The new development control demonstrates a branching-induced component, but the
proposed remedy has not yet been tested with newly trained guidance heads. Supervised guidance is much cheaper and achieves similar stability;
selection comparisons are mixed. Thus the evidence supports a functioning guidance
mechanism, while a compelling practical advantage remains an open research question.

## Latest completed experiment

**Five-parent confirmation is complete and reviewed.** All five scorer screens and
all 15 GFN/supervised seed pairs finished. Runtime: 4 h 8 min. Saved CPU evaluation
reproduces all 67 scientific output files byte-for-byte; sealed fits and source
snapshots verify. The earlier in-progress timing estimate is superseded.

[Result analysis](experiments/confirmation_results_2026-10-17.md) ·
[Original report](../runs/confirmation-v1/evaluation/report.md) ·
[Verification audit](verification/confirmation-result-analysis.json)

- GFN exact expected measured gain over constrained ProGen: +0.14461 kcal/mol,
  positive on every parent and final seed. Exact target: +0.14466.
- GFN mean TV to target: 0.000368; mechanism approximation is strong.
- Supervised gain: +0.13471, using about 1/22 of the GFN fitting time under these budgets.
- Selected GFN beats generate-and-select on three parents, loses on one, ties on
  one; average +0.05888 kcal/mol. These are sampled, duplicate-weighted batches.
- GFN averages only 12.15 unique raw candidates in 256 draws and 3.27 unique designs
  in 32 selected slots. This is not unrestricted generation or wet-lab validation.

## Current handoff: development diversity ablation

The planned CPU diagnosis is complete and the next experiment is implemented.
[Findings and commands](experiments/diversity_findings_2026-10-17.md) ·
[Fixed protocol](experiments/diversity_plan_2026-10-17.md)

- Uniform local action choices heavily favor early mutations on these supports;
  multiplying action weights by descendant candidate counts removes that effect
  in the uniform-logit control. This is a declared alternate prior, not a repair
  to the old experiment or the original ProGen conditional distribution.
- On development data, corrected exact targets raise expected distinct raw counts
  from roughly 12/12/13 to 105/14/50. Raw measured stability decreases on one parent
  while rising on two. Uniform candidate selection remains a strong comparator.
- New selection deduplicates all 256 proposals, then chooses up to 32 predicted-best
  distinct sequences without refill. Count and quality are reported together.
- The prepared GPU run holds scorers, support and learner settings fixed while
  comparing both priors across all three development parents and three seeds.
  It trains GFN/supervised heads and automatically evaluates all final policies.
- All 41 synthetic CPU tests and lint/format checks pass. CPU readiness verifies
  source/checkpoint hashes; CUDA was not initialized. The user has now launched the real ablation; outcomes have not been reviewed.

**Current stage:** the user has launched the development ablation. Results have
not yet been reviewed. After completion, compare all parents and seeds using the
fixed protocol before selecting any new procedure.

This is development, not fresh confirmation. Afterward inspect count/quality/cost
tradeoffs and target approximation before selecting any new procedure. Reserve
truly unused evaluation data before a later confirmation. Larger mutation spaces
still require measured coverage and scorer validation; no broader generation claim
is introduced. Historical confirmation outcomes stay intact.

## Earlier evidence

- [Exact mechanism gate](experiments/exact_pass_and_biological_pilot_2026-10-15.md): three toy seeds pass TV <0.05.
- [First biological pilot](experiments/biological_first_results_2026-10-15.md): good target fit without measured gain on that support.
- [Development diagnostic](experiments/target_diagnostic_results_2026-10-15.md): positive target gains on three development supports motivated confirmation.
- [Verification index](verification/README.md): historical 37-test CPU implementation check and later analysis audits.

The original pilot and development diagnostic used different supports/splits;
later positive evidence does not rewrite the original negative result.
