# Development diversity diagnostic and two-prior ablation

17 October 2026. User authorized diagnostics, the next fixed experiment, implementation
and CPU verification. Real fitting and all GPU operations remain user-run.
This is **development**, not fresh confirmation. No new held-out labels are opened.

## Decision and evidence

The completed five-parent confirmation establishes accurate target fitting, but
few distinct designs and expensive GFN fitting. We first analyzed the three saved
**development** supports with frozen MLP-0 predictions. A local uniform-logit control
puts 92–93% of probability on mutations in positions 1–3, although only 2.5–3.0% of
candidate identities mutate those positions. Branching alone therefore induces a
strong early-mutation bias. This does not numerically apportion the real ProGen bias.

The proposed correction adds the log number of descendant candidate leaves to each
legal action logit. With equal model logits it exactly produces uniform candidates,
by telescoping child/parent leaf-count ratios. This is an explicitly new prior,
not the original ProGen law conditioned on the support. It needs a finite known
support; it does not solve non-enumerable protein generation. No alpha/beta sweep,
new scorer, bigger head or mutation-budget change accompanies it.

The derivation is ours and the motivating evidence is local:
[CPU diagnostic](../../runs/diversity-diagnostic-v2/report.md) and
[confirmation results](confirmation_results_2026-10-17.md). No external paper is
claimed to establish this correction or its biological usefulness.

## Fixed experiment

- Keep all three original development parents, their original fit/inner/development
  splits and frozen MLP-0 predictions. Preserve failed inner screens and skip heads
  on failing parents. No refitting or choosing a better scorer using these results.
- Two arms: original local prior; descendant-count-corrected local prior. Every
  arm has its own matched zero-guidance baseline and exact predicted target.
  Include uniform-candidate selection as a diagnostic reference in evaluation.
- Keep beta=1, seeds 0/1/2, 20,000 updates per learner, width 128, batch 64,
  head rate .001 and TB normalizer rate .01; reduce rates tenfold after update
  8,000. Supervised reward supervision uses 4,096 rollouts. These are inherited
  engineering choices, not a new optimization or equal-compute comparison.
- Cache frozen ProGen prefix features once per parent, share them across arms;
  corrected logits drive baseline, head initialization, TB reward and supervised
  rollouts consistently. Reuse existing scorer predictions and record original
  feature/scorer costs separately from new prefix caching.
- For every final policy: exact probability metrics, plus 100 categorical batches
  of 256 proposals. Keep historical 32 repeated selection slots **and** a separate
  rule: deduplicate the full 256 first, then take up to 32 by predicted stability.
  Never refill or rank by measurements. Report actual count, fill rate, quality,
  diversity and sampling SD together. Variable-size batches are not equivalent
  32-design outputs. 100 repeats is an engineering noise-control choice, not power.
- Sampling seeds start at 200000 + 1000 * guidance seed, with repeat offsets 0–99.
  Same seed conventions across methods/arms; this does not imply paired trajectories.
- Categorical evaluation of exact learned probabilities is a finite-support
  reference; separately time one actual 256-trajectory sample per policy/seed.
  Charge setup, training, exact evaluation and decoding separately. No amortized
  efficiency or matched-compute claim from fixed equal update counts.
- Save original inputs, source/config hashes, all checkpoints and costs. Evaluate
  all final checkpoints; no test-driven or best-TV selection. Failures stop the
  run visibly; no automatic retries/resume. CPU replay uses saved artifacts only.

## Assessment, fixed before new head training

Report every parent and seed, average seeds within parent, then weight parents
equally. Separate whether the corrected target is useful from whether GFN learns
it. Report corrected-versus-original unique count **and** measured quality,
raw gain versus matched prior and original prior, GFN/supervised TV, and actual cost.
TV <=0.01 for each final GFN fit is a declared engineering approximation check,
not a biological success test; failures remain visible and do not hide results.
No automatic biological pass or scalar quality/diversity tradeoff is invented.

A practical benefit requires showing quality/coverage advantages over each matched
prior plus selection, supervised guidance and the uniform diagnostic, while making
compute tradeoffs explicit. If simple selection suffices, report that outcome.
The known concentrated 1UBQ development case must remain visible. Do not tune this
ablation on the already inspected five-parent confirmation.

## Fresh evaluation and larger supports

This run is explicitly a development ablation. It does not consume or relabel old
confirmation as fresh test data. Before any future confirmation, reserve unused
parents/splits by identity/coverage, audit similarity and prior outcome access, and
freeze the procedure before opening their labels. No current fresh confirmation
set is claimed. Larger mutation distances require separate measured coverage and
predictor validation; predicted multi-mutant scores are not measured outcomes.

## Commands and implementation status

Executable configuration: [diversity.json](../../configs/diversity.json).
Implementation, executed diagnostics, commands and verification are in the
[findings and handoff](diversity_findings_2026-10-17.md).
No new real training was executed while preparing this protocol.
