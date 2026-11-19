# Full-support IDR transfer pilot — 16 November 2026

## Question and scope

Does the [successful finite-bank TB repair](idr_variance_repair_2026-11-16.md)
improve fresh IDR generation when every position allows all 20 canonical residues?
Run one development seed with the same eight 16-residue prefixes, 64-residue tails,
frozen ProGen2-medium and continuous reward (penalty 2.5). Reuse the original
supervised starting head and the 408 training trajectories from
`runs/idr-continuous-v1`; exclude its 95 holdout rows from conditioning and fitting.
No new predictor or supervised fit. All real execution/GPU work remains user-run.

Compare unchanged starting head, ordinary fixed-bank TB (`plain_tb`), and corrected
fixed-bank TB (`svrg_tb`). Both trained heads receive identical initialization,
fixed sampling proposal and sampled trajectory indices. This is an offline TB
pilot: no new rollouts are added during training. Fresh unrestricted evaluation
is therefore essential to expose bank coverage and generalization failures.

## Probability and reward contract

Use the cached ProGen logits at **every** tail position, normalized over all 20
residues. No tree masks, forced residues, or renormalization of the learned policy
over bank membership. The log terminal reward is full-support ProGen log probability
plus the unchanged continuous log property factor. Original cached prior scores
must reconstruct from logits/actions before training. Exact global target TV and
partition sums are unavailable; do not report finite-bank softmax TV as global fit.
Matching residuals on observed paths does not identify the probability assigned
outside the bank: the learned normalizer can compensate for common changes in
bank probability. This is a transfer test, not a guarantee from offline TB alone.

Per prompt, the fixed training proposal is half the initial policy probabilities
renormalized **only for selecting bank examples**, plus one-quarter uniform passing
and one-quarter uniform failing examples. This keeps training support across the
bank; it is not a restriction on generation. Initialize each log-normalizer using
the mean reward-minus-initial-log-policy offset with equal pass/fail weight.

## Conditioning adaptation and safeguards

The earlier transform assumed fewer independent states than input dimensions.
Full trajectories contain repeated/dependent states, so construct the uncentered
second-moment matrix of augmented hidden inputs (initial LayerNorm features,
context, and constant bias coordinate), using training rows only. Weight cached
trajectory-position occurrences equally. Eigen-directions with eigenvalue above
1e-10 times the largest eigenvalue use inverse-square-root scaling capped at 32;
null directions retain scale 1. The cap is a preselected safeguard against
amplifying poorly supported directions, not an outcome-tuned setting. This is a
necessary transfer adaptation, not an exact reproduction of the earlier transform.

Apply the invertible transform and inverse-transform the initial affine weights,
so the initial function/model class/parameter count remain unchanged. LayerNorm
stays trainable. Fold the transform into effective affine weights during forward
computation and into an ordinary head for generation. Check initial and final
cached logit/log-probability equivalence; abort on nonfinite values or material
mismatch. Report initial eigen/scaling diagnostics and input amplification on
training and freshly generated states. This measures distribution shift, not a
proof that unseen-state extrapolation is safe.

## Fixed budget and rationale

- 400 optimizer updates for each trained arm; one seed, bank RNG 98000.
- 32 sampled trajectories per prompt/update (256 total), with replacement.
- Head/direct/normalizer learning rates 1e-4/1e-3/1e-2, global clipping at 1,
  rates multiplied by 0.1 after step 300. Fresh Adam per arm, no weight decay.
- Corrected arm refreshes a snapshot/full weighted gradient every 20 updates;
  current sampled minus snapshot sampled plus full snapshot gradient, then clip
  once and step. Normalizers participate in every term.
- Microbatch 16 trajectories, accumulating globally weighted gradients before
  clipping/stepping. This controls memory without changing objective weighting.
- Ordinary arm: 102,400 sampled occurrences. Corrected arm: the same plus 102,400
  snapshot sampled occurrences and 8,160 full-bank occurrences. Report extra
  head-gradient work; comparisons are matched updates/samples, not equal compute.
- Evaluate 128 fresh tails per prompt per arm: 1,024 per arm, 3,072 total. Common
  fresh evaluation RNG 113000, different from completed experiments. All arms
  generate anew with no filtering, reranking or rejection. Final checkpoints only.

## Evaluation and decision

Primary: raw joint charge/V3-disorder success and joint success excluding exact
copies of the common training bank. Report charge and predicted-disorder rates,
unique-tail fraction, pairwise identity/repetition descriptors, nearest bank
identity and exact-copy fraction, by prompt and overall. V2 remains sensitivity,
not independent biological validation. Also report full-support TB residuals on
bank and fresh outputs, and fraction of generated decision prefixes absent from
the corresponding prompt's training bank. Bank fit is a diagnostic, not target TV.

A practical progression screen requires corrected TB to exceed **both** controls
by at least 5 percentage points on raw and common-bank-noncopy joint success, with
no more than 5 points of uniqueness loss. This reuses the previous generation
screen; it is not statistical significance. Inspect per-prompt failures and
conditioning amplification. No automatic extension, tuning or checkpoint selection.
If bank fit improves without fresh success, coverage/extrapolation is the next
issue. If correction fails even on full-path bank fitting, revisit the full-path
training adaptation before attributing the failure to biology or the reward.

Persist self-contained generation/scoring inputs, configs, checkpoints, source
hashes, paired training logs, conditioning checks and costs. Support read-only
preflight and scoring recovery without retraining/ProGen execution. Existing runs
are protected; historical source modules and completed diagnostic artifacts remain
unchanged. New transfer modules and CLI commands implement this separate pilot.

Continued supervision previously gave 16.11% joint success, but is not an arm in
this matched mechanism test. Its historical score is not a fresh comparison;
passing this pilot would not establish superiority over all existing methods.

## Implementation verification before the run

Before real execution, twenty-nine focused
[CPU tests](../verification/idr-transfer-tests-2026-11-16.txt) pass, including six
transfer tests and earlier reward/diagnostic regression checks. They verify
full-support probability normalization, function-preserving conditioning/folding,
globally weighted microbatch gradients, unbiased raw gradient correction and
exactness at the snapshot, source/frozen-backbone preservation, matched draws,
fresh generation and self-contained scoring recovery. Synthetic models and a
fixture disorder scorer are used; this is correctness evidence, not efficacy.

The [read-only real-input preflight](../verification/idr-transfer-preflight-2026-11-16.json)
passes with 408 training rows, eight prompts and the pinned source/checkpoint.
No real model/scorer inference, fitting or GPU execution occurred during implementation. Real training
and generation use the user's selected GPU; conditioning setup and final scoring
run on CPU inside their command.

From the project root, after selecting `CUDA_VISIBLE_DEVICES`:

```bash
source .venv/bin/activate
scripts/run.sh idr-transfer --output runs/idr-transfer-v1 --device cuda
```

This runs both fits and all fresh generation/evaluation. The
[runbook](../runbook.md#full-support-idr-transfer-pilot--completed) also documents
preflight and recovery after completed generation. Read `report.md` and
`decision.json` before proposing another run; no automatic extension is authorized.

## Sources

The linked local repair owns the empirical motivation. The gradient correction
follows [Johnson & Zhang (2013)](https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf);
its convex guarantees do not apply to this head/Adam combination. Conditioning
regularization, pilot budget and practical screen are explicitly engineering
choices. No new biological property claim is introduced.


## Completed results — 16 November 2026

The user completed `runs/idr-transfer-v1` in **20 min 33 s**. Both 400-update
fits and all 3,072 fresh outputs completed. The predeclared practical screen
**fails**. These are one-seed development results, not a multi-seed significance
claim. No run artifacts, checkpoints or scientific settings were changed during
analysis.

| Head | Charge pass | Predicted disorder pass (V3) | Joint pass | Unique tails | Exact bank copies |
|---|---:|---:|---:|---:|---:|
| Unchanged supervised start | 136/1,024 (13.28%) | 799/1,024 (78.03%) | 87/1,024 (8.50%) | 99.80% | 0 |
| Ordinary offline TB | 131/1,024 (12.79%) | 665/1,024 (64.94%) | 66/1,024 (6.45%) | 98.14% | 0 |
| Variance-reduced offline TB | 133/1,024 (12.99%) | 679/1,024 (66.31%) | 71/1,024 (6.93%) | 98.24% | 0 |

Corrected TB is 0.49 percentage points above ordinary TB and 1.56 points below
initialization. It misses the required +5-point improvement over both controls.
Noncopy joint rates are identical to raw rates. V2 joint success is likewise
9.18%, 6.74%, and 6.84%; this sensitivity check does not reverse the conclusion.
Predicted disorder drops substantially while charge pass is roughly unchanged.
Mean continuous log property factor also worsens: −5.973 unchanged, −6.643 ordinary,
−6.493 corrected (higher is better). This is not just a hard-threshold artifact.

### Fitting works on the bank, but generation does not improve

Full-path bank TB RMSE drops from **3.7696** to **0.001255** for ordinary TB and
**0.001729** for corrected TB. Proposal-weighted final RMSE is 0.000777 and
0.001253 respectively. Both methods fit the observed paths almost exactly;
the correction gives no useful additional final-fit advantage here. Unlike the
masked finite task, ordinary TB also succeeds on this full-path training bank.
The bank gate therefore does not justify more steps or a variance-only follow-up.

A post-hoc CPU check applies the saved folded heads to the **same 95 cached
holdout trajectories**, excluded from transfer fitting/conditioning. These were
previously available diagnostic examples, not a newly sealed confirmation set.
No refitting or pretrained/scorer inference occurs. Holdout TB RMSE is **3.9810 →
2.5702 ordinary / 2.6618 corrected**; within-prompt centered RMSE is **1.9966 →
1.4434 / 1.4508**. Thus some learning transfers, but a large training/holdout gap
remains. Do not describe this as zero generalization or successful target matching.

On each arm's own fresh samples, TB RMSE is 7.318 unchanged, 7.160 ordinary and
7.077 corrected; prompt-centered RMSE remains 4.584, 4.129 and 4.162. These samples
differ between policies, unlike the common holdout check. The discrepancy is not
only a common normalizer offset. About 92.0% of trained-arm decision prefixes are
absent from the cached training trajectories. Such exact-prefix novelty is
expected in sequence generation; it documents limited coverage, not by itself
proof of the cause of failure.

Corrected joint yield improves on initialization for three prompts and worsens
for five. Predicted disorder declines for seven and ties for one. RPB1/DP00181 has
only 1/128 corrected joint passes and 110/128 unique tails, versus 6/128 and
126/128 initially. It accounts for the corrected arm's duplicates, but it does
not explain the widespread disorder decline. Zero exact bank copies do not rule
out poor extrapolation, motif repetition or reward-model exploitation.

### Numerical checks, costs and interpretation

Initial conditioning preserves logits to 4.2e-6 and full-path log probability to
9.2e-5. The 1,550-dimensional transform has 1,548 active directions and reaches
the declared gain cap. Initial training-input amplification p99 is 2.40;
corrected fresh-state p99 is 3.43 and maximum 9.94. This shows a distribution shift,
not proof that conditioning caused the failure.

Ordinary/corrected fits take 27.6/56.0 seconds, with gradients clipped on
76.5%/79.75% of updates. Generation takes about 18 min 38 s across all arms.
Corrected fitting uses 212,960 trajectory-gradient occurrences versus 102,400
ordinary; it does not establish a compute advantage. Frozen ProGen is unchanged.

The [saved-artifact audit](../verification/idr-transfer-results-2026-11-16.json)
checks all 43 manifest files and implementation provenance, reconstructs rewards
from saved disorder scores and sequence descriptors, reproduces aggregate metrics
and the decision, verifies paired draws/schedules, and reconstructs training
likelihoods from saved heads on CPU before the common-holdout check. No GPU,
pretrained inference, new disorder scoring or training was run during analysis.

### Decision

Stop this bounded transfer pilot. The CPU repair remains a valid finite-task
result, but it has **not delivered useful unrestricted IDR guidance**. The immediate
problem is the gap between near-exact offline path fitting and useful generated
outputs. The results support investigating training coverage and generalization;
they do not uniquely identify coverage, conditioning, the head architecture or
the biological proxy as the cause, nor prove the reward target is globally good.

The next design should retain the GFlowNet but explicitly train/evaluate on newly
visited prefixes, with a common held-out trajectory check and raw generation as
separate gates. Compare that intervention against the same fixed-bank control.
A refreshed-data design needs a bounded protocol; earlier online TB also failed,
so adding rollouts alone is not an established fix. Do not automatically run
longer fits, repeat seeds, change the reward again or unfreeze ProGen on this result.
No new experiment is implemented or launched by this analysis.
