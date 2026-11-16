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
are protected; source code and completed diagnostic artifacts remain unchanged.

## Sources

The linked local repair owns the empirical motivation. The gradient correction
follows [Johnson & Zhang (2013)](https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf);
its convex guarantees do not apply to this head/Adam combination. Conditioning
regularization, pilot budget and practical screen are explicitly engineering
choices. No new biological property claim is introduced.
