# Fixed reward-noise sensitivity

**Status: protocol fixed and implementation ready; real run not yet executed.**

## Question and motivation

Tom suggested deliberately adding reward noise to understand guidance reliability.
The [completed clean ridge GFN](ridge_gfn_2026-11-09.md) closely matched its target
and improved expected measured stability by 0.03948 kcal/mol over ProGen on the
original 204 variants. We now separate two possible consequences of worse scores:

1. The corrupted reward defines a biologically worse (or occasionally better) target.
2. The GFN struggles to learn that fixed target.

This study is a **fixed score-corruption sensitivity test**, not fresh query noise,
a refit on noisy labels, or an estimate of actual Mega-scale assay noise. It adds
error on top of existing ridge prediction errors. Gaussian independence and
amplitudes below are transparent engineering choices, not literature-derived
claims about measurement uncertainty. Motivation comes from Tom's transcript and
the completed local experiments; no new literature claim is needed for this test.

## Formulation

For each candidate sequence, perturb its dimensionless ridge score once:

\[
\widetilde{s}_{j,\sigma}(x)=s(x)+\sigma\epsilon_j(x),\qquad
\epsilon_j(x)\sim\mathcal{N}(0,1),\qquad
\widetilde p_{j,\sigma}(x)\propto p_0(x)\exp(\beta\widetilde{s}_{j,\sigma}(x)).
\]

- x: a full sequence in the same 204-variant support.
- s(x): clean ridge prediction minus predicted reference, divided by the fixed
  training-label standard deviation a = 1.177569 kcal/mol.
- j: noise realization, identified by its random seed.
- epsilon_j(x): independent standard-normal draw for that sequence, fixed for all
  visits during training and evaluation. The same vector is reused across amplitudes.
- sigma: chosen dimensionless noise standard deviation.
- s-tilde: corrupted dimensionless score used by the TB objective.
- p_0: the original frozen, locally constrained ProGen probability.
- beta: fixed guidance strength, 1.
- p-tilde: normalized target from those weights; proportionality means divide by
  their sum across the candidate support.

The exponential keeps terminal weights positive. Adding noise to s is equivalent
to multiplying reward by exp(beta*sigma*epsilon), not adding potentially negative
noise to reward itself. No clipping, recentering, rescaling, or perturbation of
experimental evaluation labels is performed. The reference offset is fixed.

## Fixed choices and rationale

| Setting | Choice and reason |
|---|---|
| Clean control | Reuse runs/ridge-gfn-v1 and its three final heads. No duplicate sigma-zero fitting. |
| Amplitudes | Sigma 0.25 and 0.50: 0.2944 and 0.5888 kcal/mol equivalent added prediction SD. Two levels give a small dose comparison at one quarter and one half of training-label SD. Not calibrated assay-error levels. |
| Noise seeds | 100, 101, 102, chosen in advance. Three independent realizations; pair each vector across amplitudes. |
| Training seeds | 0, 1, 2 for every noisy target, fully crossed with noise seeds to separate optimization variability from noise realization. |
| Total new fits | 2 amplitudes × 3 noise seeds × 3 training seeds = 18. Three clean fits reused. |
| Training | Unchanged 20,000-step TB schedule, 128-unit residual head, batch 64, beta 1, clip 1. Head lr 0.001 and log-Z lr 0.01, both decay tenfold after update 8,000. |
| Inputs | Same cached prefix features, prior, legal support and ridge predictions. No ProGen forward pass, predictor fit or change to the training examples. |
| Monitoring | Exact fit to the **noisy** target using predicted scores only, at the inherited log points. No measured-metric stopping or model selection. |
| Checkpoints | Final checkpoint of every fit. No best noise seed, checkpoint or model selection. |
| Budget | About 4.8–5 hours on hardware comparable to the clean run, based on 15.8–16.2 min per fit. Keep the budget unchanged to avoid confounding noise with shorter training. |

The finite realization mean/SD can differ from zero/sigma; preserve and report the
draw, rather than forcing sample moments. A local CPU generator avoids consuming
training RNG. The task's saved ordered sequence list defines correspondence.
Only synthetic noise fixtures are run during implementation; actual noisy-target
outcomes are left for the predeclared user experiment.

## Evaluation and interpretation

Use exact probabilities over the finite support, without additional sampling or
selection. Report every target and every trained policy, including:

- TV/KL of each GFN to its noisy target (optimization fidelity).
- TV from noisy target to clean target (target perturbation).
- Expected measured stability, reference-level success and diversity under each
  distribution (biological utility and concentration).
- Expected **clean ridge-predicted stability**, labeled separately. Noisy scores
  are not presented as measured or calibrated predicted stability.
- Target shift = noisy-target measured expectation minus clean-target expectation.
- Fitting gap = GFN measured expectation minus noisy-target expectation.
- Gain over original ProGen, with saved clean-GFN policies as reference controls.

The target shift plus fitting gap equals the GFN's measured change from the clean
target. Close fit with reduced measured utility implicates reward corruption;
poor fit to the noisy target indicates optimization sensitivity. Neither trend
must be monotonic for three fixed noise draws. No statistically powered robustness
claim or general assay-noise tolerance follows from this small panel.

Aggregate target outcomes once per noise seed, never treating three training
replicates of the same target as independent reward realizations. Show individual
rows and ranges alongside means. Reused inspected labels remain development
outcomes; no fresh confirmation, unseen-family or broad-generation claim.

## Execution and saved records

Choose an available GPU with CUDA_VISIBLE_DEVICES, activate .venv, then run:

```bash
scripts/run.sh reward-noise --output runs/reward-noise-v1 --device cuda
```

This runs all fits sequentially on the selected GPU and evaluates automatically.
Main report: runs/reward-noise-v1/evaluation/report.md. The directory also contains
pinned input hashes, source/protocol snapshots, the clean task and head copies,
exact noise vectors and scores, realized noise moments, per-fit checkpoints and
progress, training costs, exact results and grouped summaries. Completed per-fit
checkpoints are saved immediately. Failure status is recorded; the launcher
refuses existing output directories and does not automatically resume or overwrite.

Read-only CPU preflight (no noise outcomes or GPU queried):

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh reward-noise --check-only
```

Reevaluate saved completed heads without training into a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-noise \
  --run runs/reward-noise-v1 --output runs/reward-noise-review
```

Next decisions depend on this result. Keep SubTB, attribute conditioning and
partial ProGen training out of this comparison. Per-query stochastic reward noise
would require a distinct protocol and objective interpretation.
