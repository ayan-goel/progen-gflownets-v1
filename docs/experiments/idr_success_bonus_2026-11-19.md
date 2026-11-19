# Ninefold joint-success reward: matched GFN and supervised training

**Status: completed by the user on 19 November 2026; the substantial-GFN-advantage screen fails in all three repeats.**
All real fitting, generation and scoring were user-run. The prospective protocol
below remains unchanged; [results](#completed-results--19-november-2026) follow it.

## Question and rationale

Can stronger preference for satisfying both charge and predicted disorder make
GFN generation substantially better than equally strengthened weighted supervision?
The [previous comparison](idr_objective_2026-11-18.md) gives 31.35% GFN versus
40.92% supervised raw success; noncopy rates are 31.35% versus 33.94%. That result
does not establish a GFN advantage. The
[research assessment](../../literature/IDR_GFN_IMPROVEMENT_2026-11-19.md) owns the
target/fitting diagnosis, primary literature, alternatives and uncertainties.

Saved prior training samples estimate about 51% acceptance under the old target,
versus about 89% after a ninefold passing-weight bonus. These are uncertain,
post-hoc finite-pool estimates, particularly for RPB1, **not forecasts of trained
success**. The bonus preserves the relative weights of failures and expands the
observed log-factor range less than doubling every violation penalty. It also
introduces a jump at the acceptance boundary and could be harder to learn.

This is one fixed intervention, with no bonus sweep, new profile, additional
unfreezing, dynamic replay, annealing, loss replacement or training extension.

## Reward and unchanged acceptance

New version: `success_bonus_v1`.

\[
\log R(x\mid c)=\log p_0(x\mid c)-2.5V(x)+\log(9)\,\mathbf1_A(x).
\]

- `x`: the generated 64-amino-acid tail; `c`: the known 16-amino-acid prefix.
- `p0`: original frozen ProGen's tail probability with the inherited canonical-20
  token support, temperature one and float32 convention.
- `V`: the unchanged sum of normalized violations outside the block-charge,
  charged-fraction and predicted-disorder tolerances.
- `A`: the event that **all** requirements pass; `1_A` is one for a joint pass,
  zero for a failure.
- `R`: the unnormalized GFN terminal weight. The new property log factor is
  `-2.5 V + log(9) 1_A`; all reward values are detached labels.

Keep first-/second-half charge targets +0.125/−0.125 with tolerance 0.0625,
charged fraction 0.25 with tolerance 0.125, and at least 80% of tail residues
having metapredict V3 scores strictly above 0.5. The 0.2 disorder-deficit scale
and all threshold conventions remain unchanged. Passing sequences have property
factor nine rather than one; failing factors and their relative ordering stay
unchanged. Ninefold weight is not ninefold normalized generation probability.

GFN uses ordinary TB against the original prior and the new factor. Supervision
uses the same factor in detached within-prompt softmax weights for mean-token
negative log likelihood. The supervised target is still the weighted changing
proposal, not the same fixed GFN distribution. The
[previous objective definitions](idr_objective_2026-11-18.md#objectives-and-weighting-rationale)
are otherwise unchanged.

Relabel cloned original-bank rows before GFN normalizer initialization and replay.
Do not mutate source bank/cache files or warm-up weights. Recompute fresh rewards
from saved isolated-tail scores and retain base factor, added bonus and final
factor. The normalizer uses training-only, equal pass/fail least-squares offsets;
its initialization shifts by `log(9)/2` under the same starting head. It is not
an exact partition estimate. Both final checkpoint types record the reward version.

## Fixed training plan

Executable settings: [idr-success-bonus.json](../../configs/idr-success-bonus.json).
Retain the [previous preparation and settings](idr_objective_2026-11-18.md#fixed-settings-and-data-boundaries):
each policy owns a separate copy of ProGen's final block and the original
supervised step-800 guidance head. Earlier blocks, embeddings, final normalization,
LM projection and the original reference are frozen. No adapted 1,600-step model
is continued. GFN alone has eight learned normalizers.

| Choice | Fixed value and reason |
|---|---|
| Pass multiplier | 9; one explicit acceptance-odds change, with the original penalty 2.5 retained |
| Repeats | Three fresh paired training streams; bounded consistency check, not a power guarantee |
| Training | 1,600 updates per arm, final checkpoint only; user's retained budget |
| Shared batch | 8 GFN + 8 supervised + 8 prior + 8 original-bank replay examples; same ordered IDs for both learners |
| Refresh | Every four updates; 400 rounds per repeat |
| Replay | Original train partition, alternating success/failure by prompt and round; no dynamic bank |
| Rates | Head 1e-4; direct task matrix 1e-3; final block 1e-5; GFN normalizer 1e-2 |
| Schedule | Rates multiplied by 0.1 after update 1,200; inherited 75% convention |
| Optimization | Adam, betas (0.9,0.999), epsilon 1e-8, no weight decay; separate norm-one clipping for head/normalizer and block |
| Microbatch | Four; preserve full-batch weighting and accumulated gradients |
| Evaluation | 128 raw samples per prompt per arm; 1,024 per arm/repeat; GFN, supervised and plain prior |
| Storage | Atomic overwrite of one latest changed-weights checkpoint per arm, final rename; no intermediate checkpoint accumulation |

| Repeat | Training streams GFN / supervised / prior | Replay | Evaluation |
|---|---|---:|---:|
| replicate-1 | 1023000 / 1023001 / 1023002 | 1025000 | 1033000 |
| replicate-2 | 1123000 / 1123001 / 1123002 | 1125000 | 1133000 |
| replicate-3 | 1223000 / 1223001 / 1223002 | 1225000 | 1233000 |

Both arms restart the same evaluation stream within each repeat. Every repeat
collects 9,600 fresh training trajectories and 3,200 replay draws, with 51,200
trajectory presentations per trained arm. Totals are 28,800 fresh training paths
and 9,216 final outputs. Shared collection is coupled; report actual scorer,
generation and update costs without claiming independently costed pipelines.
No new tuning is allocated to either method in this comparison.

## Evaluation and prospective decision

Acceptance metrics remain unchanged. Report each repeat/prompt, raw joint success,
charge/disorder separately, distinct successful tails, noncopy successes,
successful clusters, whole-output duplicates, internal motif repetition and costs.
Use all raw outputs as denominators; exclude no difficult prompt. Training copies
use one conservative union of original training-bank tails and fresh training
tails from **all three** new repeats. Own-training copies use actual initial
warm-up plus same-repeat exposure. Earlier adapted fits are not exposures of
these new policies. Noncopy does not establish external biological novelty.

Retain the existing within-prompt 80% positionwise-identity connected-component
cluster definition. Components can join distant pairs through intermediate tails.
V3 is optimized predicted disorder; V2 is sensitivity, not independent biology.

A provisional **substantial GFN advantage** requires every paired repeat to show:

- at least **0.10 absolute raw joint-success improvement** over supervision;
- strictly more distinct successful noncopy tails at the same output count;
- no more than 0.05 absolute loss of predicted-disorder pass or overall uniqueness.

Noncopy rates and successful clusters remain reported, not alternative winning
criteria selected after seeing the result. These practical thresholds are our
prospective engineering screen, not a significance test. They do not rewrite
the completed experiment's five-point rule. Training repeats, not individual
sequence draws, are the independent fitting units.

Both methods receive the new reward. Beating the old 40.92% alone is insufficient.
Comparisons to old results are development context: new seeds and jointly generated
data prevent an exact paired attribution of changes to the bonus. This experiment
is not an equal-target divergence comparison or an optimally tuned methods benchmark.
Any later superiority claim requires equal declared development allowance and fresh
confirmation after locking both recipes. No extra confirmation is launched here.

Stop after the three final evaluations. If both methods improve similarly, record
better property pressure without a GFN advantage. If GFN remains weak, use its
observed fitting/coverage diagnostics to select at most one separately specified
learning intervention; no automatic replay change or sweep follows this run.

## Execution and recovery

The [runbook](../runbook.md#matched-training-with-a-success-bonus) owns the command.
The user activates `.venv`, selects `CUDA_VISIBLE_DEVICES`, and launches the script.
Training includes final generation, evaluation and a Markdown report. `--check-only`
reads manifests, settings and small CPU tensors without executing ProGen or a
disorder predictor. Recovery scores sealed final samples without loading ProGen.
Checkpoint files support generation/analysis, not optimizer/RNG training resume.

New versioned modules reuse unchanged sampling, optimization and summary helpers.
Historical IDR source modules remain byte-identical for pinned completed manifests.
The authoritative reward identity is `reward_definition.json` and the matching
checkpoint metadata. `pilot_config.json` retains its inherited historical
`reward_version` field; it does not select this versioned runner's reward.

The [verification record](../verification.md#success-bonus-comparison--19-november-2026)
links 32 passing CPU tests and a successful input-only preflight. Tiny fixtures
exercise all three paired fits, final evaluation, corruption rejection and
source-independent recovery. These establish execution correctness, not expected
real performance; no real ProGen/scorer or GPU execution occurred during setup.

## Completed results — 19 November 2026

Run: `runs/idr-success-bonus-v1`. All three paired fits reached 1,600 updates,
followed by final generation and scoring, in **5 h 57 min 2 s**. Each method
generated 3,072 raw outputs across eight prompts and three repeats: 9,216 outputs
overall. No checkpoint selection, threshold change or prompt exclusion was made.

| Repeat | GFN joint success | Supervised joint success | GFN noncopy success | Supervised noncopy success |
|---|---:|---:|---:|---:|
| 1 | 37.60% | 42.29% | 37.60% | 37.30% |
| 2 | 38.09% | 48.44% | 38.09% | 42.48% |
| 3 | 40.92% | 42.58% | 40.92% | 36.72% |
| Pooled | **38.87%** | **44.43%** | **38.87%** | **38.83%** |

All percentages retain **all raw outputs** in the denominator. Noncopy success
means a passing tail absent from the common original-bank/all-three-repeats
training-tail union; it is not success conditional on first removing copies.

| Pooled measurement, 3,072 outputs per method | GFN | Supervised |
|---|---:|---:|
| Joint passing draws | 1,194 | 1,365 |
| Passing draws excluding exact training copies | 1,194 | 1,193 |
| Distinct successful noncopy tails | 1,194 | 1,193 |
| Overall unique-tail fraction | 98.96% | 94.60% |
| Predicted-disorder pass | 93.16% | 89.16% |
| Charge pass | 42.19% | 49.15% |

Plain ProGen passes jointly on 62/3,072 outputs (**2.02%**). GFN and supervision
both steer generation substantially, but improvement over the prior is not the
methodological goal of this comparison.

**The predeclared GFN advantage screen fails in every repeat.** GFN trails
supervision in raw success by 4.69, 10.35 and 1.66 percentage points, rather than
gaining the required ten. Distinct successful noncopy count differences are
+3, −45 and +43; quality checks pass. The pooled one-candidate advantage is not
a substantial or repeat-consistent win, and the near tie does not establish
statistical equivalence. Do not replace the declared raw-success criterion with
the favorable diversity metrics after seeing outcomes.

### What changed, and what remains unresolved

The previous no-bonus comparison had 31.35% GFN versus 40.92% supervised raw
success, and 31.35% versus 33.94% noncopy success. The present GFN rate is 7.52
points higher and supervision is 3.52 points higher. This is encouraging
development evidence consistent with stronger acceptance pressure, **not a paired
causal estimate of the bonus**: repeat count, random streams and the resulting
shared training trajectories differ, and the common-copy union is larger.

The supervised raw advantage is almost entirely accounted for by 172 passing
training-copy draws; GFN has no passing copies. All supervised passing copies
come from RPB1. Nevertheless, retaining all eight prompts gives essentially the
same noncopy successful-candidate yield for the methods. GFN preserves more
overall uniqueness and predicted disorder; those are useful tradeoffs, not the
requested large success/yield improvement. Exact noncopy does not prove external
novelty, and optimized V3 predictions do not establish biological disorder.

Charge remains the main observed output failure: 57.81% of GFN outputs fail its
charge requirements, compared with 6.84% failing the V3-disorder screen. These
sets overlap. The previously estimated 89% target acceptance was a noisy
finite-pool calculation, not a promised outcome or verified attainable accuracy.

The fixed reward-only intervention is complete. Retain both methods and their
results. Do not automatically increase the multiplier or training length, switch
the property, or claim GFN superiority. Use the saved-data learning/coverage
diagnostics to choose at most one separately justified follow-up before another
GPU experiment. No new training recipe is selected by these results alone.
