# Experiment history and current pilot

Current authorized experiment: [matched GFN versus weighted supervised training](idr_objective_2026-11-18.md).
Two fresh paired repeats, the same final block/head and shared scored examples,
1,600 updates per objective. Real results are pending; other charge profiles and
external-method comparisons remain deferred.

Latest completed: [1,600 versus 800 updates](idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026).
Joint success rises from 29.39% to 34.18% and 29.39% to 31.05% in fresh paired
evaluations. Both miss the five-point gain threshold; quality is retained. The
user chose to keep 1,600 as the working budget and defer further budget tests;
the historical gate outcome is unchanged. A supervised comparison with the same
adaptable layers is now authorized above.

Earlier completed: [last-block training-budget comparison](idr_lastblock_budget_2026-11-17.md#completed-results--18-november-2026).
Joint success increases from 19.92% to 29.59% and 23.63% to 29.69% when comparing
freshly evaluated preserved 400-update models with new 800-update models. Both
predefined screens pass. Retain the larger-budget recipe with its stretched
learning-rate schedule; the subsequent 1,600-update comparison above has completed.

Earlier completed: [two fresh-seed last-block confirmations](idr_lastblock_confirmation_2026-11-17.md#completed-results--17-november-2026).
Both repeats pass all predeclared screens. Joint success is 20.41% and 23.14%
adapted, 10.45% and 11.04% frozen TB, and 2.25% and 2.15% fresh plain ProGen.
All eight prompts favor adaptation over frozen TB in each repeat. Retain the
method and prioritize improving this same charge task. The
[post-hoc diagnosis](idr_lastblock_confirmation_2026-11-17.md#post-hoc-headroom-analysis--17-november-2026)
motivates the authorized bounded 800-update comparison above.
These completed fits are not new-prompt/property evidence.
The [original matched pilot](idr_lastblock_2026-11-16.md#completed-results--16-november-2026)
remains separate from the confirmation decision.

Earlier completed: [full-support transfer pilot](idr_transfer_2026-11-16.md#completed-results--16-november-2026).
Joint generation success is 8.50% unchanged, 6.45% ordinary TB, 6.93% corrected TB.
Both fit the bank almost exactly but leave a substantial holdout gap; the practical
screen fails. Next investigate coverage/generalization rather than longer bank fits.

Completed: [CPU variance diagnosis and repair](idr_variance_repair_2026-11-16.md).
Corrected fixed-bank TB passes the local gate in six sampled fits (TV 0.019–0.021),
including three fresh confirmation seeds. Matched ordinary fixed-bank training
still fails. The completed pilot above fails to transfer this into useful generation. The [averaged-sampling](idr_averaged_sampling_2026-11-16.md)
and [exact-expectation](idr_expected_conditioning_2026-11-16.md) controls retain
preceding evidence and cost limits.

Completed: [continuous-reward GFlowNet comparison](idr_continuous_2026-11-15.md#completed-results--15-november-2026).
Joint success is 9.86% unchanged, 7.52% previous TB and 9.57% continuous TB.
The practical screen fails. This motivated the exact-task learning investigation
above; reward-only follow-ups remain stopped.

Completed: [supervised-start comparison](idr_warmstart_2026-11-15.md#completed-results--15-november-2026).
Joint success is 9.38% unchanged, 16.11% after more supervision and 7.03% after
TB/replay. TB fails the progression screen despite improved selected-bank target
fit. End this continuation route; assess reward design before another GPU run.

Completed: [replay TB and supervised same-head pilot](idr_replay_2026-11-13.md#completed-results--14-november-2026).
Joint success is 3.91% on-policy TB, 4.79% replay and 8.69% supervised. Neither
passed its predeclared screen; this motivated the initialization comparison above.

Completed: [joint-acceptance TB reward comparison](idr_acceptance_2026-11-13.md#completed-results-and-next-decision--13-november-2026).
New TB achieves 3.13% joint success versus old TB's 3.65% and fails the practical
screen. Saved trajectories motivate the replay pilot above.

Completed: [matched TB/SubTB charge training](idr_subtb_2026-11-13.md#completed-results-and-next-decision--13-november-2026).
TB beats SubTB in all three seeds, but gains remain modest. The subsequent
[reward audit](idr_reward_audit_2026-11-13.md) motivated the revised reward;
the completed training comparison above tests whether it improves raw generation.

Completed: [branch-only learning-rate follow-up](idr_branch_rate_2026-11-12.md#completed-results--verified-13-november-2026).
Small observed charge/joint gain; this motivates the matched training comparison above.

Completed: [direct charge-context comparison](idr_context_2026-11-12.md#completed-results--12-november-2026).
Context sensitivity increased, but generation success did not improve; this
motivates the bounded follow-up above.

Completed: [real-state charge-learning diagnostic](idr_learning_diagnostic_2026-11-12.md#completed-results--12-november-2026).
Weak block-specific steering motivated the comparison above.

Completed: [IDR charge-guidance pilot](idr_charge_2026-11-12.md#completed-results--12-november-2026).
Guidance did not improve charge matching; selection and swaps performed better.
The [CPU learning diagnosis](idr_charge_diagnosis_2026-11-12.md) records checkpoint
checks, exact toy controls and the next evidence needed.

These completed studies are groundwork for the IDP/IDR pivot. The [IDR generation baseline](idr_baseline_2026-11-12.md) is complete, with prompted generation improving predicted-disorder yield; see [current status](../status.md).

| Stage | Evidence |
|---|---|
| Standalone ProGen2 and TF-Bind-8 | [Prior results](../prior_results.md) |
| Initial exact-task failures and diagnostics | [Exact diagnostics](exact_diagnostics_2026-10-14.md) |
| Longer constant-rate training | [Long-run result](exact_long_2026-10-14.md) |
| Late learning-rate reduction | [Scheduled result](exact_decay_2026-10-14.md) |
| Three-seed exact test passes | [Milestone and original biological protocol](exact_pass_and_biological_pilot_2026-10-15.md) |
| First biological pilot | [Completed results](biological_first_results_2026-10-15.md) |
| Historical partial-run observation | [Live loss check](biological_live_check_2026-10-15.md), superseded by completed results |

Preserve failures and final outcomes. New studies need explicit questions,
comparators, data boundaries and decision rules before real execution. The old
pilot's proposed follow-ups are historical recommendations, not implemented
current workflows. The [current scope](../status.md) controls next work.

Completed: [local predictor/reward study, arithmetic audit and results](predictor_reward_study_2026-11-03.md). It motivated the subsequently completed [ridge-reward GFN rerun](ridge_gfn_2026-11-09.md).

Completed: [ridge-reward GFN results](ridge_gfn_2026-11-09.md). All three seeds realize the modest target gain; this motivated the completed noise study.

Completed: [fixed reward-noise study](reward_noise_2026-11-09.md), all 18 fits closely match noisy targets; some corrupted targets lose biological utility.

Completed: [local three-split robustness study](split_robustness_2026-11-09.md). Both predictors improve target means; ridge is not consistently better. See current status for the next proposed check.


Completed: [mutation-position holdout study](position_holdout_2026-11-09.md). Both targets improve their own baselines; ridge gains are larger on all three grouped splits. No new GFN training.


Completed: [GFN on position-held-out ridge rewards](position_gfn_2026-11-09.md),
all nine fits closely match their targets and realize measured stability gains.


Completed: [multi-protein stability predictor benchmark and error diagnosis](multiprotein_2026-11-10.md).

Completed: [ridge regularization with fixed protein coverage](regularization_2026-11-10.md); lower prediction error but worse stable-candidate selection.

Completed: [more training proteins with a fixed ridge rule](coverage_2026-11-10.md); modest, non-monotonic selection gains and persistent absolute-prediction errors.
