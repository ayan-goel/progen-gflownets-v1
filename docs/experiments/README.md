# Experiment history and current pilot

Ready: [direct charge-context comparison](idr_context_2026-11-12.md), with fresh
matched fits of both heads and automatic evaluation.

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
