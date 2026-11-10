# Retained experiment history

The active project returns to the first biological pilot. See [current status](../status.md).

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

Completed: [local predictor/reward study, arithmetic audit and results](predictor_reward_study_2026-11-03.md). Next is the [ridge-reward GFN rerun](ridge_gfn_2026-11-09.md), prioritized after Tom's follow-up meeting.

Completed: [ridge-reward GFN results](ridge_gfn_2026-11-09.md). All three seeds realize the modest target gain; next is a controlled reward-noise protocol.

Completed: [fixed reward-noise study](reward_noise_2026-11-09.md), all 18 fits closely match noisy targets; some corrupted targets lose biological utility.

Completed: [local three-split robustness study](split_robustness_2026-11-09.md). Both predictors improve target means; ridge is not consistently better. See current status for the next proposed check.


Completed: [mutation-position holdout study](position_holdout_2026-11-09.md). Both targets improve their own baselines; ridge gains are larger on all three grouped splits. No new GFN training.


Completed: [GFN on position-held-out ridge rewards](position_gfn_2026-11-09.md),
all nine fits closely match their targets and realize measured stability gains.


Ready for user execution: [multi-protein stability predictor benchmark](multiprotein_2026-11-10.md).
