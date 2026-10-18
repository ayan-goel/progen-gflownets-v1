# Verification evidence index

[Verification history](../verification.md) explains the checks and their limits.
This directory stores supporting logs, hashes, readiness checks and numerical
audits; it does not replace raw experiment outputs. For current execution state,
read [status](../status.md).

| Stage | Supporting evidence |
|---|---|
| Diversity development implementation | [41-test CPU log](diversity-tests.txt), [readiness](diversity-readiness.json), [handoff audit](diversity-implementation-audit.json) |
| Documentation cleanup, 17 Oct | [Protected-file/link audit](documentation_cleanup_2026-10-17.json) |
| Confirmation completed results | [Seal/source checks, identical CPU replay and analysis](confirmation-result-analysis.json) |
| Confirmation implementation | [37-test CPU log](confirmation-tests.txt), [preflight/readiness](confirmation-readiness.json) |
| Development diagnostic | [CPU tests](target-diagnostic-tests.txt), [readiness](target-diagnostic-readiness.json), [verified result analysis](target-diagnostic-result-analysis.json) |
| Biological pilot | [configuration readiness](biological-pilot-config.json), [final result analysis](biological-first-analysis.json), [label-free prior audit](post-pilot-prior-audit.json) |
| Exact gate pass | [final analysis](exact-pass-analysis.json), [long scheduled configuration](exact-decay-long-config.json) |
| Learning-rate schedule | [CPU tests](schedule-tests.txt), [compatibility](schedule-compatibility.json), [scheduled-run analysis](exact-decay-analysis.json) |
| Longer constant-rate run | [configuration](exact-long-config.json), [result analysis](exact-long-analysis.json) |
| Earlier exact diagnostics | [CPU tests](diagnostic-tests.txt) |
| Initial implementation | [CPU tests](cpu-tests.txt), [audit](audit.json), [environment doctor](doctor.json) |

A readiness record is a pre-run check, not proof that a later run completed.
Synthetic test logs are not biological evidence. Preserve old logs and dated
records; add new files for a new experiment or substantive verification stage.
