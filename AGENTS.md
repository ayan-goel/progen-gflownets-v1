# Project instructions

Do not launch any GPU job, including inference, model checks, sequence generation, or training. Provide the user with commands to execute themselves. CPU-only setup, file inspection, analysis, and documentation are allowed. Tiny CPU-only training runs are allowed solely to test correctness and end-to-end execution. Leave full training runs and all GPU execution to the user.

Project scope: the frozen-ProGen2 stability-guidance project defined in spec.md. Read docs/status.md for the current experiment and next decision; do not treat historical first-GPU milestones as the active task. All implementation belongs in this v1 folder; the test folder is a read-only reference. Document decisions and hyperparameters in docs/. Work in atomic slices and verify each slice before proceeding. Preserve raw outputs and clearly distinguish smoke tests, real experiments, failures, and experimentally established biological function. SMC and speculative decoding remain out of scope.

## Research documentation (required)

Documentation is part of completing the work, not a separate optional follow-up.
Record every experiment, analysis, implementation/configuration change, research
investigation and decision as the project progresses. Record unsuccessful attempts,
negative findings, interruptions and corrections too. Group routine inspections
and small related edits into concise dated entries; do not paste terminal transcripts
or duplicate the same explanation across several files.

- **Before a real experiment:** create or update a dated protocol under
  `docs/experiments/` with the question, hypothesis, fixed comparisons, exact command,
  configuration, data/split identities, hyperparameters, seeds, budgets, metrics,
  checkpoint-selection rules, outcome-access rules and failure handling. State what
  is planned, what the user will execute, and what remains unimplemented. Link the
  configuration and existing protocol for unchanged settings, listing any differences.
- **For every run/check/analysis:** record who executed it (user or agent), date,
  exact command or reproducible analysis procedure, input/config/code provenance,
  output/log paths, CPU/GPU or smoke status, observed completion/failure, relevant
  cost and verification. Batch routine read-only checks in the implementation log.
  Never describe a prepared command as executed, a live log as completion, or a CPU
  fixture as a real result. Unknown information stays explicitly unknown.
- **For every decision:** record what changed, why, the supporting evidence, relevant
  alternatives/tradeoffs, and the next test or consequence. Include all numerical
  choices and defaults that affect scientific behavior: filtering, splits, scaling,
  architecture, optimization, seeds, budgets, stopping/selection and evaluation.
  Record unchanged settings by reference rather than silently dropping their provenance.
- **For motivating sources:** cite the primary paper, official documentation, dataset
  release or upstream code near the claim it supports. For empirical project choices,
  link the local result and artifacts. Mark engineering defaults, hypotheses and our
  own derivations as such; do not invent a literature justification for a convenience
  choice or imply that a cited paper establishes our result. Record source version,
  access date and reading depth in `literature/` when adding or revisiting research.
- **After outcomes arrive:** preserve raw outputs; add a dated result/analysis record
  comparing against the declared baselines, including failures, duplicate handling,
  uncertainty, limitations and the next decision. Distinguish exact expectations,
  sampled batches, predicted scores, measured labels and newly performed experiments.
  Document any post-hoc analysis or departure from the original protocol explicitly.
- **Keep the right files authoritative:** `spec.md` owns the scientific contract;
  `docs/status.md` owns the current stage and next decision; `docs/runbook.md` owns
  runnable instructions; dated experiment records own protocols/results;
  `docs/decisions.md` owns cross-experiment reasoning; `docs/implementation_log.md`
  owns implementation/check history; `literature/` owns source review. Use links
  for detail rather than copying drifting summaries between these files.
- **Keep navigation current:** update the experiment index and relevant documentation
  indexes when adding records. Update current status after a verified transition;
  distinguish user-reported progress from inspected artifacts. Change scientific
  definitions in the spec only when the contract changes, and record why; keep its
  navigation/status pointers accurate without changing the definitions. Preserve dated history;
  clarify superseded claims with a dated note/link rather than rewriting the past.
- **Protect active experiments:** do not edit active run outputs, source snapshots,
  data/split manifests or live configurations to tidy documentation. Keep existing
  artifact paths stable. Do not reveal reserved outcomes early to populate a report.
  Documentation work never authorizes GPU execution, a restart or a new real fit.
- **Verify documentation changes:** check affected local links and consistency of
  commands, settings, statuses and claims. Use proportionate checks; documentation-only
  work does not require rerunning training or model tests.

Use `docs/experiments/TEMPLATE.md` for new records, adapting it to the task and
omitting irrelevant sections explicitly. Keep reports clear, concise and usable
by a researcher who did not read the chat.

## Code readability

Write code for an undergraduate researcher to read, understand, and modify.

- Use descriptive variable and function names. Avoid single-letter names except for conventional, very local mathematical notation.
- Use one statement per line; do not compress control flow or multiple assignments with semicolons.
- Organize imports normally and avoid dynamic imports for ordinary dependencies.
- Use functions when they separate a substantial task or avoid meaningful duplication. Keep short, single-use steps inline when that is easier to follow; do not force a tiny `main()` or add wrappers just for structure.
- Add module and function docstrings describing purpose. Comment scientific choices, units, indexing, stopping rules, reproducibility assumptions, compatibility workarounds, and non-obvious logic. Explain why rather than narrating obvious syntax.
- Format long calls, dictionaries, and report text over readable lines. Use standard Python spacing and consistent formatting.
- During readability refactors, preserve sampling settings, seed behavior, output formats, and scientific definitions. Validate with CPU-only comparisons where possible; never launch GPU work to verify a change.
- Preserve the source needed to interpret existing run hashes when editing scripts during an active experiment. Do not modify active run outputs or restart the user's process.

## Simplicity and scope

Implement exactly what the current task needs. Do not overengineer or expand the
scope with unsolicited features, future-proofing, extra dependencies, generic
frameworks, configuration layers, or bells and whistles. Before adding anything,
ask whether it is necessary for the requested result; if not, leave it out.

Prefer straightforward code and a small number of purposeful functions. Keep
comments useful and concise. Readability does not mean adding an abstraction or
a long explanation for every step. When cleaning up code, remove unnecessary
machinery rather than merely reorganizing it. Preserve the checks and experiment
records that are needed for correctness and reproducibility. Verify changes in
proportion to their risk; do not build extra testing infrastructure for a small
cleanup.
