# Repository instructions for Codex

## Objective and starting point

Complete the CISC8005 computer vision project defined in `PROJECT_PLAN.md`. Read `.agents/skills/run-cvpr-course-project/SKILL.md`, `docs/STATUS.md`, and the relevant protocol before implementation. Treat the checked-in material as a plan, never as evidence that experiments ran.

The user intends to execute this on an H800 server. Do the implementation, run justified experiments within the project budget, analyze real outputs, and produce a paper-style report. Do not stop after drafting another plan. Resume from verified artifacts after interruption. Explain progress in concise Chinese; use English for the final paper unless the user requests otherwise.

## Scientific rules

1. Keep the official CIFAR-100 test set sealed until configurations are frozen. Use only the training-derived validation set for checkpoint, recipe, protection ratio, and budget selection.
2. Compare methods using the same dense checkpoint, sample IDs, precision, batch size, and per-layer token counts. Separate matched-backend comparisons from practical comparisons against an optimized dense model.
3. Do not invent accuracy, speedup, completed milestones, plots, citations, or run IDs. Synthetic unit-test fixtures must be clearly marked and excluded from results.
4. Account for scoring, sorting, gathering, scatter reduction, and all synchronization in measured runtime. FLOP estimates are not latency evidence. Do not silently disable a faster dense attention backend.
5. Preserve failed runs and negative findings. Do not keep changing the method after viewing test results. Record deviations before rerunning.

## Execution constraints

- Use an isolated project environment. Do not replace the server driver, CUDA installation, or global Python packages. Inspect existing PyTorch before choosing versions; lock the tested environment.
- Do not kill other jobs or change GPU clocks/power settings. Select an available GPU and report contention.
- Plan for a total of 24–60 GPU-hours; use the measured pilot to revise estimates. Cap automated work at 80 GPU-hours or 50 GB of project storage. Track usage across sessions. Pause new costly jobs at the cap and report completed deliverables.
- Use one GPU by default. Independent training may use a second idle GPU; do not run concurrent jobs during formal timing. No DDP dependency.
- Run a <=10 minute training pilot before full training. Save recoverable checkpoints, config snapshots, run manifests, logs, and measured cost.
- Commit source, small metrics, split metadata, and documentation only. Keep datasets, weights, full traces, private logs, hostnames, tokens, and environment secrets out of Git. Inspect staged changes before publishing.
- Do not upload the instructor's PDF. `docs/COURSE_REQUIREMENTS.md` is the paraphrased source record.
- Use pinned upstream revisions and retain notices if adapting code. ToMe is CC-BY-NC-4.0; do not relabel its code as MIT or remove attribution.

## Completion and reporting

Follow the milestone gates in `docs/IMPLEMENTATION_PLAN.md`. Keep `docs/STATUS.md` current with the next executable step and real evidence paths. Implement missing code before running the planned CLI examples. Produce a reproducibility command list, main table, Pareto curves, ablations, qualitative examples, limitations, and report draft from actual results. The user reviews and submits the course report; do not contact the instructor or submit coursework automatically.
