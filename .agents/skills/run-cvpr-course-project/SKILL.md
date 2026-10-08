---
name: run-cvpr-course-project
description: Complete and resume the CISC8005 cvprcourseproject on NVIDIA H800, including DeiT image classification, ToMe/AP-ToMe implementation, controlled experiments, profiling, and evidence-backed course reporting. Use for this repository's implementation, training, evaluation, debugging, analysis, report writing, or interrupted-run recovery.
---

# Run the CVPR course project

Execute the requested research work in `crycryout/cvprcourseproject`; do not merely restate its plan. Treat this as a repository-scoped skill intended for the user's H800 Codex client, not as proof of a globally installed skill.

## Establish state

1. Locate the repository containing `AGENTS.md`, `PROJECT_PLAN.md`, and `configs/project.json`. If invoked elsewhere, request a repository path; do not launch jobs in an unrelated directory.
2. Read `AGENTS.md`, `docs/STATUS.md`, and the relevant sections of `PROJECT_PLAN.md`. Inspect existing processes, manifests, checkpoints, and local changes before resuming work.
3. Read `docs/EXPERIMENT_PROTOCOL.md` before changing algorithms or running experiments; read `docs/IMPLEMENTATION_PLAN.md` for the active milestone. Use `references/evidence-gates.md` for completion checks.
4. Run `python3 scripts/preflight.py --output artifacts/preflight.json` and `python3 scripts/make_matrix.py --output artifacts/validation_matrix.json` when environment/config evidence is absent or stale. Do not treat these planning tools as model experiments.
5. Identify the first unmet gate and perform it. Communicate the current milestone, evidence, remaining uncertainty, and next concrete action in Chinese.

## Execute the milestone sequence

- **M0:** Verify dataset and non-distilled DeiT-Small weights, isolate the environment, create the class-stratified 45k/5k split, record hashes and exact dependency versions. Read `docs/RELATED_WORK.md` and resolve the nearest prior-work overlap before making novelty claims.
- **M1:** Implement the training/evaluation foundation; pass a tiny-set overfit check and a <=10-minute pilot. Estimate measured cost before training seed17. Select checkpoints only on validation.
- **M2:** Implement the pinned ToMe reference behavior and shared attention adapter. Pass r=0 dense equivalence, token mass/shape/CLS tests, and small-tensor reference comparisons before scaling.
- **M3:** Implement partition-wise AP protection and matched random protection. Protect tokens from both source and destination roles; keep actual per-layer token counts identical to ToMe. Select p only with seed17 validation and save a frozen protocol.
- **M4:** Complete the other training seeds under the fixed recipe, evaluate the sealed test set after freezing, benchmark both explicit and SDPA tracks, and analyze sample-aligned predictions and raw timing records.
- **M5:** Generate tables/figures from completed runs and write the English report using `docs/REPORT_OUTLINE.md`. Write reproducibility commands and evidence links, then return results for the user's review.

Read `docs/H800_HANDOFF.md` for startup/recovery wording. Implement future `cvpr_project` CLI commands before trying to run them; only the two planning tools exist in the initial repository.

## Preserve experimental meaning

- Reuse one dense checkpoint per training seed across every compression method. Do not fine-tune AP alone or compare differently trained backbones.
- Keep the official test labels out of training, recipe selection, checkpoint selection, and p/r tuning. Require a matching freeze artifact for final test. If leaked, disclose the leak and downgrade claims rather than concealing it.
- Use the exact parity partition and mass-weighted aggregation in the plan. Do not silently replace per-partition protection with global top-p, or decrease r to satisfy a mask.
- Verify p=0 equivalence, actual sequence lengths, per-forward mass reset, deterministic random-protection identity, and protected-token invariance at the merge boundary.
- Include attention-score extraction, sorting, matching, gathering, scattering, and required synchronization in latency. Do not compare only against an intentionally slow dense backend.
- Separate algorithmic FLOPs, GPU event latency, host-wall latency, and full pipeline scope. Do not label batched milliseconds as per-image latency.
- Keep precision, preprocessing, sample IDs, and backend metadata explicit. Profile actual kernels; an SDPA call does not guarantee a FlashAttention kernel.
- Preserve unsuccessful configurations and negative results. Do not guarantee gains, infer measurements from paper tables, or fill absent metrics with plausible values.

## Use resources and recover safely

Inspect existing PyTorch and available GPU memory before installation. Do not change system drivers/global packages, GPU power settings, or other users' processes. Use one GPU by default; use the second only for independent work on an idle device. Reserve uncontended intervals for formal timing.

Track GPU-hours and storage in manifests across sessions. Plan for 24–60 GPU-hours; stop new costly runs at the 80 GPU-hour or 50 GB project cap and report what is complete. Implement checkpoint recovery and avoid duplicate jobs. If over budget, remove optional work and use the documented minimum scope, labeling omitted seeds or experiments.

If downloading data/weights is blocked, give exact required local inputs and resume once supplied. Never silently substitute random weights. Do not ask again for ordinary reversible implementation choices already authorized by the project request.

## Finish honestly

Update `docs/STATUS.md` after each gate with real evidence, active runs, cost, and next action. Never mark a run complete from an existing directory alone. Preserve run manifests and failed runs.

Before publishing source/results, inspect staged files for large weights, raw course PDFs, secrets, private paths, and license notices. Do not upload those files. Do not submit coursework or contact the instructor. The user reviews the final report.
