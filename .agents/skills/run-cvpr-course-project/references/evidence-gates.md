# Evidence gates

Use this checklist when deciding whether to advance or report completion. File presence alone is insufficient.

| Gate | Minimum evidence | If missing |
|---|---|---|
| M0 | CUDA smoke; actual model identifier/weight hash; data/split hashes and class counts; environment lock | Repair environment/data, do not train or call the model pretrained without verified weights |
| M1 | Tiny-set overfit; measured pilot; seed17 train/validation curves; recoverable checkpoint with best-validation selection | Diagnose training or run the bounded pilot; no invented accuracy target |
| M2 | r=0 equivalence; p=0/reference merge agreement; correct mass/CLS/token lengths; actual validation runs | Repair algorithm before benchmarking |
| M3 | Protected source+destination exclusion; deterministic random identity; validation p selection; frozen protocol | Complete controls before official test |
| M4 | Completed manifests; all requested training seeds or declared downgrade; sample-level predictions; raw timing; backend/kernel evidence | Execute missing runs within budget or disclose reduced scope |
| M5 | Report numbers trace to runs; generated figures; runnable reproduction instructions; prior-work/limitations | Fix traceability, never fill gaps with estimates |

## Mandatory manifest semantics

Store at least `run_id`, `stage`, `status`, `git_sha`, `dirty_diff_sha256` if dirty, `config_sha256`, `dataset_split_sha256`, `checkpoint_sha256` when used, `training_seed`, `protection_seed` when used, `backend`, `dtype`, `batch_size`, `started_at`, `ended_at`, `gpu_hours`, `command`, and evidence paths.

Use `planned`, `running`, `completed`, `failed`, or `interrupted` states; do not store estimates in measured fields. Cumulative compute includes failed runs. Count each active GPU's elapsed job duration and distinguish allocated GPU-hours from utilization. Hash sanitized config snapshots, not secret-bearing environment dumps.

Store predictions keyed by stable sample ID; join only on identical sample IDs. Pair within the same checkpoint/seed and split. Keep performance repetitions separate from training seeds. Compute summary files from completed evidence only.

## Recovery examples

- Existing best checkpoint but missing manifest: inspect its metadata and logs; reconstruct provenance only when supported, otherwise mark unverified and rerun a bounded verification.
- No H800 visible: implement CPU correctness tests and data plumbing, but do not claim H800 performance or silently replace the planned hardware.
- AP slower than dense: report the slowdown; inspect score/match/scatter costs; complete the controlled study without hiding the optimized dense comparison.
- Test evaluated before freeze: preserve the event and document compromised selection independence; do not erase results or claim untouched test.
