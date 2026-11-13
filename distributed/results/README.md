# Evidence index

Read the English [report](report.pdf) ([Markdown](report.md)) and
[12-slide presentation](slides.pdf). The report answers all three research questions;
negative results and implementation limitations remain visible.

1. **Online performance:** [summary.csv](summary.csv), [controls.csv](controls.csv)
   and [stages.csv](stages.csv). The `runs/` and `controls/` directories contain
   sanitized metrics for 120 two-host cases and 48 single-host controls. Seeds
   are paired across policies and topologies; ranges in figures are not confidence
   intervals. All offered cohort requests count toward SLO success.
2. **Detection correctness:** [quality.csv](quality.csv), [quality.json](quality.json),
   [stress.json](stress.json) and [stress_rebuild.json](stress_rebuild.json).
   Ten evaluations each cover 4000 images. The original measurement IDs are retained;
   [quality_compatibility.json](quality_compatibility.json),
   [quality_origin_frozen.json](quality_origin_frozen.json) and
   [quality_origin_benchmark.py](quality_origin_benchmark.py) document unchanged
   inference semantics after arrival-harness repair.
3. **Frozen methodology:** [frozen.json](frozen.json),
   [service_curves.json](service_curves.json), [capacity.json](capacity.json),
   [environment.json](environment.json) and the repository's
   [protocol](../docs/PROTOCOL.md). These distinguish calibration from evaluation.
4. **Audit and exclusions:** [audit.json](audit.json) records independent raw-log
   recomputation before export; [attempts.json](attempts.json) preserves excluded
   attempts and their reasons. [resources.json](resources.json) counts both worker
   lifetimes, including failed/idle time; [storage.json](storage.json) audits the
   aggregate storage budget. [delivery.json](delivery.json) records final checks.
   [checksums.json](checksums.json) identifies every released evidence file.
5. **CPU-only verification:** from the repository root run
   `PYTHONPATH=src python scripts/verify_results.py` with Python 3.10 and NumPy 1.26.4.
   No GPU, dataset, checkpoint or private endpoint is required. This verifies public
   consistency and provenance; hashes alone cannot establish unpublished raw data.

Raw JPEGs, model weights and full request traces remain outside Git. Real inference
reproduction regenerates them with [scripts/reproduce.sh](../scripts/reproduce.sh).
