# v2 evidence gates

| Gate | Required evidence | Failure response |
|---|---|---|
| M0 | Model/processor/hash, calibration/eval IDs, correct masks/categories/boxes, reference AP | Fix vision pipeline before optimization |
| M1 | Real E0/E1 events, 1000-request pilot, stage profile, actual budget estimate | Identify bottleneck; do not assume one |
| M2 | Safe slot/event lifecycle, matching outputs/AP, partial-batch stress, capture coverage, overlap trace | Diagnose races or graph breaks; retain valid partial capture within bounded effort |
| M3 | Open-loop trace determinism, loadgen lag, denominator accounting, tuned controls, frozen service/SLO/load table | Repair harness before held-out comparisons |
| M4 | Request logs, completed/failed counts, offline AP, paired trace results, A0 control, compile status | Run missing cases within cap or disclose reduced scope |
| M5 | All tables trace to runs, executable reproduction, visual examples, limitations/prior work | Never fill missing metrics with guesses |

## Manifest contract

Include protocol_version=2, unique run_id, policy, trace_seed/type/hash, frozen_config_sha256, model/processor/data-split hashes, source Git SHA and dirty diff hash, precision/backend/bucket set, CPU workers, GPU model, timing scopes, timestamps, actual cost and evidence paths. Do not include credentials or private hostnames in public files.

Request records preserve scheduled and actual arrival, stage host timestamps, CUDA durations in a separate clock domain, request/image IDs, deadline, batch/slot/valid_count, completion and status. Distinguish completed, rejected, failed, missed deadline and unfinished; late completion is a completion and an SLO failure, not two distinct requests.

Runs may be planned/running/completed/completed_with_failures/interrupted/failed. Synthetic tests cannot enter measured summary. Count all offered requests for the fixed measurement cohort even if they finish in drain; report drain and unfinished censoring explicitly.

## Recovery checks

- Existing graph object is not a portable checkpoint: reconstruct/capture in a new process and revalidate addresses/weights/config.
- Existing output folder without manifest is unverified. Inspect real logs, never mark success based on filenames.
- Faster GPU-forward but slower E2E means no E2E win; profile CPU/queue/copy costs and report both.
- Faster than E0 but slower than F0/C0 means the proposal did not beat the optimized baseline.
- Near-zero transfer share means PCIe optimization has limited headroom; do not amplify inputs or offload small weights to manufacture a bottleneck.
