#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CVPR_GPU:-0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=4
python scripts/preflight.py --output artifacts/preflight_delivery_v2.json
python -m cvpr_project prepare-data
python -m cvpr_project prepare-model
python -m cvpr_project verify-model
python -m cvpr_project pilot
python -m cvpr_project compile-baseline
python scripts/run_when_idle.py calibrate
python scripts/run_when_idle.py validate-slots
python -m cvpr_project freeze
python scripts/make_matrix.py --output artifacts/serving_matrix_v2.json
python scripts/run_when_idle.py evaluate-quality
python scripts/run_when_idle.py run-matrix
for policy in E1 P0 C0; do
    if [[ "$policy" == C0 ]] && ! python -c 'import json; assert json.load(open("artifacts/compile_v2.json"))["status"] == "passed"'; then
        continue
    fi
    python scripts/run_when_idle.py profile -- --policy "$policy" --rate-multiplier 0.3 --output "artifacts/profile_${policy}_low"
    python scripts/run_when_idle.py profile -- --policy "$policy" --rate-multiplier 1.1 --output "artifacts/profile_${policy}"
done
python -m cvpr_project analyze
