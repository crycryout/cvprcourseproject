#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CVPR_GPU:-0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=4
python -m cvpr_project prepare-data
python -m cvpr_project prepare-model
python -m cvpr_project verify-model
python -m cvpr_project pilot
python -m cvpr_project compile-baseline
python scripts/run_when_idle.py calibrate
python scripts/validate_slots.py
python -m cvpr_project freeze
python scripts/make_matrix.py --output artifacts/serving_matrix_v2.json
python scripts/run_when_idle.py evaluate-quality
python scripts/run_when_idle.py run-matrix
python -m cvpr_project profile --policy E1 --output artifacts/profile_E1
python -m cvpr_project profile --policy P0 --output artifacts/profile_P0
python -m cvpr_project profile --policy C0 --output artifacts/profile_C0
python -m cvpr_project analyze
