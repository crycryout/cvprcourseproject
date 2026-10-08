#!/usr/bin/env bash
# Resume this experiment epoch only. Never tune again after a freeze.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --id="${CVPR_GPU:-0}" --query-gpu=uuid --format=csv,noheader)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/inductor-cache"
if [[ ! -f artifacts/frozen_v2.json ]]; then
    if [[ ! -f artifacts/precision_v2.json ]]; then
        python scripts/run_when_idle.py verify-model
    fi
    if [[ ! -f artifacts/pilot_E1_v2.json || ! -f artifacts/service_table_v2.json || ! -f artifacts/stress_v2.json || ! -f artifacts/graph_pool_v2.json || ! -f artifacts/cpu_workers_v2.json ]]; then
        python scripts/run_when_idle.py pilot
    fi
    if [[ ! -f artifacts/compile_v2.json ]] || ! python -c 'import json; assert json.load(open("artifacts/compile_v2.json"))["status"] in {"passed", "failed"}'; then
        python scripts/run_when_idle.py compile-baseline
    fi
    python scripts/run_when_idle.py calibrate
    python scripts/run_when_idle.py validate-slots
    python -m cvpr_project freeze
fi
python scripts/preflight.py --output artifacts/preflight_delivery_v2.json
python scripts/make_matrix.py --output artifacts/serving_matrix_v2.json
python scripts/run_when_idle.py evaluate-quality
python scripts/run_when_idle.py run-matrix
for policy in E1 P0 C0; do
    if [[ "$policy" == C0 ]] && ! python -c 'import json; assert json.load(open("artifacts/frozen_v2.json"))["compile"]["status"] == "passed"'; then
        continue
    fi
    for suffix in _low high; do
        rate=0.3
        output="artifacts/profile_${policy}_low"
        if [[ "$suffix" == high ]]; then rate=1.1; output="artifacts/profile_${policy}"; fi
        if [[ ! -f "$output/summary.json" ]]; then
            python scripts/run_when_idle.py profile -- --policy "$policy" --rate-multiplier "$rate" --output "$output"
        fi
    done
done
python scripts/run_when_idle.py collect-nsight
python -m cvpr_project analyze
python scripts/verify_results.py
