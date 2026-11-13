#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ ! -f configs/hosts.local.json ]]; then
  echo 'Create configs/hosts.local.json from configs/hosts.example.json and start the two workers.' >&2
  exit 2
fi
if [[ -e artifacts/reproduction_v3/frozen.json ]]; then
  echo 'Existing immutable freeze found. Use the explicit resume commands in README.md.' >&2
  exit 2
fi
trap '.venv/bin/python scripts/manage_workers.py stop' EXIT
.venv/bin/python -m pytest -q
.venv/bin/python -m batchserve.experiments info --output artifacts/reproduction_v3
.venv/bin/python -m batchserve.experiments stress --output artifacts/reproduction_v3
.venv/bin/python -m batchserve.experiments calibrate --output artifacts/reproduction_v3
.venv/bin/python -m batchserve.experiments quality --output artifacts/reproduction_v3
.venv/bin/python -m batchserve.experiments matrix --output artifacts/reproduction_v3
.venv/bin/python -m batchserve.experiments controls --output artifacts/reproduction_v3
.venv/bin/python scripts/manage_workers.py stop
trap - EXIT
.venv/bin/python scripts/analyze.py --input artifacts/reproduction_v3 --output results-reproduction
.venv/bin/python scripts/make_slides.py --results results-reproduction
.venv/bin/python scripts/verify_results.py --results results-reproduction
