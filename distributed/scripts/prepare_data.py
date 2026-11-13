#!/usr/bin/env python3
"""Prepare immutable data/weights; no inference, training or held-out tuning."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cvpr_project.data import prepare
from cvpr_project.model import prepare_weights

if __name__ == "__main__":
    (ROOT / "artifacts").mkdir(exist_ok=True)
    os.chdir(ROOT)
    prepare()
    prepare_weights()
