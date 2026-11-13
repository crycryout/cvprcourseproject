#!/usr/bin/env python3
"""Collect hardware/software facts without public hostnames or network addresses."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from batchserve.common import write_json

PROBE = r'''
import json,os,platform,subprocess
cpu={}
for line in open('/proc/cpuinfo'):
    if ':' in line:
        k,v=line.split(':',1)
        if k.strip() in ['model name','cpu cores','siblings']:
            cpu[k.strip()]=v.strip()
mem={k:v.strip() for k,v in (line.split(':',1) for line in open('/proc/meminfo')) if k in ['MemTotal']}
drivers=sorted(set(subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()))
print(json.dumps({'cpu':cpu,'logical_cpus':os.cpu_count(),'memory':mem,'kernel':platform.release(),'python':platform.python_version(),'machine':platform.machine(),'nvidia_driver_versions':drivers}))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remote-alias", required=True)
    p.add_argument("--output", default="artifacts/environment.json")
    args = p.parse_args()
    import shlex
    local = json.loads(subprocess.check_output([sys.executable, "-c", PROBE], text=True))
    remote = json.loads(subprocess.check_output(["ssh", "-o", "BatchMode=yes", args.remote_alias,
        shlex.join(["CISC8003Project/.venv/bin/python", "-c", PROBE])], text=True))
    write_json(ROOT / args.output, {"protocol_version": 3, "H800_worker": local, "RTX5090_worker": remote})


if __name__ == "__main__":
    main()
