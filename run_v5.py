"""
run_v5.py — wrapper that runs seva_benchmark_4060.py and logs all output.

Usage:
    python run_v5.py
    python run_v5.py --reset
    python run_v5.py --fast
    python run_v5.py --corpus 2000
    python run_v5.py --corpus 2000 5000

Monitor:
    Get-Content benchmark_v5.log -Wait -Tail 30
"""

import subprocess
import sys
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "benchmark_v5.log")
BENCHMARK = os.path.join(os.path.dirname(__file__), "seva_benchmark_4060.py")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONUTF8"] = "1"

extra_args = sys.argv[1:]

print(f"[run_v5] Starting benchmark. Logging to: {LOG_FILE}")
print(f"[run_v5] Args: {extra_args}")
print(f"[run_v5] Monitor with:  Get-Content benchmark_v5.log -Wait -Tail 30\n")

with open(LOG_FILE, "w", encoding="utf-8") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", BENCHMARK] + extra_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()

    process.wait()

print(f"\n[run_v5] Benchmark finished with exit code {process.returncode}.")
