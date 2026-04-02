"""
run_benchmark.py — wrapper that runs benchmark.py and logs all output to
benchmark.log (UTF-8) while also printing to the terminal in real time.

Usage (with venv active):
    python run_benchmark.py

Monitor from another terminal:
    Get-Content benchmark.log -Wait -Tail 30
"""

import subprocess
import sys
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "benchmark.log")
BENCHMARK = os.path.join(os.path.dirname(__file__), "seva_env", "benchmark.py")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONUTF8"] = "1"

print(f"[run_benchmark] Starting benchmark. Logging to: {LOG_FILE}")
print(f"[run_benchmark] Monitor with:  Get-Content benchmark.log -Wait -Tail 30\n")

with open(LOG_FILE, "w", encoding="utf-8") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", BENCHMARK],
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

print(f"\n[run_benchmark] Benchmark finished with exit code {process.returncode}.")
