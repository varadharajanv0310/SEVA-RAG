import subprocess
import sys
import os

def run_50k():
    print(f"\n{'='*80}")
    print(f" STARTING FINAL 50K HYBRID BENCHMARK ")
    print(f"{'='*80}\n")
    
    corpus_size = 50000
    log_file = f"benchmark_{corpus_size}_hybrid.log"
    
    # We use --reset to ensure a clean 50k run
    cmd = [
        "seva_env\\Scripts\\python.exe", "-u", "seva_env/benchmark.py", 
        "--corpus", str(corpus_size), "--reset"
    ]
    
    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        process.wait()
        
    if process.returncode != 0:
        print(f"\nERROR: 50k Benchmark failed with exit code {process.returncode}")
        return False
    return True

if __name__ == "__main__":
    run_50k()
