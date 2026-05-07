import subprocess
import sys
import os

def run_benchmark(corpus_size):
    print(f"\n{'='*80}")
    print(f" STARTING BENCHMARK FOR {corpus_size} CORPUS ")
    print(f"{'='*80}\n")
    
    log_file = f"benchmark_{corpus_size}.log"
    cmd = [
        sys.executable, "-u", "seva_env/benchmark.py", 
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
        print(f"\nERROR: Benchmark for {corpus_size} failed with exit code {process.returncode}")
        return False
    return True

if __name__ == "__main__":
    sizes = [1000, 2000, 5000]
    for size in sizes:
        success = run_benchmark(size)
        if not success:
            print("Aborting subsequent runs due to error.")
            sys.exit(1)
    
    print("\nAll benchmark runs completed successfully.")
