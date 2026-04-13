"""Quick runner for 1% tier only."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from seva_benchmark_4060 import run, flush
import time

t0 = time.perf_counter()
out = run(100000, poison_ratio=0.01)
print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")
print(f"\n=== 1% TIER VALIDATION ===")
print(f"  ASR  = {out['asr']:.2f}% {'PASS' if out['asr'] <= 20 else 'FAIL'} (target <= 20%)")
print(f"  FPR  = {out['doc_fpr']:.2f}% (for reference)")
