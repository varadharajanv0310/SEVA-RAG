"""Smoke test: 2k corpus, 1% poison ratio."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from seva_benchmark_4060 import run, flush
import time

t0 = time.perf_counter()
out = run(2000, poison_ratio=0.01)
elapsed = time.perf_counter() - t0
print(f"\nTotal: {elapsed / 60:.1f} min ({elapsed:.0f}s)")
print(f"\n{'='*50}")
print(f"  SMOKE TEST RESULTS")
print(f"{'='*50}")
print(f"  ASR       = {out['asr']:.2f}%")
print(f"  FPR       = {out['doc_fpr']:.2f}%")
print(f"  atk_succ  = {out['attacks']['successes']}")
print(f"  atk_att   = {out['attacks']['attempts']}")
print(f"  ml_catches= {out['catches']['ml']}")
print(f"  TAU       = {out['tau']:.4f}")
print(f"{'='*50}")
asr_ok = out['asr'] < 50
fpr_ok = out['doc_fpr'] < 5
att_ok = out['attacks']['attempts'] > 100
print(f"  atk_att > 100:  {'PASS' if att_ok else 'FAIL'} ({out['attacks']['attempts']})")
print(f"  ASR < 50%:      {'PASS' if asr_ok else 'FAIL'} ({out['asr']:.1f}%)")
print(f"  FPR < 5%:       {'PASS' if fpr_ok else 'FAIL'} ({out['doc_fpr']:.1f}%)")
print(f"  Overall:        {'PASS' if (asr_ok and fpr_ok and att_ok) else 'FAIL'}")
