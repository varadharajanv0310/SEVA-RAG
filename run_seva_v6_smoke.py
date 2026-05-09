"""
SEVA v6.2.1 Multi-Seed Smoke Test — 10k corpus, 3 tiers (1%, 5%, 10%).

v6.2.1 changes vs v6.2:
  - BENIGN_Q scaled 500 → 2000: ~800 benign eval queries per tier (~3500 clean
    doc evaluations) for statistically robust FPR confidence intervals.
  - Comprehensive data logging: per-signal SNR/mean/std, raw TP/FP/TN/FN counts,
    weights+tau per layer in results JSON (signal_stats, counts fields).
  - Result filename: seva_v6_2_results_{ptag}_s{seed}.json

v6.2 changes vs v6:
  - Doc-level cluster_coh replaces query-level: immune to retrieval contamination.
  - Density-adaptive oracle calibration removed: FPR_TARGET applied uniformly.
  - Multi-seed validation: 3 cal/eval split seeds [42, 7, 123], reporting
    mean ± std to eliminate single-seed optimism bias.

Pass criteria (applied to per-seed means):
  mean_ASR <= ASR_LIMIT  AND  mean_FPR <= FPR_LIMIT  across all tiers & layers.
  Per-seed worst FPR is also reported; individual seeds may exceed FPR_LIMIT
  by up to +1pp without failing if the mean is within bounds.
"""
import subprocess, sys, os, json, io
import numpy as np

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

PYTHON = sys.executable
BENCH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seva_benchmark_4060.py")
CWD    = os.path.dirname(os.path.abspath(__file__))
ENV    = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

ASR_LIMIT   = 22.0
FPR_LIMIT   = 3.0
FPR_PER_SEED_SLACK = 1.0   # individual seeds may exceed FPR_LIMIT by this much

CAL_EVAL_SEEDS  = [42, 7, 123]
fpr_target      = 0.0069   # patched into benchmark before each run


def ptag(pr): return f"p{int(pr*1000):03d}"


# ── Cache management ─────────────────────────────────────────────────────────

def delete_p3_caches(poison_ratios, seeds):
    """Delete Phase 3 v6.2 caches for all seed variants so next run recalibrates."""
    for pr in poison_ratios:
        d = os.path.join(CWD, f"seva_checkpoints_4060_10k_{ptag(pr)}")
        for seed in seeds:
            p = os.path.join(d, f"p3_v6.2_s{seed:03d}.json")
            if os.path.exists(p):
                os.remove(p)
                print(f"  Deleted: {ptag(pr)}/p3_v6.2_s{seed:03d}.json")


def delete_p1_query_caches(poison_ratios):
    """Delete Phase 1 query caches to force query regeneration."""
    for pr in poison_ratios:
        d = os.path.join(CWD, f"seva_checkpoints_4060_10k_{ptag(pr)}")
        p = os.path.join(d, "p1_query.json")
        if os.path.exists(p):
            os.remove(p)
            print(f"  Deleted: {ptag(pr)}/p1_query.json (force regen with BENIGN_Q=2000)")


# ── Tier runner ──────────────────────────────────────────────────────────────

def run_tier(corpus, pr, log_path, cal_seed):
    args = [PYTHON, "-u", BENCH,
            "--corpus", str(corpus),
            "--poison-ratio", str(pr),
            "--cal-seed", str(cal_seed)]
    print(f"\n{'='*60}\n  {corpus//1000}k / {pr*100:.1f}%  seed={cal_seed}\n{'='*60}", flush=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(args, cwd=CWD, env=ENV,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lf.write(line)
        proc.wait()
    return proc.returncode


# ── Result loading ───────────────────────────────────────────────────────────

def load_results(pr, seed):
    rf = os.path.join(CWD, f"seva_v6_2_results_10k_{ptag(pr)}_s{seed:03d}.json")
    return json.load(open(rf, encoding="utf-8")) if os.path.exists(rf) else None


def load_p3(pr, seed):
    ck = os.path.join(CWD, f"seva_checkpoints_4060_10k_{ptag(pr)}",
                      f"p3_v6.2_s{seed:03d}.json")
    return json.load(open(ck, encoding="utf-8")) if os.path.exists(ck) else None


# ── FPR_TARGET patch ─────────────────────────────────────────────────────────

def update_fpr_target(new_target):
    """Patch FPR_TARGET constant in seva_benchmark_4060.py."""
    with open(BENCH, "r", encoding="utf-8") as f:
        src = f.read()
    import re
    src = re.sub(
        r'^FPR_TARGET\s*=\s*[\d.]+',
        f'FPR_TARGET = {new_target:.4f}',
        src, count=1, flags=re.MULTILINE
    )
    with open(BENCH, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  FPR_TARGET patched → {new_target:.4f}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

poison_ratios = [0.01, 0.05, 0.10]
all_sigs = ["topic_drift", "sent_unif", "kw_density", "doc_length_signal",
            "avg_sent_len_signal", "punct_signal", "content_ttr_signal", "cluster_coh"]

print("\n  Deleting old Phase 1 query caches (force BENIGN_Q=2000 regen)...")
delete_p1_query_caches(poison_ratios)

update_fpr_target(fpr_target)

# ── Multi-seed runs ───────────────────────────────────────────────────────────
# seed_results[seed][pr] = result_json or None
seed_results = {}

for seed in CAL_EVAL_SEEDS:
    print(f"\n{'#'*70}")
    print(f"  SEVA v6.2.1 SMOKE TEST — Seed {seed} / {CAL_EVAL_SEEDS}".center(70))
    print(f"  FPR_TARGET={fpr_target:.4f}   BENIGN_Q=2000   doc-level cluster_coh")
    print(f"{'#'*70}")

    delete_p3_caches(poison_ratios, [seed])
    seed_results[seed] = {}

    for pr in poison_ratios:
        log = os.path.join(CWD, f"smoke_v6.2.1_10k_{ptag(pr)}_s{seed:03d}.log")
        rc = run_tier(10000, pr, log, cal_seed=seed)
        if rc != 0:
            print(f"  WARNING: seed={seed} tier={pr*100:.1f}% exited with code {rc}")
        res = load_results(pr, seed)
        seed_results[seed][pr] = res
        if res is None:
            print(f"  WARNING: no result file found for seed={seed} pr={pr*100:.1f}%")


# ── Aggregate across seeds ────────────────────────────────────────────────────

W = 72
print(f"\n\n{'#'*W}")
print("  SEVA v6.2.1 SMOKE TEST — MULTI-SEED AGGREGATE RESULTS (10k)".center(W))
print(f"  Seeds: {CAL_EVAL_SEEDS}   FPR_TARGET={fpr_target:.4f}   ASR_LIMIT={ASR_LIMIT}%  FPR_LIMIT={FPR_LIMIT}%")
print(f"{'#'*W}")

# Collect per-seed worst values and per-layer/tier tables
layer_keys  = [("L1", "L1 — NAIVE"), ("L2", "L2 — STANDARD ADAPTIVE"), ("L3", "L3 — COMPOUND ADAPTIVE")]

overall_pass   = True
all_mean_asrs  = []
all_mean_fprs  = []

for layer_key, layer_label in layer_keys:
    print(f"\n{layer_label}:")
    print(f"  {'Tier':<5} | {'mean_ASR':>9} | {'std_ASR':>8} | {'mean_FPR':>9} | {'std_FPR':>8} | {'worst_FPR':>10} | {'Pass?':>6}")
    print(f"  {'-'*68}")
    for pr in poison_ratios:
        asrs, fprs = [], []
        for seed in CAL_EVAL_SEEDS:
            res = seed_results[seed].get(pr)
            if res and layer_key in res:
                asrs.append(res[layer_key]["asr"])
                fprs.append(res[layer_key]["doc_fpr"])

        if not asrs:
            print(f"  {pr*100:<5.1f} |  NO DATA")
            overall_pass = False
            continue

        mean_asr   = float(np.mean(asrs))
        std_asr    = float(np.std(asrs))
        mean_fpr   = float(np.mean(fprs))
        std_fpr    = float(np.std(fprs))
        worst_fpr  = float(np.max(fprs))
        all_mean_asrs.append(mean_asr)
        all_mean_fprs.append(mean_fpr)

        tier_pass = (mean_asr <= ASR_LIMIT and
                     mean_fpr <= FPR_LIMIT and
                     worst_fpr <= FPR_LIMIT + FPR_PER_SEED_SLACK)
        if not tier_pass:
            overall_pass = False
        status = "PASS" if tier_pass else "FAIL"

        print(f"  {pr*100:<5.1f} | {mean_asr:>9.2f} | {std_asr:>8.2f} | "
              f"{mean_fpr:>9.2f} | {std_fpr:>8.2f} | {worst_fpr:>10.2f} | {status:>6}")

# Per-seed breakdown
print(f"\n{'─'*W}")
print("  Per-seed worst_ASR / worst_FPR:")
for seed in CAL_EVAL_SEEDS:
    w_asr = w_fpr = 0.0
    for pr in poison_ratios:
        res = seed_results[seed].get(pr)
        if res:
            for lk, _ in layer_keys:
                if lk in res:
                    w_asr = max(w_asr, res[lk]["asr"])
                    w_fpr = max(w_fpr, res[lk]["doc_fpr"])
    print(f"  Seed {seed:>3}: worst_ASR={w_asr:.2f}%  worst_FPR={w_fpr:.2f}%")

print(f"\n{'─'*W}")
grand_asr = float(np.mean(all_mean_asrs)) if all_mean_asrs else float('nan')
grand_fpr = float(np.mean(all_mean_fprs)) if all_mean_fprs else float('nan')
print(f"  Grand mean across all seeds/tiers/layers:  ASR={grand_asr:.2f}%  FPR={grand_fpr:.2f}%")
print(f"\n  OVERALL: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
print(f"  Criteria: mean_ASR ≤ {ASR_LIMIT}%  AND  mean_FPR ≤ {FPR_LIMIT}%  "
      f"AND  worst_FPR ≤ {FPR_LIMIT + FPR_PER_SEED_SLACK}%")

# SNR table (from seed=42 tier reference)
print(f"\n{'─'*W}")
print("  SIGNAL SNR (doc-level cluster_coh, seed=42 ref):")
print(f"  {'Signal':<22} | {'1%':>6} | {'5%':>6} | {'10%':>6}")
print(f"  {'-'*46}")
for sig in all_sigs:
    row = f"  {sig:<22} |"
    for pr in poison_ratios:
        p3 = load_p3(pr, 42)
        if p3:
            val = p3.get("snrs", {}).get(sig, float("nan"))
            row += f" {val:>6.2f} |" if val == val else f" {'N/A':>6} |"
        else:
            row += f" {'N/A':>6} |"
    print(row)

print(f"\n{'#'*W}")
print("  END — awaiting explicit approval before 100k run".center(W))
print(f"{'#'*W}\n")
