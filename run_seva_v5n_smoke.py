"""
SEVA v5n Smoke Test — 10k corpus, 3 tiers (1%, 5%, 10%).
Fix: L3 scores recomputed from raw signals after weight adaptation.
"""
import subprocess, sys, os, json, io

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

PYTHON = r"C:\Users\varad\miniconda3\envs\seva\python.exe"
BENCH  = r"C:\Users\varad\OneDrive\Desktop\SEVA\seva_benchmark_4060.py"
CWD    = r"C:\Users\varad\OneDrive\Desktop\SEVA"
ENV    = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def ptag(pr): return f"p{int(pr*1000):03d}"


def run_tier(corpus, pr, log_path):
    args = [PYTHON, "-u", BENCH, "--corpus", str(corpus), "--poison-ratio", str(pr)]
    print(f"\n{'='*60}\n  {corpus//1000}k / {pr*100:.1f}%\n{'='*60}", flush=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(args, cwd=CWD, env=ENV,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lf.write(line)
        proc.wait()
    return proc.returncode


# Delete stale Phase 3 checkpoints for 10k tiers
for pr in [0.01, 0.05, 0.10]:
    d = os.path.join(CWD, f"seva_checkpoints_4060_10k_{ptag(pr)}")
    for stale in ["p3_v5l.json", "p3_v5m.json", "p3_v5n.json"]:
        p = os.path.join(d, stale)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Deleted: {ptag(pr)}/{stale}")

# Run 3 tiers
poison_ratios = [0.01, 0.05, 0.10]
for pr in poison_ratios:
    log = os.path.join(CWD, f"smoke_v5n_10k_{ptag(pr)}.log")
    run_tier(10000, pr, log)


# ── Report ───────────────────────────────────────────────────────────
W = 66
print(f"\n{'#'*W}")
print("  SEVA v5n SMOKE TEST RESULTS — 10k Corpus".center(W))
print(f"{'#'*W}")

all_sigs = ["topic_drift", "sent_unif", "kw_density", "doc_length_signal",
            "avg_sent_len_signal", "punct_signal", "content_ttr_signal"]

print(f"\nSIGNAL SNR:")
print(f"  {'Signal':<22} | {'1%':>6} | {'5%':>6} | {'10%':>6}")
print(f"  {'-'*46}")
for sig in all_sigs:
    row = f"  {sig:<22} |"
    for pr in poison_ratios:
        ck = os.path.join(CWD, f"seva_checkpoints_4060_10k_{ptag(pr)}", "p3_v5n.json")
        if os.path.exists(ck):
            d = json.load(open(ck, encoding="utf-8"))
            val = d.get("snrs", {}).get(sig, float("nan"))
            row += f" {val:>6.2f} |" if val == val else f" {'N/A':>6} |"
        else:
            row += f" {'N/A':>6} |"
    print(row)

for label, key, tpr_key in [("L1 — NAIVE", "L1", "cal_tpr_L1"),
                              ("L2 — STANDARD ADAPTIVE", "L2", "cal_tpr_L2"),
                              ("L3 — COMPOUND ADAPTIVE", "L3", "cal_tpr_L3")]:
    print(f"\n{label}:")
    print(f"  {'Tier':<5} | {'ASR%':>7} | {'FPR%':>7} | {'cal_TPR':>8} | {'tau':>8}")
    print(f"  {'-'*46}")
    for pr in poison_ratios:
        rf = os.path.join(CWD, f"seva_results_4060_10k_{ptag(pr)}.json")
        ck = os.path.join(CWD, f"seva_checkpoints_4060_10k_{ptag(pr)}", "p3_v5n.json")
        if os.path.exists(rf):
            res = json.load(open(rf, encoding="utf-8"))
            p3 = json.load(open(ck, encoding="utf-8")) if os.path.exists(ck) else {}
            layer = res[key]
            tpr = p3.get(tpr_key, float("nan"))
            tau_key = {"L1": "tau_L1", "L2": "tau_L2", "L3": "tau_L3"}[key]
            tau = p3.get(tau_key, float("nan"))
            tpr_str = f"{tpr*100:.1f}%" if tpr == tpr else "N/A"
            print(f"  {pr*100:<5.1f} | {layer['asr']:>7.2f} | {layer['doc_fpr']:>7.2f} | {tpr_str:>8} | {tau:>8.4f}")
        else:
            print(f"  {pr*100:<5.1f} | {'NO RESULT':>7}")

print(f"\n  L3 weights (after adaptation):")
ck1 = os.path.join(CWD, f"seva_checkpoints_4060_10k_p010", "p3_v5n.json")
if os.path.exists(ck1):
    d = json.load(open(ck1, encoding="utf-8"))
    w3 = d.get("L3_weights", {})
    active = {k: v for k, v in w3.items() if v > 0}
    print(f"  {active}")

print(f"\n{'#'*W}")
print("  END — awaiting explicit approval before 100k run".center(W))
print(f"{'#'*W}\n")
