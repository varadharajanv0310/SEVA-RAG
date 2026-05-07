"""
SEVA v5m Driver — 100k three-tier evaluation, Phase 2 cached.
L3_FPR_TARGET = 0.015 (tighter tau to fix punct_signal tail noise).

Hard stops:
  L3 FPR > 5%        → stop
  L1 ASR > 1%        → stop
  L2 ASR > 15%       → stop
  L3 ASR > 50% AND cal_tpr < 50% → stop
"""

import subprocess, sys, os, json, time, io

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

PYTHON = r"C:\Users\varad\miniconda3\envs\seva\python.exe"
BENCH  = r"C:\Users\varad\OneDrive\Desktop\SEVA\seva_benchmark_4060.py"
CWD    = r"C:\Users\varad\OneDrive\Desktop\SEVA"
ENV    = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

CHOUDHARY_STD      = 24.0
CHOUDHARY_ADAPTIVE = 35.0


def run_tier(poison_ratio, log_path):
    args = [PYTHON, "-u", BENCH, "--corpus", "100000", "--poison-ratio", str(poison_ratio)]
    print(f"\n{'='*72}\n  RUNNING: 100k, {poison_ratio*100:.1f}% poison\n  Log: {log_path}\n{'='*72}\n", flush=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(args, cwd=CWD, env=ENV,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lf.write(line)
        proc.wait()
    return proc.returncode


def ptag(pr): return f"p{int(pr*1000):03d}"

def load_results(pr):
    rf = os.path.join(CWD, f"seva_results_4060_100k_{ptag(pr)}.json")
    return json.load(open(rf, encoding="utf-8")) if os.path.exists(rf) else None

def load_p3(pr):
    ck = os.path.join(CWD, f"seva_checkpoints_4060_100k_{ptag(pr)}", "p3_v5m.json")
    return json.load(open(ck, encoding="utf-8")) if os.path.exists(ck) else None


poison_ratios = [0.01, 0.05, 0.10]
tier_results = {}
tier_p3 = {}
STOP_REASON = None

for pr in poison_ratios:
    log = os.path.join(CWD, f"v5m_100k_{ptag(pr)}.log")
    run_tier(pr, log)

    res = load_results(pr)
    p3  = load_p3(pr)

    if res is None:
        print(f"\n  HARD STOP: {pr*100:.1f}% tier produced no results (Phase 3 gate).")
        STOP_REASON = f"{pr*100:.1f}% Phase 3 stopped"
        break

    tier_results[pr] = res
    tier_p3[pr] = p3

    l1_asr = res["L1"]["asr"]; l1_fpr = res["L1"]["doc_fpr"]
    l2_asr = res["L2"]["asr"]; l2_fpr = res["L2"]["doc_fpr"]
    l3_asr = res["L3"]["asr"]; l3_fpr = res["L3"]["doc_fpr"]
    l3_tpr = p3.get("cal_tpr_L3", 0.0) if p3 else 0.0
    tau_l3 = p3.get("tau_L3", 0.0) if p3 else 0.0

    if l3_fpr > 5.0:
        print(f"\n  HARD STOP: L3 FPR={l3_fpr:.2f}% > 5% at {pr*100:.1f}% tier (tau_L3={tau_l3:.4f}).")
        STOP_REASON = f"L3 FPR={l3_fpr:.2f}% at {pr*100:.1f}%"
        break
    if l1_asr > 1.0:
        print(f"\n  HARD STOP: L1 ASR={l1_asr:.2f}% > 1% at {pr*100:.1f}% tier.")
        STOP_REASON = f"L1 ASR={l1_asr:.2f}% at {pr*100:.1f}%"
        break
    if l2_asr > 15.0:
        print(f"\n  HARD STOP: L2 ASR={l2_asr:.2f}% > 15% at {pr*100:.1f}% tier.")
        STOP_REASON = f"L2 ASR={l2_asr:.2f}% at {pr*100:.1f}%"
        break
    if l3_asr > 50.0 and l3_tpr < 0.50:
        print(f"\n  HARD STOP: L3 ASR={l3_asr:.2f}% > 50% AND cal_tpr={l3_tpr*100:.1f}% < 50% — signal collapsed.")
        STOP_REASON = f"L3 collapsed at {pr*100:.1f}%"
        break

    print(f"\n  Tier {pr*100:.1f}% OK — L1={l1_asr:.2f}% | L2={l2_asr:.2f}% | L3={l3_asr:.2f}% | L3_FPR={l3_fpr:.2f}%", flush=True)
    if pr != poison_ratios[-1]:
        print("  Waiting 30s for GPU memory...", flush=True)
        time.sleep(30)


# ─── FINAL REPORT ────────────────────────────────────────────────────────────────
W = 72
print(f"\n\n{'#'*W}")
print("  SEVA v5m FINAL RESULTS — 100k Corpus".center(W))
print(f"{'#'*W}")

all_sigs = ["topic_drift", "sent_unif", "kw_density", "doc_length_signal",
            "avg_sent_len_signal", "punct_signal"]

print(f"\nSIGNAL SNR (Phase 3):")
print(f"  {'Signal':<22} | {'1%':>6} | {'5%':>6} | {'10%':>6}")
print(f"  {'-'*46}")
for sig in all_sigs:
    row = f"  {sig:<22} |"
    for pr in poison_ratios:
        p3 = tier_p3.get(pr)
        val = p3["snrs"].get(sig, float("nan")) if p3 and "snrs" in p3 else float("nan")
        row += f" {val:>6.2f} |" if val == val else f" {'N/A':>6} |"
    print(row)

def layer_table(label, key, tpr_key):
    print(f"\n{label}:")
    print(f"  {'Tier':<5} | {'ASR%':>7} | {'FPR%':>7} | {'cal_TPR':>8} | {'Lat(ms)':>8}")
    print(f"  {'-'*46}")
    for pr in poison_ratios:
        res = tier_results.get(pr)
        p3  = tier_p3.get(pr)
        if res is None:
            print(f"  {pr*100:<5.1f} | {'STOPPED':>7}")
            continue
        layer = res[key]
        tpr = p3.get(tpr_key, float("nan")) if p3 else float("nan")
        tpr_str = f"{tpr*100:.1f}%" if tpr == tpr else "N/A"
        print(f"  {pr*100:<5.1f} | {layer['asr']:>7.2f} | {layer['doc_fpr']:>7.2f} | {tpr_str:>8} | {layer['latency']['mean']:>8.1f}")

layer_table("L1 — NAIVE ADVERSARY:", "L1", "cal_tpr_L1")
layer_table("L2 — STANDARD ADAPTIVE:", "L2", "cal_tpr_L2")
layer_table("L3 — COMPOUND ADAPTIVE:", "L3", "cal_tpr_L3")

print(f"\nMonotonicity:")
for key in ["L1", "L2", "L3"]:
    asrs = [tier_results[pr][key]["asr"] for pr in poison_ratios if pr in tier_results]
    if len(asrs) == 3:
        mono = "YES" if max(asrs) - min(asrs) <= 5.0 or asrs[0] <= asrs[1] <= asrs[2] else "NO"
        print(f"  {key}: {asrs[0]:.1f}%→{asrs[1]:.1f}%→{asrs[2]:.1f}% — {mono}")
    else:
        print(f"  {key}: {'→'.join(f'{a:.1f}%' for a in asrs)} — (incomplete)")

print(f"\nChoudhary comparison at 10% density:")
res_10 = tier_results.get(0.10)
if res_10:
    l2 = res_10["L2"]["asr"]; l3 = res_10["L3"]["asr"]
    print(f"  L2 ASR vs Choudhary standard 24%:  SEVA L2 {l2:.2f}% — {'BEATS' if l2 < CHOUDHARY_STD else 'LOSES'}")
    print(f"  L3 ASR vs Choudhary adaptive 35%:  SEVA L3 {l3:.2f}% — {'BEATS' if l3 < CHOUDHARY_ADAPTIVE else 'LOSES'}")
else:
    print("  10% tier not completed.")

if STOP_REASON:
    print(f"\n  NOTE: Stopped early — {STOP_REASON}")

print(f"\n{'#'*W}")
print("  END REPORT — awaiting explicit user approval before any further action".center(W))
print(f"{'#'*W}\n")
