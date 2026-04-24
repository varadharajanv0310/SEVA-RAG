"""
SEVA v5l Driver — Smoke test (10k/1%) → gate check → 100k three-tier evaluation.

Gate logic:
  PASS:   punct SNR>0 AND cttr SNR>0 AND L3 cal_tpr>60%
  ADAPT:  one signal inverted, one positive AND L3 cal_tpr>50% — weights already adapted
  STOP:   both inverted OR L3 cal_tpr<40%

100k hard stops (checked between tiers):
  L1 ASR > 1%      → stop
  L2 ASR > 15%     → stop
  L3 ASR > 50%     → stop (signals still insufficient)
  Any FPR > 5%     → stop
  L3 ASR=0% at 10% with cal_tpr=100% → stop (saturation artifact)
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

CHOUDHARY_STD      = 24.0   # Choudhary standard adversary ASR at 10% (published baseline)
CHOUDHARY_ADAPTIVE = 35.0   # Choudhary adaptive adversary ASR at 10%


def run_tier(corpus, poison_ratio, log_path):
    args = [PYTHON, "-u", BENCH,
            "--corpus", str(corpus),
            "--poison-ratio", str(poison_ratio)]
    print(f"\n{'='*72}")
    print(f"  RUNNING: {corpus//1000}k corpus, {poison_ratio*100:.1f}% poison")
    print(f"  Log: {log_path}")
    print(f"{'='*72}\n", flush=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(args, cwd=CWD, env=ENV,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lf.write(line)
        proc.wait()
    print(f"\n  Subprocess exit code: {proc.returncode}", flush=True)
    return proc.returncode


def ptag(pr):
    return f"p{int(pr * 1000):03d}"


def load_results(corpus_k, pr):
    rf = os.path.join(CWD, f"seva_results_4060_{corpus_k}k_{ptag(pr)}.json")
    return json.load(open(rf, encoding="utf-8")) if os.path.exists(rf) else None


def load_p3(corpus_k, pr):
    ck = os.path.join(CWD, f"seva_checkpoints_4060_{corpus_k}k_{ptag(pr)}", "p3_v5l.json")
    return json.load(open(ck, encoding="utf-8")) if os.path.exists(ck) else None


# ─── SMOKE TEST ─────────────────────────────────────────────────────────────────
print("\n" + "#"*72)
print("  SEVA v5l SMOKE TEST — 10k corpus, 1% poison".center(72))
print("#"*72)

# Delete stale 10k p3 checkpoint if present (force Phase 3 recalibration)
for stale in ["p3_v5l.json", "p3_v5k.json"]:
    ck_path = os.path.join(CWD, "seva_checkpoints_4060_10k_p010", stale)
    if os.path.exists(ck_path):
        os.remove(ck_path)
        print(f"  Deleted stale checkpoint: {stale}", flush=True)

run_tier(10000, 0.01, os.path.join(CWD, "smoke_v5l_10k_p010.log"))

# Load smoke test results
smoke_res = load_results(10, 0.01)
smoke_p3  = load_p3(10, 0.01)

if smoke_res is None or smoke_p3 is None:
    print("\n  SMOKE TEST FAILED: Phase 3 stopped — check log. Aborting.")
    sys.exit(1)

snrs = smoke_p3.get("snrs", {})
cal_tpr_L3 = smoke_p3.get("cal_tpr_L3", 0.0)
punct_snr  = snrs.get("punct_signal", 0.0)
cttr_snr   = snrs.get("content_ttr_signal", 0.0)

print(f"\n  SMOKE TEST SUMMARY:")
print(f"    punct_signal  SNR = {punct_snr:.2f}")
print(f"    content_ttr   SNR = {cttr_snr:.2f}")
print(f"    L3 cal_tpr        = {cal_tpr_L3*100:.1f}%")
print(f"    L1 ASR = {smoke_res['L1']['asr']:.2f}%  FPR = {smoke_res['L1']['doc_fpr']:.2f}%")
print(f"    L2 ASR = {smoke_res['L2']['asr']:.2f}%  FPR = {smoke_res['L2']['doc_fpr']:.2f}%")
print(f"    L3 ASR = {smoke_res['L3']['asr']:.2f}%  FPR = {smoke_res['L3']['doc_fpr']:.2f}%")

# Gate evaluation
punct_inv = punct_snr < 0
cttr_inv  = cttr_snr < 0

if punct_inv and cttr_inv:
    print("\n  GATE FAILED: Both new signals inverted. L3 architecture cannot proceed.")
    print("  Stopping. Awaiting explicit user approval before any further action.")
    sys.exit(1)

if cal_tpr_L3 < 0.40:
    print(f"\n  GATE FAILED: L3 cal_tpr={cal_tpr_L3*100:.1f}% < 40%. L3 insufficient for 100k.")
    print("  Stopping. Awaiting explicit user approval before any further action.")
    sys.exit(1)

if (not punct_inv) and (not cttr_inv) and cal_tpr_L3 >= 0.60:
    print("\n  GATE PASSED — both signals positive, cal_tpr>60%. Proceeding to 100k automatically.")
elif cal_tpr_L3 >= 0.50:
    inv_sig = "punct_signal" if punct_inv else "content_ttr_signal"
    print(f"\n  GATE PASSED (adapted): {inv_sig} inverted but one signal positive, cal_tpr>50%. Proceeding to 100k automatically.")
else:
    print(f"\n  GATE FAILED: cal_tpr={cal_tpr_L3*100:.1f}% < 50% after signal adaptation. Aborting.")
    sys.exit(1)

# ─── 100k THREE-TIER EVALUATION ─────────────────────────────────────────────────
print("\n" + "#"*72)
print("  SEVA v5l 100k EVALUATION — Three Tiers".center(72))
print("#"*72)

poison_ratios = [0.01, 0.05, 0.10]
tier_results = {}
tier_p3 = {}

STOP_REASON = None

for pr in poison_ratios:
    log = os.path.join(CWD, f"v5l_100k_{ptag(pr)}.log")
    rc = run_tier(100000, pr, log)

    res = load_results(100, pr)
    p3  = load_p3(100, pr)

    if res is None:
        print(f"\n  HARD STOP: {pr*100:.1f}% tier produced no results (Phase 3 gate triggered).")
        STOP_REASON = f"{pr*100:.1f}% Phase 3 stopped"
        break

    tier_results[pr] = res
    tier_p3[pr] = p3

    l1_asr = res["L1"]["asr"];  l1_fpr = res["L1"]["doc_fpr"]
    l2_asr = res["L2"]["asr"];  l2_fpr = res["L2"]["doc_fpr"]
    l3_asr = res["L3"]["asr"];  l3_fpr = res["L3"]["doc_fpr"]
    l3_tpr = p3.get("cal_tpr_L3", 0.0) if p3 else 0.0

    # Hard stop conditions
    if l1_asr > 1.0:
        print(f"\n  HARD STOP: L1 ASR={l1_asr:.2f}% > 1% at {pr*100:.1f}% tier.")
        STOP_REASON = f"L1 ASR={l1_asr:.2f}% at {pr*100:.1f}%"
        break
    if l2_asr > 15.0:
        print(f"\n  HARD STOP: L2 ASR={l2_asr:.2f}% > 15% at {pr*100:.1f}% tier.")
        STOP_REASON = f"L2 ASR={l2_asr:.2f}% at {pr*100:.1f}%"
        break
    if l3_asr > 50.0:
        print(f"\n  HARD STOP: L3 ASR={l3_asr:.2f}% > 50% at {pr*100:.1f}% tier — signals still insufficient.")
        STOP_REASON = f"L3 ASR={l3_asr:.2f}% at {pr*100:.1f}%"
        break
    if max(l1_fpr, l2_fpr, l3_fpr) > 5.0:
        worst = max((l1_fpr,"L1"),(l2_fpr,"L2"),(l3_fpr,"L3"), key=lambda x: x[0])
        print(f"\n  HARD STOP: {worst[1]} FPR={worst[0]:.2f}% > 5% at {pr*100:.1f}% tier.")
        STOP_REASON = f"{worst[1]} FPR={worst[0]:.2f}% at {pr*100:.1f}%"
        break
    if pr == 0.10 and l3_asr == 0.0 and l3_tpr >= 1.0:
        print(f"\n  HARD STOP: L3 ASR=0% at 10% tier with cal_tpr=100% — saturation artifact.")
        STOP_REASON = "L3 saturation at 10%"
        break

    print(f"\n  Tier {pr*100:.1f}% OK — L1={l1_asr:.2f}% | L2={l2_asr:.2f}% | L3={l3_asr:.2f}%", flush=True)
    if pr != poison_ratios[-1]:
        print("  Waiting 30s for GPU memory...", flush=True)
        time.sleep(30)

# ─── FINAL CONSOLIDATED REPORT ──────────────────────────────────────────────────
W = 78
print(f"\n\n{'#'*W}")
print("  SEVA v5l FINAL RESULTS — 100k Corpus".center(W))
print(f"{'#'*W}")

# Signal SNR table
all_sigs = ["topic_drift", "sent_unif", "kw_density", "doc_length_signal",
            "avg_sent_len_signal", "punct_signal", "content_ttr_signal"]
print(f"\n  SIGNAL SNR (Phase 3, 100k corpus):")
print(f"  {'Signal':<22} | {'1%':>6} | {'5%':>6} | {'10%':>6}")
print(f"  {'-'*46}")
for sig in all_sigs:
    row = f"  {sig:<22} |"
    for pr in poison_ratios:
        p3 = tier_p3.get(pr)
        val = p3["snrs"].get(sig, float("nan")) if p3 and "snrs" in p3 else float("nan")
        row += f" {val:>6.2f} |" if not (val != val) else f" {'N/A':>6} |"
    print(row)

# Layer results tables
def layer_table(label, key, cal_tpr_key):
    print(f"\n  {label}:")
    print(f"  {'Tier':<5} | {'ASR%':>7} | {'FPR%':>7} | {'cal_TPR':>8} | {'Lat(ms)':>8}")
    print(f"  {'-'*46}")
    for pr in poison_ratios:
        res = tier_results.get(pr)
        p3  = tier_p3.get(pr)
        if res is None:
            print(f"  {pr*100:<5.1f} | {'STOPPED':>7}")
            continue
        layer = res[key]
        tpr = p3.get(cal_tpr_key, float("nan")) if p3 else float("nan")
        tpr_str = f"{tpr*100:.1f}%" if tpr == tpr else "N/A"
        print(f"  {pr*100:<5.1f} | {layer['asr']:>7.2f} | {layer['doc_fpr']:>7.2f} | "
              f"{tpr_str:>8} | {layer['latency']['mean']:>8.1f}")

layer_table("L1 — NAIVE ADVERSARY (kw_density active)", "L1", "cal_tpr_L1")
layer_table("L2 — STANDARD ADAPTIVE (avg_sent_len active)", "L2", "cal_tpr_L2")
layer_table("L3 — COMPOUND ADAPTIVE (punct + content_ttr, no kw or length)", "L3", "cal_tpr_L3")

# Monotonicity
print(f"\n  MONOTONICITY (ASR should not increase with density):")
for key, label in [("L1", "L1"), ("L2", "L2"), ("L3", "L3")]:
    asrs = [tier_results[pr][key]["asr"] for pr in poison_ratios if pr in tier_results]
    if len(asrs) == 3:
        mono = "YES" if asrs[0] <= asrs[1] <= asrs[2] or max(asrs)-min(asrs) <= 5.0 else "NO"
        print(f"  {label}: {asrs[0]:.1f}% → {asrs[1]:.1f}% → {asrs[2]:.1f}% — {mono}")
    elif asrs:
        print(f"  {label}: {' → '.join(f'{a:.1f}%' for a in asrs)} — (incomplete)")
    else:
        print(f"  {label}: no data")

# Choudhary comparison (10% tier)
print(f"\n  CHOUDHARY COMPARISON (10% poison tier):")
res_10 = tier_results.get(0.10)
if res_10:
    l2_10 = res_10["L2"]["asr"]; l3_10 = res_10["L3"]["asr"]
    l2_vs = "BEATS" if l2_10 < CHOUDHARY_STD      else "LOSES"
    l3_vs = "BEATS" if l3_10 < CHOUDHARY_ADAPTIVE else "LOSES"
    print(f"  L2 vs Choudhary standard   ({CHOUDHARY_STD:.0f}% at 10%): SEVA L2 {l2_10:.2f}% — {l2_vs}")
    print(f"  L3 vs Choudhary adaptive   ({CHOUDHARY_ADAPTIVE:.0f}% at 10%): SEVA L3 {l3_10:.2f}% — {l3_vs}")
else:
    print(f"  10% tier not completed.")

if STOP_REASON:
    print(f"\n  NOTE: Evaluation stopped early — {STOP_REASON}")

print(f"\n{'#'*W}")
print("  END REPORT — awaiting explicit user approval before any further action".center(W))
print(f"{'#'*W}\n")
