"""
Sequential 100k dual-run orchestrator for SEVA v5j/v5k comparison.

Run 1 (p3_v5j): L2 = topic_drift=0.25, sent_unif=0.25, doc_len=0.25, avg_sent_len=0.25
Run 2 (p3_v5k): L2 = topic_drift=0.35, sent_unif=0.35, doc_len=0.30, avg_sent_len=0.00

Saves RUN1_RESULTS and RUN2_RESULTS, prints final consolidated report.
"""

import subprocess, sys, os, json, time, shutil, re, io

# Force stdout to UTF-8 so ± and other Unicode characters don't crash the orchestrator
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

PYTHON = r"C:\Users\varad\miniconda3\envs\seva\python.exe"
BENCH  = r"C:\Users\varad\OneDrive\Desktop\SEVA\seva_benchmark_4060.py"
CWD    = r"C:\Users\varad\OneDrive\Desktop\SEVA"

# ─── helper: run benchmark, tee output to log, return combined output text ───────
def run_benchmark(label, log_path, extra_args=None):
    args = [PYTHON, "-u", BENCH, "--multitier", "--mtcorpus", "100000"]
    if extra_args:
        args += extra_args
    print(f"\n{'='*72}")
    print(f"  LAUNCHING {label}")
    print(f"  Log: {log_path}")
    print(f"{'='*72}\n", flush=True)

    # Force subprocess to use UTF-8 for its stdout
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    lines = []
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            args, cwd=CWD, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace"
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lines.append(line)
        proc.wait()

    print(f"\n  {label} subprocess exit code: {proc.returncode}", flush=True)
    return "".join(lines)


# ─── helper: load multitier summary JSON ────────────────────────────────────────
def load_summary():
    sf = os.path.join(CWD, "seva_multitier_summary.json")
    if not os.path.exists(sf):
        return None
    return json.load(open(sf, encoding="utf-8"))


# ─── helper: load SNRs from Phase 3 caches ──────────────────────────────────────
def load_snrs(poison_ratios):
    """Returns dict keyed by poison_ratio → snrs dict (from p3 JSON)."""
    PTAG = {0.01: "p010", 0.05: "p050", 0.10: "p100"}
    snrs_by_tier = {}
    for pr in poison_ratios:
        ptag = PTAG.get(pr, f"p{int(pr*1000):03d}")
        for ck_name in ("p3_v5k.json", "p3_v5j.json"):
            ck = os.path.join(CWD, f"seva_checkpoints_4060_100k_{ptag}", ck_name)
            if os.path.exists(ck):
                d = json.load(open(ck, encoding="utf-8"))
                snrs_by_tier[pr] = d.get("snrs", {})
                break
    return snrs_by_tier


# ─── helper: apply Run 2 code changes to benchmark file ─────────────────────────
def apply_run2_changes():
    print("\n  Applying Run 2 code changes to seva_benchmark_4060.py ...", flush=True)
    with open(BENCH, "r", encoding="utf-8") as f:
        src = f.read()

    # 1. Redistribute L2 weights (avg_sent_len=0.00, others redistributed)
    old_l2 = '''\
        self.L2_weights = {                  # Adaptive adversary (keyword-aware only, kw_density=0)
            "topic_drift":         0.25,
            "sent_unif":           0.25,
            "doc_length_signal":   0.25,
            "avg_sent_len_signal": 0.25,
            "kw_density":          0.00,
            "ttr_signal":          0.00,   # dropped
            "repeat_rate":         0.00,   # dropped
        }'''
    new_l2 = '''\
        self.L2_weights = {                  # Adaptive adversary (keyword-aware only, kw_density=0)
            "topic_drift":         0.35,
            "sent_unif":           0.35,
            "doc_length_signal":   0.30,
            "avg_sent_len_signal": 0.00,
            "kw_density":          0.00,
            "ttr_signal":          0.00,   # dropped
            "repeat_rate":         0.00,   # dropped
        }'''
    if old_l2 not in src:
        print("  ERROR: L2 weights pattern not found — skipping weight update", flush=True)
    else:
        src = src.replace(old_l2, new_l2, 1)
        print("  L2 weights updated: topic_drift=0.35, sent_unif=0.35, doc_len=0.30, avg_sent_len=0.00", flush=True)

    # 2. Rename checkpoint from p3_v5j → p3_v5k
    old_ck = 'ck = self._ck("p3_v5j.json")'
    new_ck = 'ck = self._ck("p3_v5k.json")'
    if old_ck not in src:
        print("  ERROR: checkpoint filename pattern not found — skipping rename", flush=True)
    else:
        src = src.replace(old_ck, new_ck, 1)
        print("  Phase 3 checkpoint renamed: p3_v5j.json → p3_v5k.json", flush=True)

    with open(BENCH, "w", encoding="utf-8") as f:
        f.write(src)
    print("  File saved.", flush=True)


# ─── helper: delete p3_v5j Phase 3 caches for the 3 tiers ───────────────────────
def delete_run1_phase3_caches():
    PTAGS = ["p010", "p050", "p100"]
    print("\n  Deleting Run 1 Phase 3 caches (p3_v5j.json) ...", flush=True)
    for ptag in PTAGS:
        ck = os.path.join(CWD, f"seva_checkpoints_4060_100k_{ptag}", "p3_v5j.json")
        if os.path.exists(ck):
            os.remove(ck)
            print(f"    Deleted: .../{ptag}/p3_v5j.json", flush=True)
        else:
            print(f"    Not found (ok): .../{ptag}/p3_v5j.json", flush=True)


# ─── helper: print consolidated final report ─────────────────────────────────────
def print_final_report(run1_summary, run2_summary, poison_ratios):
    W = 78
    print(f"\n\n{'#'*W}")
    print("  SEVA v5j/v5k FINAL CONSOLIDATED REPORT — 100k Corpus".center(W))
    print(f"{'#'*W}")

    header = (
        f"  {'Poison%':<9} {'P.Docs':<7} | "
        f"{'L1 ASR':<8} {'L1 FPR':<8} {'L1 Lat':>7} | "
        f"{'L2 ASR':<8} {'L2 FPR':<8} {'L2 Lat':>7}"
    )

    for run_label, summary in [("RUN 1 (v5j: avg_sent_len active)", run1_summary),
                                ("RUN 2 (v5k: avg_sent_len=0.00)", run2_summary)]:
        print(f"\n  ── {run_label} ──")
        print(f"  L1 = naive (kw_density active, topic_drift=0.25, sent_unif=0.35, kw_density=0.40)")
        if "v5j" in run_label:
            print(f"  L2 = standard adaptive (topic_drift=0.25, sent_unif=0.25, doc_len=0.25, avg_sent_len=0.25)")
        else:
            print(f"  L2 = geometric-only adaptive (topic_drift=0.35, sent_unif=0.35, doc_len=0.30, avg_sent_len=0.00)")
        print(header)
        print(f"  {'-'*W}")

        if summary is None:
            print(f"  [No results — run did not produce summary JSON]")
            continue

        for tr in summary.get("tiers", []):
            pr = tr["poison_ratio"]
            if tr.get("skipped"):
                print(f"  {pr*100:<9.1f} {tr['poisoned']:<7} | STOPPED (Phase 3 gate)")
            else:
                l1 = tr["L1"]; l2 = tr["L2"]
                print(
                    f"  {pr*100:<9.1f} {tr['poisoned']:<7} | "
                    f"{l1['asr']:>7.2f}% {l1['doc_fpr']:>7.2f}% {l1['latency']['mean']:>7.1f}ms | "
                    f"{l2['asr']:>7.2f}% {l2['doc_fpr']:>7.2f}% {l2['latency']['mean']:>7.1f}ms"
                )
        rt = summary.get("total_runtime_min", 0)
        print(f"  Total runtime: {rt:.1f} min")

    # SNR table (from Run 1's Phase 3 caches — Run 2 has same signal set minus avg_sent_len)
    snrs_by_tier = load_snrs(poison_ratios)
    if snrs_by_tier:
        print(f"\n  ── Signal SNRs (from Phase 3 calibration, 100k corpus) ──")
        sig_order = ["kw_density", "topic_drift", "sent_unif", "doc_length_signal", "avg_sent_len_signal",
                     "ttr_signal", "repeat_rate"]
        header_snr = f"  {'Signal':<22}" + "".join(f"  {p*100:.0f}% tier" for p in poison_ratios)
        print(header_snr)
        print(f"  {'-'*60}")
        for sig in sig_order:
            row = f"  {sig:<22}"
            for pr in poison_ratios:
                s = snrs_by_tier.get(pr, {})
                if sig in s:
                    row += f"  {s[sig]:>9.2f}"
                else:
                    row += f"  {'N/A':>9}"
            print(row)

    print(f"\n{'#'*W}")
    print("  END REPORT — awaiting explicit user approval before any further action".center(W))
    print(f"{'#'*W}\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    poison_ratios = [0.01, 0.05, 0.10]

    # ── RUN 1 ───────────────────────────────────────────────────────────────────
    log1 = os.path.join(CWD, "run1_100k_v5j.log")
    run_benchmark("RUN 1 (v5j — avg_sent_len active)", log1)
    run1_summary = load_summary()

    RUN1_RESULTS = run1_summary
    print("\n  RUN1_RESULTS saved in memory.", flush=True)

    # ── Prepare Run 2 ───────────────────────────────────────────────────────────
    apply_run2_changes()
    delete_run1_phase3_caches()

    print("\n  Waiting 120 seconds for GPU memory to stabilize ...", flush=True)
    time.sleep(120)

    # ── RUN 2 ───────────────────────────────────────────────────────────────────
    log2 = os.path.join(CWD, "run2_100k_v5k.log")
    run_benchmark("RUN 2 (v5k — avg_sent_len=0.00)", log2)
    run2_summary = load_summary()

    RUN2_RESULTS = run2_summary
    print("\n  RUN2_RESULTS saved in memory.", flush=True)

    # ── Final consolidated report ────────────────────────────────────────────────
    print_final_report(RUN1_RESULTS, RUN2_RESULTS, poison_ratios)
