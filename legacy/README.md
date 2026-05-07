# Legacy Files — v5 Era and Obsolete Analyses

These files pre-date SEVA v6.2.1 and are **not used by the current benchmark**. They are archived
here for historical reference only.

---

## Why they are here and NOT in the repo root

| File | Reason archived |
|------|----------------|
| `run_seva_v5l.py` | v5 driver — uses centroid_dist scoring replaced by cluster_coh in v6 |
| `run_seva_v5m.py` | v5 driver — same issue |
| `run_seva_v5n_smoke.py` | v5 smoke driver — same issue |
| `run_v5.py` | v5 entry point |
| `run_both_100k.py` | **Broken**: reads `p3_v5j.json` / `p3_v5k.json` cache files that v6.2.1 never creates; will silently return empty SNR tables |
| `cross_corpus_analysis.py` | **Broken**: reads `phase2_primary_embs.npy` — a path that does not exist under the v6 checkpoint schema (current file is `p2_pe.npy`) |
| `final_summary.md` | Stale: 1k / 2k / 5k results from v5 (ASR 12–28%, FPR 14–30%); superseded by Table IV of the paper (v6.2.1 at 100k) |
| `seva_results_analysis.md` | Stale: 1k-corpus triage from v5 where ASR = 100% — a pre-fix diagnostic, not a result; superseded entirely by v6.2.1 |

---

## Current entry points (repo root)

| File | Purpose |
|------|---------|
| `seva_benchmark_4060.py` | Core benchmark — v6.2.1 |
| `run_seva_v6_smoke.py` | 10k smoke test |
| `run_seva_v6_100k.py` | 100k production run |
| `adaptive_attack_seva.py` | Adaptive attack evaluator |
| `generate_poison_corpus.py` | Poison corpus generator |

See [HOW_TO_REPRODUCE.md](../HOW_TO_REPRODUCE.md) for end-to-end instructions.
