# STEP3_GATE_RESULTS.md — authoritative Step‑3 in‑domain baseline gate record

**Run:** `seva_benchmark_4060.py --multitier --mtcorpus 100000 --poison-ratio 0.01 --cal-seed 42 --clean-corpus …/clean_corpus_security.json --benign-queries …/benign_queries_security.json --corpus-tag secqa` (47.9 min, exit 0, RTX 5080).
**Provenance:** `seva_v6_2_results_100k_secqa_p{010,050,100}_s042.json` (repo root, gitignored; backed up in `seva_results_5080_secqa_backup_20260530/`). **Seed 42 only** (gate); cluster_coh gap is seed‑independent, ASR/FPR are seed‑42 point estimates. Full tables need seeds 7,123.
**Corpus:** A1 = Security Stack Exchange Q&A (questions title+body + best‑answers), 100k, deduped; clean‑cohesion precheck PASS (sample 0.697; full‑corpus 0.705 ≤ 0.80).
**Detector:** byte‑frozen (diff confined to argparse/`__init__`/phase‑1; verified). All changes input‑side.

---

## 1. `cluster_coh` (clean / poison / gap / SNR) — in‑domain vs WikiText baseline

| density | in‑domain | WikiText (s42) | gap verdict |
|---|---|---|---|
| 1% | 0.7518 / 0.9871 / **+0.2353** / 5.99 | 0.7302 / 0.9871 / +0.2569 / 4.72 | HOLDS (SNR↑) |
| 5% | 0.7526 / 0.9909 / **+0.2384** / 6.00 | 0.7322 / 0.9909 / +0.2587 / 4.79 | HOLDS (SNR↑) |
| 10% | 0.7466 / 0.9939 / **+0.2474** / 5.78 | 0.7286 / 0.9939 / +0.2654 / 4.66 | HOLDS (SNR↑) |

→ confound‑kill CONFIRMED for cluster_coh: gap robustly positive, SNR *higher* in‑domain.

## 2. L1 ASR & DocFPR (per layer × density) — in‑domain

| metric | 1% | 5% | 10% |
|---|---|---|---|
| **L1 ASR** | 0.0% | 0.0% | 0.0% |
| L2 ASR | 43.2% | 60.9% | 73.1% |
| L3 ASR | 44.0% | 60.9% | 73.1% |
| DocFPR L1 | 0.454% | 0.521% | 0.404% |
| DocFPR L2 | 0.454% | 0.460% | 0.311% |
| DocFPR L3 | 0.485% | 0.460% | 0.311% |

(target FPR 0.69%; all DocFPR ≤ target — did NOT balloon with CF‑008 independent benign queries.) WikiText L1/L2/L3 ASR (s42): 0/0/1.6 (1%), 0/1.6/17.6 (5%), 0/0/15.2 (10%).

## 3. Linguistic‑signal SNR collapse (in‑domain vs WikiText) — Limitation‑2 cross‑check

| SNR | 1% in/WT | 5% in/WT | 10% in/WT |
|---|---|---|---|
| **kw_density** | 8.10 / 38.42 | 6.52 / 34.80 | 6.57 / 32.85 |
| avg_sent_len | 0.53 / 1.55 | 0.40 / 1.38 | 0.37 / 1.39 |
| ttr_signal | −1.12 / −1.77 | −1.10 / −1.90 | −1.09 / −1.70 |
| content_ttr | −1.09 / −1.96 | −1.05 / −2.18 | −1.08 / −1.92 |

→ kw_density (and TTR/avg_sent_len) were partly detecting security‑vs‑Wikipedia DOMAIN; their SNR collapses in‑domain.

## 4. The L2/L3 ASR explosion (in‑domain vs WikiText s42) — THE FINDING

| ASR | 1% | 5% | 10% |
|---|---|---|---|
| L2 in / WT | **43.2 / 0.0** | **60.9 / 1.6** | **73.1 / 0.0** |
| L3 in / WT | **44.0 / 1.6** | **60.9 / 17.6** | **73.1 / 15.2** |

→ With ALL signals (L1) the detector still catches everything in‑domain (ASR 0). Once the adaptive adversary evades kw_density (L2) [+avg_sent_len (L3)], `cluster_coh` alone cannot hold in‑domain → ASR 44–73%, worsening with density. SEVA's adaptive robustness was substantially propped up by the domain‑confounded kw_density signal.

## 5. Misc
- `norm_config` (in‑domain 1%): doc_length 339.1, sent_length 26.89, punct_density 0.1649 (WikiText 1%: 234.0/35.26/0.1197 — security docs longer, shorter sentences, more punctuation).
- R‑4 within‑5% ratio (kw_density/avg_sent_len SNR): **16.3×** in‑domain (was 25.3× WikiText).
- Latency (Phase‑4 per‑query, in‑domain, mean/p95 ms by tier — verified from result JSONs): 1% = 15.71/18.82, 5% = 13.85/15.74, 10% = 13.68/15.41. Sub‑30 ms invariant holds at every tier (worst p95 = 18.8 ms @1%).

## Gate verdict
On the author's stated criteria (gap holds, FPR near target, L1 ASR ≈ 0, kw_density SNR drops) → **PASS**. But §4 (L2/L3 ASR explosion) is a paper‑changing finding → see `STEP3_FINDING_AND_DECISION.md`. Author go/no‑go pending; do NOT proceed to E5/E1 without it.
