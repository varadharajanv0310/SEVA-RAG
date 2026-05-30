# PREREG_SCALE.md — pre-registration for the calibration-scale + high-encounter-CI experiments

**Written & committed BEFORE reading any result.** 2026-06-02. In-domain Security-SE, cross-platform
hard-gate infrastructure (`xplat_handoff/`).

## Frozen invariants (must not change)
- **Detector byte-frozen.** `cluster_coh` / `_compute_doc_coh` / `_score` / `_snr_weights` and every
  constant (K=5, K_FETCH=20, EMB_DIM=1024, INDEX_M=32, INDEX_EF=200, FPR_TARGET=0.0069) untouched.
  All math reused VERBATIM from `xplat_handoff/seva_xplat_common.py` (`doc_coh_full`, `retrieve_topk`,
  `sha256_corpus_canonical`, `embed_resumable`) and `xplat_poison_gen.generate_corpus` (deterministic).
- **`cluster_coh` HARD GATE only** — flag iff `cluster_coh > tau`; the v7.1.3 10-signal composite is
  NEVER constructed or scored.
- **Calibration is NON-ORACLE:** `tau` = (1 − FPR_TARGET) percentile of CLEAN `cluster_coh` only — no
  poison knowledge, no oracle density.
- **Corpus identity gate (every run):** the clean corpus is verified against canonical
  `28ec38114ee64e6010ec489d01e6d3ee13d9b3758fd30a169c99ed078732f8a9` (order-sensitive) AND
  `corpus_fingerprint.txt` (doc-by-doc). **STOP on any mismatch** — a divergent corpus invalidates the
  comparison. Poison is regenerated and hash-checked against `4f7ee3f3…a733c`.

## Pre-registered expectations (fixed before results)
1. **`cluster_coh` gap is density-invariant:** across the 3 densities, gap range ≤ 0.05 and every gap > 0.15.
2. **Templated ASR = 0%** on every condition (hash-matched corpus).
3. **Doc-FPR converges toward the 0.69% target as N grows:** the held-out (eval-half) doc-level FPR at
   N=10k OVER-SHOOTS the 0.69% target by MORE than at N=100k — the O(1/√N) percentile-estimation
   direction (τ estimated from a smaller calibration set is noisier).

**Private-negative rule:** a non-convergent or RISING Doc-FPR with N, OR any ASR > 0 on a hash-matched
corpus, is a PRIVATE NEGATIVE — reported with its JSON, NOT auto-published or folded into the paper.

## Experiment A — calibration scale points (hard gate, frozen, non-oracle)
Hash-verify the canonical 100k corpus; form the deterministic 10k subset = `corpus[:10000]` of that
verified corpus. For each **N ∈ {10000, 100000} × density ∈ {1%, 5%, 10%} × seed ∈ {42, 7, 123}**:
regenerate+hash-check the templated poison, inject at `corpus[0:P]` (P = round(N·density)), embed
(chunked/resumable/stderr-logged), **60/40 cal/eval split of the clean docs (seeded)**, τ =
(1−FPR_TARGET) percentile of the **cal-half** clean coh (no poison sight), then **score the eval half**.
Record per cell: `gap, snr, asr_pct, tau, n_clean_eval, docfpr_eval_doclevel_pct` (the held-out
doc-level FPR = the convergence metric), `docfpr_benign_retrieval_pct, query_fpr_ge1_pct,
query_fpr_ge2_pct`. Emit `result_scale10k.json` / `result_scale100k.json` with `corpus.{canonical_sha256,
hash_match, fingerprint_check, n_docs}`, `detector.{type:"cluster_coh_hard_gate", composite_used:false}`,
the 9-cell grid, and `scale_summary.grand_mean_docfpr_pct` per N. Commit each (no push).

## Experiment B — high-encounter Wilson CI (in-domain templated, frozen gate)
On the 100k canonical corpus at the frozen non-oracle τ_coh, expand the templated set to ≈25,000
templated-poison gate encounters (generate 25k deterministic templated poison, inject, score each with
the frozen hard gate); count evasions (coh ≤ τ_coh). **Pre-registered:** 0 evasions → ASR 0% with 95%
Wilson upper ≈ z²/(n+z²) ≈ 0.015% at n≈25k. Emit `result_hienc_ci.json` (n_encounters, evasions,
asr_pct, wilson_upper_pct, corpus hash, detector type). Commit (no push).

## Deferred
The 2M-document scale is explicitly NOT run now. On failure: re-run the same (resumable) command or
report the error — do not "fix" the science.
