# Step-1 "R" Reframes — proposed text-only edits for the SEVA manuscript

**Status:** PROPOSALS for the author to apply/refine in the manuscript. The paper LaTeX source is **not** in this repo (only the PDF), so these are a spec, not direct edits. Review wording before putting it in submittable prose.
**Scope:** the text-only "R" fixes from EXPERIMENT_PLAN.md (rev. 5), Section 2 → R. No experiments; detector byte-frozen.
**Stability flags:** **[stable]** = does not depend on any experiment outcome; **[provisional → E2]** = the specific number changes after the E2 in-domain re-baseline and must be recomputed then.

---

## R-1 · CF-006 — "We prove" → "we argue/show" **[stable]**
- **Location:** §I-C contribution C3; §V-E.
- **Current (per KNOWN_ISSUES CF-006):** "We **prove** that cluster_coh is **invariant** under adversarial normalization." §V-E gives an informal mechanistic argument — no theorem, no proof.
- **Proposed:** drop "prove"/"invariant". e.g. C3 → "We **show** (empirically) and **argue** (mechanistically, §V-E) that cluster_coh is **robust to** adversarial normalization." In §V-E, label the argument explicitly: "This is a **mechanistic argument, not a formal proof**."
- **Rationale:** "prove" with no theorem is an easy rigor attack and a research-integrity smell.
- **Post-experiment upgrade path:** after E1b, if it returns "necessity," you may strengthen to "we **show** that suppressing cluster_coh to the clean band requires sacrificing retrievability (§E1b)" — a defensible empirical claim, still not "prove."

## R-2 · CF-005 — drop "hardware-agnostic"; substitute the verified CUDA reproduction + honest MPS caveat **[stable]**
- **Location:** Abstract; §I-C C6; §IX.
- **Current:** SEVA is "hardware-agnostic" — directly contradicted by §VI-F (layer-L3 ASR 0% on M4 vs 17.07% on RTX 4060).
- **Proposed:** remove "hardware-agnostic." Substitute: "SEVA's detection is **reproducible across CUDA GPUs**: we verify **bit-faithful per-seed reproduction on an RTX 5080 (Blackwell, sm_120) exactly matching the RTX 4060 baseline (seeds 42 & 7)**. Calibration is **backend-sensitive**: under Apple MPS, layer-L3 calibration differs (L3 ASR 0% vs 17% on CUDA, §VI-F), which we attribute to **calibration, not architecture** (Limitation 3)."
- **Rationale:** converts a false claim + a documented contradiction into a verified reproducibility strength + an honest, scoped caveat.
- **Provenance:** `seva_results_5080_baseline_backup_20260529/VERIFICATION_SUMMARY.md` (5080≡4060 per-seed, exact).

## R-3 · CF-001 — calibration iterations 100 → 50 **[stable]**
- **Location:** §IV-E.
- **Current:** "100 binary-search iterations." **Code:** `seva_benchmark_4060.py:~767` does `range(50)`.
- **Proposed:** change "100" → "**50** binary-search iterations."
- **Rationale:** paper/code mismatch; the locked, verified baseline used 50. (Do NOT re-run with 100 — it would invalidate the verified baseline for no scientific gain.)

## R-4 · CF-004 — "~28×" cross-density ratio → within-density "~25×" **[provisional → E2]**
- **Location:** §IV-C.
- **Current:** "avg_sent_len is ~28× weaker than kw_density by SNR" — mixes densities (kw_density SNR@1% 38.42 / avg_sent_len SNR@5% 1.38 = 27.8×, cross-density).
- **Proposed (structural fix — stable):** use a **within-density** comparison and state the density. e.g. "at 5% density, avg_sent_len is **~25× weaker** than kw_density by SNR."
- **Proposed (number — provisional):** on the current (pre-E2) corpus the within-5% ratio is **kw_density SNR 34.80 / avg_sent_len SNR 1.38 ≈ 25.3×** (provenance: `results/seva_v6_2_results_100k_p050_s042.json` + 5080 reproduction). **⚠ Recompute this exact ratio on the final E2 corpus** — SNRs change when the corpus changes.
- **Rationale:** the original ratio compared different densities; a within-density ratio is the honest fix.

## R-5 · W-002 — disclose `K_FETCH=20` over-fetch **[stable]**
- **Location:** §IV-B.
- **Current:** "retrieve K=5 nearest neighbours" — omits the over-fetch.
- **Proposed:** "We retrieve **K_FETCH=20** candidates via HNSW and rerank to the top **K=5** by exact inner product; the over-fetch mitigates HNSW approximation error."
- **Rationale:** undisclosed retrieval parameter.

## R-6 · W-007 — disclose `NORM_PERCENTILE=90` **[stable]**
- **Location:** §IV-B (normalization).
- **Current:** describes normalization but omits the percentile.
- **Proposed:** "Corpus-derived normalization constants (doc_length, sent_length, punct_density) use the **90th percentile** (NORM_PERCENTILE=90)."
- **Rationale:** undisclosed normalization parameter.

## R-7 · Lim 3 wording — calibration-dependent L3 floor **[stable framing; numbers provisional → E2/E5]**
- **Proposed:** "Layer-L3 exhibits an irreducible error floor (~17% at ≥5% density on CUDA). We find this floor is **calibration-dependent** — it falls to 0% under M4/MPS calibration — and therefore **a calibration choice, not an architectural limit** (see Limitation 4 and our ensemble-calibration analysis, E5)." **Do not present M4's 0% as "better"** — it is the same calibration knob.

## R-8 · Lim 5 wording — two-point scaling **[framing now; final wording → E7]**
- **Proposed (now):** state honestly that the O(1/√N) calibration law is "currently validated on two corpus sizes (10k, 100k)."
- **Final wording deferred to E7:** if the 500k point runs and lands on the fit → "validated on three sizes (10k/100k/500k)"; if E7 is skipped → "a third scaling point is left to future work." **Do not finalize until E7 resolves.**

## R-9 · Lim 6 wording — head-to-head framing **[framing now; final wording → E4-HH]**
- **Proposed (now):** remove any implication that Table IX is a *direct* comparison on shared data (it currently cites published values). Set the direction: "We reproduce **RAGDefender** (the closest LLM-free competitor; per-query, density-estimating) on our corpus for a direct head-to-head; **AV Filter and RobustRAG require decoder/LLM access incompatible with SEVA's LLM-free operating point and are cited for reference** under their own assumptions."
- **⚠ The "RAGDefender requires an LLM" framing is FALSE and must not appear** (RAGDefender is LLM-free; SecAI-Lab/RAGDefender, MIT).
- **Final wording deferred to E4-HH:** fill the actual regime-split result (and any honest flip-side / complementarity) after the experiment.

---

### Summary for the author
- **Apply now (stable):** R-1, R-2, R-3, R-5, R-6, R-7 (framing), R-9 (direction only, no false LLM claim).
- **Apply after E2 re-baseline:** R-4's exact ratio (use within-density framing now, finalize the number on the E2 corpus).
- **Apply after experiments:** R-8 (E7), R-9 final numbers (E4-HH).
- All number-bearing claims must trace to a results file produced on this machine (provenance invariant).
