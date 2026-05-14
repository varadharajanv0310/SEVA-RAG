# PAPER_EDITS_LOG.md — authoritative to-do for the final manuscript revision pass

**Purpose.** The single durable, append-mostly record of **every** change to make to the SEVA manuscript, so the final revision can be applied in **one batch pass** without re-deriving anything across context resets. By author decision, all manuscript edits are batched to the **end** of the project; the paper `.tex` is intentionally **not** in this repo, and prose is **not** edited as we go — only logged here.

## How to use (final revision pass)
1. Apply every entry whose **Status = FINAL** to the manuscript, at the stated location, replacing OLD with NEW.
2. For **PROVISIONAL** entries, wait until the finalizing experiment has landed and the entry has been updated to FINAL (or SUPERSEDED).
3. Skip **SUPERSEDED** entries (kept for history only).
4. Every number-bearing change must match its cited **Provenance** results file (provenance invariant).

## Entry conventions
Each entry has: **ID · Added (date) · Origin · Location** (paper §/Table/Figure) · **OLD** (current text/number) · **NEW** (proposed) · **Provenance** (results file / stable code fact / doc) · **Status** · **Rationale/notes**.

**Status legend:** `FINAL` = apply as-is · `PROVISIONAL (→ X)` = number/wording finalized after experiment X; do not apply the specific number until updated · `SUPERSEDED (by Y)` = do not apply.

## Standing rule (rest of project)
Whenever any experiment (E1, E1b, E2, E3, E4-HH, E5, E6, E7) **changes, adds, or removes** a claim / number / table in the paper, **append a dated entry here** with exact location, OLD, NEW, provenance (results file), and FINAL-vs-PROVISIONAL — then **commit + push** this log. Append-mostly: never delete entries; mark superseded ones `SUPERSEDED`.

---

## Status index

| ID | Paper location | Status |
|---|---|---|
| R-1 | §I-C (C3); §V-E | FINAL |
| R-2 | Abstract; §I-C (C6); §IX | FINAL |
| R-3 | §IV-E | FINAL |
| R-4 | §IV-C | PROVISIONAL (→ E2) |
| R-5 | §IV-B | FINAL |
| R-6 | §IV-B | FINAL |
| R-7 | Limitation 3 | FINAL (framing); numbers PROVISIONAL (→ E2/E5) |
| R-8 | Limitation 5 | PROVISIONAL (→ E7) |
| R-9 | Limitation 6 / Table IX | FINAL (direction); result PROVISIONAL (→ E4-HH) |

---

## Entries — Group: Step-1 "R" reframes
*Origin: `PAPER_REFRAMES_R.md`, added 2026-05-30. Text-only fixes; detector byte-frozen.*

### R-1 · "We prove" → "we argue/show"
- **Added:** 2026-05-30 · **Origin:** CF-006
- **Location:** §I-C contribution C3; §V-E
- **OLD:** "We **prove** that cluster_coh is **invariant** under adversarial normalization." (§V-E gives an informal mechanistic argument — no theorem/proof.)
- **NEW:** C3 → "We **show** (empirically) and **argue** (mechanistically, §V-E) that cluster_coh is **robust to** adversarial normalization." In §V-E, add: "This is a **mechanistic argument, not a formal proof.**"
- **Provenance:** stable (claim-language fix; KNOWN_ISSUES CF-006)
- **Status:** FINAL
- **Notes:** Post-E1b upgrade path — if E1b returns "necessity," may strengthen to "we **show** that suppressing cluster_coh to the clean band requires sacrificing retrievability (§E1b)" (still not "prove"). Log that as a new entry if/when it lands.

### R-2 · Drop "hardware-agnostic"; substitute verified CUDA reproduction + honest MPS caveat
- **Added:** 2026-05-30 · **Origin:** CF-005
- **Location:** Abstract; §I-C contribution C6; §IX
- **OLD:** SEVA is "hardware-agnostic" (contradicted by §VI-F: layer-L3 ASR 0% on M4 vs 17.07% on RTX 4060).
- **NEW:** Remove "hardware-agnostic." Substitute: "SEVA's detection is **reproducible across CUDA GPUs**: we verify **bit-faithful per-seed reproduction on an RTX 5080 (Blackwell, sm_120) exactly matching the RTX 4060 baseline (seeds 42 & 7)**. Calibration is **backend-sensitive**: under Apple MPS, layer-L3 calibration differs (L3 ASR 0% vs 17% on CUDA, §VI-F), which we attribute to **calibration, not architecture** (Limitation 3)."
- **Provenance:** `seva_results_5080_baseline_backup_20260529/VERIFICATION_SUMMARY.md` (5080≡4060 per-seed, exact)
- **Status:** FINAL

### R-3 · Calibration iterations 100 → 50
- **Added:** 2026-05-30 · **Origin:** CF-001
- **Location:** §IV-E
- **OLD:** "100 binary-search iterations."
- **NEW:** "**50** binary-search iterations."
- **Provenance:** stable code fact — `seva_benchmark_4060.py:~767` does `range(50)`; the locked verified baseline used 50.
- **Status:** FINAL
- **Notes:** Do NOT re-run with 100 — it would invalidate the verified baseline for no scientific gain.

### R-4 · "~28×" cross-density ratio → within-density "~25×"
- **Added:** 2026-05-30 · **Origin:** CF-004
- **Location:** §IV-C
- **OLD:** "avg_sent_len is ~28× weaker than kw_density by SNR" (mixes densities: kw_density SNR@1% 38.42 / avg_sent_len SNR@5% 1.38 = 27.8×).
- **NEW (structural, stable):** "at 5% density, avg_sent_len is **~25× weaker** than kw_density by SNR."
- **NEW (number, provisional):** current pre-E2 within-5% ratio = kw_density SNR 34.80 / avg_sent_len SNR 1.38 ≈ **25.3×**.
- **Provenance:** `results/seva_v6_2_results_100k_p050_s042.json` (+ 5080 reproduction)
- **Status:** PROVISIONAL (→ E2) — apply the within-density *framing* now; **recompute the exact ratio on the final E2 corpus** (SNRs change with the corpus) and update this entry to FINAL.

### R-5 · Disclose `K_FETCH=20` over-fetch
- **Added:** 2026-05-30 · **Origin:** W-002
- **Location:** §IV-B
- **OLD:** "retrieve K=5 nearest neighbours" (omits over-fetch).
- **NEW:** "We retrieve **K_FETCH=20** candidates via HNSW and rerank to the top **K=5** by exact inner product; the over-fetch mitigates HNSW approximation error."
- **Provenance:** stable code fact — `K_FETCH=20` (`seva_benchmark_4060.py:65`)
- **Status:** FINAL

### R-6 · Disclose `NORM_PERCENTILE=90`
- **Added:** 2026-05-30 · **Origin:** W-007
- **Location:** §IV-B (normalization)
- **OLD:** normalization described, percentile omitted.
- **NEW:** "Corpus-derived normalization constants (doc_length, sent_length, punct_density) use the **90th percentile** (NORM_PERCENTILE=90)."
- **Provenance:** stable code fact — `NORM_PERCENTILE=90` (`seva_benchmark_4060.py:72`)
- **Status:** FINAL

### R-7 · Limitation 3 — calibration-dependent L3 floor
- **Added:** 2026-05-30 · **Origin:** Lim 3 framing
- **Location:** Limitation 3
- **OLD:** (floor presented as ~17%; M4 0% noted but framing implies architectural)
- **NEW:** "Layer-L3 exhibits an irreducible error floor (~17% at ≥5% density on CUDA). We find this floor is **calibration-dependent** — it falls to 0% under M4/MPS calibration — and therefore **a calibration choice, not an architectural limit** (see Limitation 4 and our ensemble-calibration analysis, E5)." **Do not present M4's 0% as "better."**
- **Provenance:** §VI-F + (numbers) E2/E5 once run
- **Status:** FINAL (framing); the specific floor numbers are PROVISIONAL (→ E2 re-baseline / E5 ensemble-calibration) — update when those land.

### R-8 · Limitation 5 — two-point scaling
- **Added:** 2026-05-30 · **Origin:** Lim 5 framing
- **Location:** Limitation 5
- **NEW (now):** state honestly "the O(1/√N) calibration law is currently validated on two corpus sizes (10k, 100k)."
- **Provenance:** E7 once run
- **Status:** PROVISIONAL (→ E7) — if 500k runs and fits → "validated on three sizes (10k/100k/500k)"; if E7 skipped → "a third point is left to future work." Do not finalize until E7 resolves.

### R-9 · Limitation 6 / Table IX — head-to-head framing
- **Added:** 2026-05-30 · **Origin:** Lim 6 framing
- **Location:** Limitation 6; Table IX
- **OLD:** Table IX implies a direct comparison but cites published values (no shared-corpus replication).
- **NEW (direction, now):** "We reproduce **RAGDefender** (the closest LLM-free competitor; per-query, density-estimating) on our corpus for a direct head-to-head; **AV Filter and RobustRAG require decoder/LLM access incompatible with SEVA's LLM-free operating point and are cited for reference** under their own assumptions."
- **⚠ HARD CONSTRAINT:** the "RAGDefender requires an LLM" framing is **FALSE and must not appear** (RAGDefender is LLM-free; SecAI-Lab/RAGDefender, MIT — verified, see `RAGDEFENDER_STANDUP.md`).
- **Provenance:** E4-HH once run
- **Status:** FINAL (direction + the AV-Filter/RobustRAG framing + the no-false-LLM constraint); the actual regime-split result/numbers + any complementarity finding are PROVISIONAL (→ E4-HH) — append when the experiment lands.

---

## Entries — Group: Experiment-driven edits

### E2-1 · In-domain clean corpus (domain-confound fix, Limitation 2)
- **Added:** 2026-05-30 · **Origin:** E2 / Limitation 2
- **Location:** dataset/corpus section (wherever clean = WikiText-103 is described); Limitation 2; method §III.
- **OLD:** clean corpus = WikiText-103 (general domain); poison = security-domain templates → domain-confounded (clean and poison differ in BOTH domain and templating).
- **NEW:** clean corpus = **Security Stack Exchange Q&A** (in-domain, diverse human-written; source `flax-sentence-embeddings/stackexchange_title_body_jsonl :: security.stackexchange.com`, CC-BY-SA). Poison stays security-templated → **domain controlled**; the only distinguishing feature of poison becomes templating/near-duplication. Add the clean-cohesion precheck as a validity control.
- **Precheck result (provenance: `precheck_cohesion.py`, 8000-doc sample):** clean cohesion **mean = 0.6973** (median 0.6974, p90 0.752; 0 near-duplicates) vs WikiText clean ~0.73 and poison ~0.99 → in-domain clean is at least as diverse as WikiText; confound-kill premise confirmed.
- **Provenance:** `precheck_cohesion.py` (sample). Final Table I/II/V/VI/VII numbers (incl. R-4 ratio) → Step-3 E2 baseline run.
- **Status:** PROVISIONAL (→ Step 3 baseline) — corpus methodology + diversity validated; the new baseline numbers come from the 100k 3-seed run.

### E2-2 / E3-1 · Step-3 in-domain baseline results (seed 42, 100k security SE corpus)
- **Added:** 2026-05-30 · **Origin:** E2 (Lim 2) + E3 (CF-007/008/009)
- **Provenance:** `seva_v6_2_results_100k_secqa_p{010,050,100}_s042.json` (repo root, gitignored; produced on this machine, run `step3_baseline_secqa_s42.log`, 47.9 min).
- **Status:** seed-42 only → PROVISIONAL (final tables need seeds 7,123). The *direction* of every finding below is clear at seed 42.

**(a) `cluster_coh` HOLDS in-domain (clean/poison/gap/SNR) — vs WikiText baseline:**
| density | in-domain | WikiText |
|---|---|---|
| 1% | 0.7518 / 0.9871 / **+0.2353** / SNR 5.99 | 0.7302/0.9871/+0.2569/4.72 |
| 5% | 0.7526 / 0.9909 / **+0.2384** / SNR 6.00 | 0.7322/0.9909/+0.2587/4.79 |
| 10% | 0.7466 / 0.9939 / **+0.2474** / SNR 5.78 | 0.7286/0.9939/+0.2654/4.66 |
→ Limitation-2 PRIMARY concern resolved: cluster_coh is domain-independent (gap holds, SNR *higher* in-domain). **Paper edit: report in-domain cluster_coh; the confound critique is answered for cluster_coh.**

**(b) Linguistic signals were PARTLY domain-confounded (SNR collapse, in-domain vs WikiText):**
| signal SNR | 1% | 5% | 10% |
|---|---|---|---|
| kw_density | **8.10** (was 38.42) | **6.52** (34.80) | **6.57** (32.85) |
| avg_sent_len | 0.53 (1.55) | 0.40 (1.38) | 0.37 (1.39) |
| ttr_signal | -1.12 (-1.77) | -1.10 (-1.90) | -1.09 (-1.70) |
| content_ttr | -1.09 (-1.96) | -1.05 (-2.18) | -1.08 (-1.92) |
→ confirms Limitation-2's premise *for the linguistic signals*: kw_density/TTR separation partly reflected security-vs-Wikipedia domain. **Paper edit: disclose that the linguistic-signal SNRs are domain-sensitive; kw_density SNR is ~8 in-domain, not ~38.**

**(c) ⚠ MAJOR FINDING — L2/L3 (adaptive-adversary) ASR explodes in-domain; L1 holds:**
| layer ASR | 1% | 5% | 10% |
|---|---|---|---|
| L1 (all signals) | 0.0% | 0.0% | 0.0% (WikiText 0/0/0) |
| L2 (evades kw_density) | **43.2%** | **60.9%** | **73.1%** (WikiText 0/1.6/0) |
| L3 (evades kw_density+avg_sent_len) | **44.0%** | **60.9%** | **73.1%** (WikiText 1.6/17.6/15.2) |
→ With ALL signals (L1) the in-domain detector still catches everything (ASR 0). But once the adaptive adversary evades kw_density, cluster_coh alone cannot hold the line in-domain → ASR 44–73%. The paper's adaptive-adversary robustness (Table V/VI L2/L3) was **substantially propped up by the domain-confounded kw_density signal.** **Paper edit (MAJOR, decision pending author): Table V/VI L2/L3 numbers change dramatically in-domain; the adaptive-robustness claim must be re-framed. This also strengthens the case that E1's white-box attack (targeting cluster_coh) is the decisive test.**

**(d) DocFPR (E3 independent benign queries, CF-008) — near/below target:** L1 0.45/0.52/0.40%, L2 0.45/0.46/0.31%, L3 0.48/0.46/0.31% (target 0.69%). FPR did NOT balloon with independent benign queries → the original FPR was not (heavily) propped by query-corpus coupling. **Paper edit: report honest in-domain FPR (still ≤0.7%).**

**(e) E3 methodology (paper Methods + eval):** benign queries = held-out security question titles (independent, CF-008); 50 unique adversarial templates (CF-007); benign pool seeded per cal_seed (CF-009). **Paper edit: describe the corrected eval protocol.**

### RESCOPE-1 · Narrow the adaptive-adversary claim; supersede Table V/VI L2/L3 with in-domain values
- **Added:** 2026-05-30 · **Origin:** E2 in-domain finding (see E2-2(c)) · **Status:** PROVISIONAL (→ E1/E1b + seeds 7,123)
- **Trigger:** in-domain L2/L3 ASR = 44/61/73% (vs WikiText ~0–17%); kw_density SNR collapsed 38→~7 → the composite's adaptive robustness was domain-confounded, carried by kw_density.
- **SUPERSEDES:** Table V & Table VI **L2/L3 ASR cells** — the WikiText L2/L3 numbers are not representative once the domain confound is removed. Report the **in-domain** L2/L3 ASR (keep L1=0% and FPR).
- **CLAIM NARROWING — every affected section to edit:**
  - **Abstract:** remove/soften "resists adaptive adversaries" / "evasion-resistant"; scope robustness to the L1 operating point + the `cluster_coh` geometric core.
  - **§I-C contributions:** narrow the adaptive-robustness contribution; lead with domain-independent doc-level `cluster_coh` + density-agnostic calibration (EXPERIMENT_PLAN §0.5).
  - **Table V / Table VI:** replace L2/L3 ASR with in-domain values (44/61/73% @1/5/10%); keep L1 ASR (0%) + DocFPR.
  - **§VI results (L2/L3 discussion):** rewrite to the in-domain reality.
  - **Limitations:** add — multi-signal adaptive robustness was partly domain-confounded; a keyword-dropping adversary largely defeats SEVA in-domain; `cluster_coh` alone is the load-bearing signal under evasion.
  - **Conclusion:** align.
- **Note:** FINAL framing depends on whether `cluster_coh` survives the E1 white-box suppression attack — **E1/E1b are now decisive, not confirmatory.** Do not finalize the rescoped claim until E1/E1b land; also pending seeds 7,123 for in-domain L2/L3 mean±std.

*(further entries appended per the Standing rule as E1 / E1b / E4-HH / E5 / E6 / E7 / additional seeds complete)*
