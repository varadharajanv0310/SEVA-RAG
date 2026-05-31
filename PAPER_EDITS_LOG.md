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

> **⚠ Log AS THE FINDING LANDS — never defer logging to an end-of-project pass.** Deferral is how findings get lost (this warning was added 2026-05-31 after several E1/E1b findings sat unlogged in commit messages/JSONs only). The LEDGER is always current; **only the manuscript *prose*/`.tex` edits are batched to the end** — the log itself is updated the moment a result changes a claim, even if the result is provisional or mid-investigation.

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
| E2-1 | clean corpus / Limitation 2 | FINAL (direction) |
| E2-2/E3-1 | §VI; Tables V/VI | FINAL (3-seed; see SEEDS-1) |
| RESCOPE-1 | Abstract; §I-C; Tables V/VI; Limitations; Conclusion | FINAL (in-domain 3-seed L2/L3) |
| E1B-1 | Abstract; §I-C; necessity; Conclusion | SETTLED-directional (geometry seed-invariant) |
| E1-1 | Abstract; §I-C; threat model; results; Limitations; Conclusion | **MAJOR** — FINAL (3-seed) |
| E1-2 | results; Limitations; Tables V/VI companion | **MAJOR** — FINAL (frozen L1 88.8%; 3-seed SEVA-ASR 86.7±1.0%) |
| E1-3 | method/system; any dedup claim | FLAG — verify vs paper |
| OPEN-CAL-1 | paper-wide (calibration disclosure) | RESOLVED (attack-specific/oracle; → E1-2 frozen 88.8%) |
| E3-2 | §IV methods; error bars; Limitations | DISCLOSURE (FINAL direction) |
| E-CAL-1 | §I-C; results; Limitations | POSITIVE — FINAL (3-seed; L1 0% all seeds) |
| E-CAL-2 | Tables V/VI L2/L3; §VI; Limitations | SETTLED s42 (adaptive collapse confirmed 3-seed via E2-2) |
| E4-HH | §I-C; Limitations; Table IX / head-to-head | CAUTIONARY (complementarity refuted) — FINAL (3-seed) |
| SEEDS-1 | 3-seed generalization (all above) | FINAL — calibration variance only (E3-2) |
| E1-4 | threat model; attack demo; Limitations | SETTLED s42 (8/8 answer-flips) — demo |

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

## Entries — Group: E1 / E1b (white-box attack on the rescoped `cluster_coh` core)

*Backfilled 2026-05-31. These findings existed only in commit messages + result JSONs until now — logged retroactively per the corrected as-you-go rule. Numbers are seed-42; cite the named commits/JSONs for exact values.*

### E1B-1 · Do NOT claim geometric/fundamental necessity for `cluster_coh` — empirical manifold-realizability only
- **Added** 2026-05-31 · **Origin** E1b geometric feasibility probe (commits `9b0b9b4`; framing-corrected `5aba517`) · **Status** SETTLED (directional); PROVISIONAL pending full E1b writeup
- **Finding:** pure-geometric necessity is unclaimable by inspection. In bge's 1024-d space the retrievability cone (cos ≥ r\* ≈ 0.72) permits poison at pairwise cohesion ≈ r\*² ≈ 0.52 ≪ τ_coh = 0.844; the synthetic probe reached ~0.59. "Retrievable ⟹ high inter-poison cohesion" is FALSE as geometry.
- **Manuscript impact:** remove any framing of `cluster_coh` robustness as *fundamental / geometric / necessary / provable*. If necessity is discussed, frame strictly as **empirical manifold-realizability** (can real encodable text reach the low-cohesion region?). That question is what E1's cheap-end answered → E1-1.
- **Affected:** Abstract; §I-C; necessity/E1b discussion; Conclusion.

### E1-1 · `cluster_coh` is bounded to TEMPLATED / payload-concentrated poison — cheaply evaded by low-prominence clone-inject
- **Added** 2026-05-31 · **Origin** E1 cheap-end clone-inject (`6566f43`) + prominence–cohesion frontier (`0bc043c`) · **Status** SETTLED (seed 42); PROVISIONAL pending 3-seed. **MAJOR.**
- **Finding:** real BGE clone-inject (clone a diverse on-topic Security-SE doc, inject the payload) inherits a clean-like neighborhood → `cluster_coh` ≈ 0.77–0.79 < τ_coh 0.844 at low prominence (1 rep) while retrievable (63% ≈ baseline 64%). It only fires when the payload is **concentrated** (≥~62%, coh > 0.84), where retrievability collapses. Crossover table in `0bc043c`.
- **Interpretation (load-bearing):** `cluster_coh` detects **templated near-duplication / payload concentration, NOT poisoning-in-general.** Scope the headline contribution to "domain-independent detection of **templated** corpus poisoning (the dominant literature pattern, e.g. PoisonedRAG-style near-duplicate injection)," explicitly **not** low-prominence single-clone injection.
- **Affected:** Abstract; §I-C contributions; threat model; results; Limitations; Conclusion.

### E1-2 · Composite ASR on clone-inject: under realistic FROZEN calibration SEVA leaks at EVERY layer (L1 88.8%)
- **Added** 2026-05-31 · **Updated** 2026-05-31 (STEP 1 — OPEN-CAL-1 resolved; `986140e`) · **Origin** composite frozen `phase3`+`phase4` (`b53acca`); frozen-gate calibration (`986140e`) · **Status** SETTLED (seed 42) / PROVISIONAL (→ 3-seed). **MAJOR.**
- **Finding (FROZEN templated-gate calibration — the REALISTIC deployed number):** L1 **88.8%**, L2 **100%**, L3 **100%**; `hash_catch = 0`; DocFPR ≤ 0.65%; τ frozen at the gate's templated value (τ_L1 = 0.5915). Provenance: `full_check_frozenincorpus_s042.json`.
- **Finding (recalibrated-on-attack — ORACLE, NOT realistic):** L1 **8.9%**, L2 **91.1%**, L3 **97.0%**; `hash_catch = 0`. Provenance: `full_check_incorpus_s042.json`.
- **Resolution (lifts the OPEN-CAL-1 block; confirms the prediction):** the 8.9% L1 "hold" was an **oracle artifact** — recalibrating τ+weights on the clone-inject poison lets τ drop into its score band. Under deployed/frozen calibration, `kw_density`'s separation (SNR 5.40) is **insufficient** to lift the composite over a τ tuned for templated poison (which fires `cluster_coh` AND `kw_density`) → **even naive L1 leaks clone-inject at 88.8%** (L2/L3 = 100%). `hash` never fires (per-doc tamper check, not near-dup — see E1-3).
- **Manuscript impact (MAJOR):** under realistic deployed calibration SEVA does **not** hold against clone-and-inject at any layer; the lone separating signal (`kw_density`) is domain-confounded and sub-threshold at the deployed τ. Use the **88.8%** (frozen) number, NOT 8.9%. Supersedes any "L1 ensemble catches clone-inject" framing.
- **Affected:** results; Limitations; Tables V/VI clone-inject companion row; adaptive-robustness discussion.

### E1-3 · FLAG (code-vs-paper): the "hash" catch is a per-doc `sha256` tamper check, NOT a near-dup / dedup detector
- **Added** 2026-05-31 · **Origin** `seva_benchmark_4060.py:930` + empirical `hash_catch = 0` (`b53acca`)
- **Finding:** `hash` compares a doc's text to its own stored sha256 — it cannot detect near-duplicate or cloned documents. SEVA has **no** near-dup/clone defense. (My earlier hash-near-dup hypothesis: tested and **refuted**.)
- **Manuscript impact:** if the paper describes hashing/dedup as a near-duplicate or clone defense, **correct it** — clone-inject is not caught by hashing.
- **Affected:** method/system description; any dedup claim.

### OPEN-CAL-1 · RESOLVED (paper-wide): SEVA is ATTACK-SPECIFIC (oracle) calibrated as implemented; realistic = frozen reference
- **Added** 2026-05-31 · **Resolved** 2026-05-31 (STEP 0 code audit + STEP 1 frozen test; `986140e`) · **Status** RESOLVED
- **Resolution (code-authoritative; paper PDF not auto-extractable in the frozen `seva` env):** as implemented, SEVA's SNR weights are derived from the *deployment corpus's* poison (`seva_benchmark_4060.py:685, 701, 744–750`) and τ is calibrated to FPR_TARGET on cal-clean *under those weights* (`:817–836`) → **attack-specific / oracle**. A reference/frozen protocol is supported by the Phase-3 cache (`:653–672`). Paper §0.5 "universal-FPR calibration" = τ density-agnostic (no poison-RATIO input), NOT attack-agnostic weights (SNR fundamentally needs poison to compute).
- **Consequence (paper-wide):** ALL per-corpus-recalibrated ASR numbers (incl. the gate's L1 0% and L2/L3 44–73%) carry an **oracle-calibration assumption that MUST be disclosed** — or be re-run under frozen reference calibration. The realistic frozen clone-inject number (L1 **88.8%**) is canonical and recorded in **E1-2**.
- **Affected:** §IV calibration disclosure; every ASR claim's calibration assumption; Limitations.

### NOTE-1 · clone-inject is itself a contribution (consider a "boundary of geometric detection" subsection)
- **Added** 2026-05-31 — the clone-inject evasion + the prominence/cohesion boundary + the templating-vs-poisoning distinction are a **novel result**. Consider presenting as a positive "boundary of geometric detection" subsection rather than only a limitation. Decide at framing time (after OPEN-CAL-1 + E4-HH).

### E3-2 · Methods disclosure: "N seeds" certifies CALIBRATION-sampling variance only (poison/corpus/embeddings/SNR-weights are seed-invariant)
- **Added** 2026-05-31 · **Origin** seed-semantics audit (`seva_benchmark_4060.py`: `SNR_SAMPLE_SEED=99` @681 fixed; benign sampling @560–564 and cal/eval split @325 use `cal_seed`)
- **Finding:** clean corpus, poison set, embeddings, `doc_coh`, AND SNR weights are all seed-**independent**; only benign-query sampling + cal/eval split vary with `cal_seed` (→ τ). So multi-seed mean±std bounds **calibration-sampling** variance, not poison/corpus-draw variance. The `cluster_coh` gap is fully seed-invariant (report without seed error bars).
- **Also:** the first seeds-7/123 background run was INVALID (per-tier cache not seed-tagged → reused seed-42's cached queries). Correct protocol: delete `p1_query.json` per seed before each run (embeddings reused).
- **Manuscript impact:** when reporting error bars / "N seeds," **disclose** they reflect calibration sampling only; for camera-ready consider one additional poison realization to bound draw variance.
- **Affected:** §IV methods; any error-bar / multi-seed statement; Limitations.

---

## Entries — Group: E-CAL (OPEN-CAL-1 re-runs — frozen / realistic calibration)

### E-CAL-1 · LINCHPIN: `cluster_coh` catches TEMPLATED poison under REALISTIC frozen (non-oracle) calibration
- **Added** 2026-05-31 · **Origin** OPEN-CAL-1 re-run #1 (frozen matched-templated SPLIT; `whitebox_attack_seva.py linchpin`) · **Status** SETTLED (seed 42) / PROVISIONAL (→ 3-seed). **POSITIVE — the surviving claim.**
- **Method:** calibrate SNR weights+τ on templated **half-A** (2500 docs, disjoint, interleaved), FREEZE, evaluate held-out templated **half-B** (2500) with **NO recalibration** on half-B. Provenance: `whitebox_attack_results/linchpin_s042.json`, `linch_{A,B}_s042.json`.
- **Result:** held-out templated `cluster_coh` = **0.9915**; STEP-A calibration cluster_coh SNR 5.95, τ_L1 0.5870 (L1 cluster_coh wt 0.154). Held-out **L1 ASR = 0.0%** at **DocFPR 0.58%** (≤ target 0.69%); L2 = L3 = **59.8%** (attempts = 112).
- **Finding:** the templated-detection claim is **NOT an oracle artifact** — weights+τ from a *disjoint* templated half generalize to held-out templated poison (both cluster at ≈0.99), so `cluster_coh` catches templated/near-duplicate poison at the **L1 operating point under realistic frozen calibration**. Contrast **E1-2**: clone-inject (cohesion 0.77) leaks L1 **88.8%** under the *same* frozen method → the boundary is clean (templated caught; subtle clone-inject evaded).
- **Manuscript impact:** the surviving positive contribution — domain-independent detection of **templated** corpus poisoning (the dominant literature pattern) — **holds under non-oracle calibration**; keep it, scoped, with the clone-inject boundary as the honest limitation (NOTE-1). The L2/L3 byproduct (59.8% frozen, pre-frozen adaptive weights) previews the adaptive collapse; the precise *no-re-adaptation* adaptive-frozen number is **E-CAL-2** (OPEN-CAL-1 re-run #2, pending).
- **Affected:** §I-C contributions (positive claim); results (frozen-calibration row); Limitations (boundary).

### E-CAL-2 · Adaptive L2/L3 under FROZEN L1 (strict no-re-adaptation): held-out templated leaks ~49–57%, comparable to oracle-adaptive
- **Added** 2026-05-31 · **Origin** OPEN-CAL-1 re-run #2 (`whitebox_attack_seva.py ecal2`) · **Status** SETTLED (seed 42) / PROVISIONAL (→ 3-seed).
- **Method:** freeze L1 weights+τ (templated half-A from E-CAL-1); evaluate held-out templated half-B; adversary neutralizes the evaded signal at the **feature level**, **no re-normalization**. Frozen τ_L1=0.5870; L1 weights kw_density 0.154, cluster_coh 0.154, avg_sent_len **0.000**. attempts=125. Provenance: `whitebox_attack_results/ecal2_s042.json`.
- **Sanity:** frozen L1, NO neutralization → ASR **0.0%** (templated caught at L1; matches E-CAL-1).
- **Result** (L2 = evade kw_density; L3 = also evade avg_sent_len):
  | variant | L2 | L3 |
  |---|---|---|
  | (b) pre-frozen adaptive weights | 53.6% | 53.6% |
  | **(c1) value-based clean-draw (strict no-re-adaptation; headline candidate)** | **48.8%** | **48.8%** |
  | (c2) drop-the-term (zero weight, no renorm) | 56.8% | 56.8% |
  L3 = L2 because the frozen L1 calibration assigned **avg_sent_len weight 0** → evading it has no effect under frozen L1.
- **Finding:** under realistic frozen single-detector calibration, a kw_density-evading adversary defeats SEVA on held-out templated at **~49–57%** — **comparable to** the oracle-recalibrated adaptive (gate 44–73%), **not dramatically worse**. The "doubly-damaging" effect is **muted here** because clean kw_density ≈ 0.047 → value-neutralization ≈ zeroing (hence c1 ≈ c2, c1 slightly lower as some clean draws carry minor keyword evidence). The adaptive-robustness collapse holds **regardless of calibration protocol** (consistent with RESCOPE-1): cluster_coh (0.99 for templated, wt 0.154) is not weighted enough to hold the composite once kw_density is evaded.
- **Manuscript impact:** report the adaptive L2/L3 collapse as **realistic (frozen: ~49% strict / ~54% pre-adapted / ~57% drop-term)**, not only oracle. Completes the frozen-calibration picture: **templated L1 0% (E-CAL-1) · templated adaptive ~49% (E-CAL-2) · clone-inject L1 88.8% / adaptive 100% (E1-2).**
- **Affected:** Tables V/VI L2/L3; §VI adaptive-adversary discussion; Limitations.

### E4-HH · SQUEEZE TEST (SEVA × RAGDefender, templated × clone-inject): complementarity FAILS — clone-inject is a shared blind spot
- **Added** 2026-05-31 · **Origin** E4-HH squeeze (`whitebox_attack_seva.py squeezegen` [seva env] + `e4hh_ragdefender.py` [ragdefender env]) · **Status** SETTLED (seed 42) / PROVISIONAL (→ 3-seed). **CAUTIONARY — keystone refuted.**
- **Setup:** SAME clone-inject artifact (provenance); SEVA = frozen gate-p050 L1 (τ_L1=0.5915); RAGDefender v0.2.0 **single_hop** (confirmed via `docs/algorithm.md`: clustering-based, NQ/MS-MARCO/k=5 regime — matches our short factual queries), **embedder = BAAI/bge-large-en-v1.5** (= SEVA's encoder → isolates the *algorithmic* difference, not embedder). Per-query top-K=5. Catch metric = retrieved-poison-NOT-removed (ASR, comparable to SEVA). Provenance: `whitebox_attack_results/squeeze_retrieval_s042.json`, `e4hh_s042.json`.
- **2×2 at RAGDefender's NATIVE defend() operating point — NOT a fair comparison (its forced ~50% FPR; see Fair-comparison below):**
  | poison | SEVA frozen-L1 | RAGDefender (native) |
  |---|---|---|
  | templated (near-dup) | **0.0%** (DocFPR≈0.6%) | 18.9% ASR @ 50.4% FPR |
  | clone-inject (n=5/query) | **87.7%** | 41.7% ASR @ 50.4% FPR |
  - **RAGDefender benign doc-FPR = 50.4%** — native `defend()` removes ~half of the *clean* retrieved docs on benign queries (its poison "catch" is largely indiscriminate removal, not detection).
- **Squeeze (clone-inject per-query count n; SURVIVE-BOTH = poison neither SEVA-flagged nor RAGDef-removed):**
  | n/query | attempts | SEVA-catch | RAGD-catch | **SURVIVE-BOTH ASR** | **q-level survive-both** |
  |---|---|---|---|---|---|
  | 1 | 82 | 11.0% | 70.7% | 25.6% | 33% |
  | 2 | 124 | 12.9% | 63.7% | 29.8% | 56% |
  | 3 | 145 | 12.4% | 57.2% | 37.2% | 71% |
  | 5 | 163 | 12.3% | 58.3% | 36.8% | 76% |
  | 8 | 175 | 12.6% | 57.1% | 37.1% | 80% |
- **Finding (keystone REFUTED):** at **every** per-query density, clone-inject survives **both** defenses on 26–37% of retrieved poison and on **33–80% of target queries** (rising with n). The attacker is **not squeezed**: clone-inject is corpus-diffuse (evades SEVA cohesion → ~88% ASR, flat in n) **and** evades RAGDefender's per-query clustering (the clones mimic clean docs; and as n rises the poison-majority flips RAGDefender's minority/majority vote → RAGD-catch *drops* 70.7%→57%). **Complementarity / defense-in-depth does NOT hold for realizable clone-inject.**
- **FAIR-COMPARISON checks (operating-point corrected; provenance `e4hh_fair_s042.json`):**
  - **CHECK 1 (Nadv-GATE):** single_hop has **no no-attack gate** — `estimate_n_adv` returns `min/max(cluster sizes) ∈ [1,|R|-1]`, never 0. On 30 CLEAN k=5 sets it removes **2.30/5 docs on avg (0/30 gated to 0)** → the 50% FPR is **structural forced cluster-removal**, not selective in-domain FP. We invoke it as the authors intend (defend() filters every set); single_hop has no non-clustering path. So the native ASRs above sit at a remove-~half point.
  - **CHECK 2 (MATCHED-FPR ROC):** defend() has no FPR knob → sweep RAGDefender's **concentration signal** (per-doc within-window mean pairwise cos — the premise both grouping strategies rest on; a **generous, best-case** RAGDefender with an idealized threshold it does not actually have). benign clean s_mean median 0.749; templated 0.993; **clone-inject 0.786 ≈ clean**. ROC (benign-FPR → templated-catch / clone-catch): **0.8% → 89.3% / 16.0%**; 5% → 93.4% / 28.8%; 25% → 95.9% / 58.9%; 50% → 97.5% / 77.9%. FPR floor ~0.20%.
- **FAIR head-to-head (matched ~0.7–0.8% FPR = SEVA's operating point):**
  | poison | SEVA catch | RAGDefender catch (best-case threshold) |
  |---|---|---|
  | templated | **100%** | ~89% |
  | clone-inject | 12% | ~16% |
  → On **templated** SEVA has a **modest, real edge** (100% vs ~89%: its corpus-level view catches lone templated docs RAGDefender's per-query view misses when no near-dup co-occurs in the window) — **NOT** "0% vs 18.9%@50%FPR domination" (that framing is **retired**). On **clone-inject both fail (~12–16%)** even at matched FPR → the shared-blind-spot / keystone-refuted finding is **robust to operating point**.
- **Manuscript impact (MAJOR):** complementarity reframe **not available**. Fair, defensible framing: SEVA **modestly out-detects** the SOTA per-query competitor on **templated** poison at a matched low FPR (and RAGDefender's native filter is **not deployable** on in-domain corpora — forced ~50% FPR), while **realizable clone-and-inject defeats both** (open problem / shared blind spot). Table IX: use the **matched-FPR** table (templated 100% vs ~89%; clone-inject both ~12–16%) — do **NOT** state "SEVA dominates 0% vs 18.9%".
- **Caveats:** seed 42; the matched-FPR ROC uses RAGDefender's concentration signal with an *idealized* threshold (generous — its real `defend()` cannot threshold and runs at ~50% FPR); RAGDefender embedder = bge-large (minilm sensitivity optional; clustering-evasion expected encoder-agnostic).
- **Affected:** §I-C; Limitations (shared blind spot / open problem); Table IX / head-to-head; the complementarity discussion.

### SEEDS-1 · 3-SEED generalization (seeds 42/7/123, CORRECTED per-seed calibration) — lifts PROVISIONALs to FINAL
- **Added** 2026-05-31 · **Origin** corrected per-seed multitier (E3-2 protocol: deleted the non-seed-tagged `p1_query.json` + stale `p3` per tier → each seed re-selects its benign pool; verified "50 targeted + 2000 benign queries" rebuild, NOT cache-load; `step3_secqa_s{7,123}_corrected.log`) + `whitebox_attack_seva.py seeds3` · **Status** FINAL (seeds 42/7/123).
- **Disclosure (E3-2):** seeds bound **CALIBRATION-sampling variance ONLY** (poison, clean, embeddings, `doc_coh`, SNR weights are seed-invariant; only benign sampling + cal/eval split vary → τ_L1). **τ_L1(p050) = 0.5915 / 0.5854 / 0.5877 → 0.5882 ± 0.0025.**
- **(a) In-domain BASELINE 3-seed (templated; finalizes E2-2(c) / RESCOPE-1):** L1 ASR **0.0% at all 3 seeds × all densities** (positive claim rock-solid). L2/L3 ASR mean±std: 1% ≈ L2 44±1 / L3 42±2; **5% = 57.0 ± 5.5** (s42/7/123 = 60.9/49.3/60.9); **10% = 72.4 ± 0.9**. DocFPR ≤ 0.83% (seed-7 slightly above the 0.69% target — calibration variance; disclose). The adaptive L2/L3 collapse (44–73%) holds across seeds.
- **(b) Frozen 2×2 + squeeze 3-seed (E4-HH / E1-2 / E-CAL-1):** templated SEVA-ASR **0.0 ± 0.0%** (L1 catches templated, all seeds); clone-inject SEVA-ASR **86.7 ± 1.0%** (n5; consistent with the 88.8% full-frozen). Squeeze SURVIVE-BOTH (clone-inject): n1 **25.6 ± 0.0%** / n3 **37.0 ± 0.3%** / n5 **36.4 ± 0.6%** / n8 **37.1 ± 0.0%**; q-level survive-both **33 → 80%** (±≤1%, rising with n). **RAGDefender catch + the matched-FPR fair-ROC are seed-INVARIANT** (no per-seed calibration; concentration = corpus property).
- **Verdict:** every E1 / E-CAL / E4-HH finding is **stable across 3 seeds** (tight std); the seed-42 numbers generalize. Provenance: `seeds3_s042.json`, corrected `seva_v6_2_results_100k_secqa_p{010,050,100}_s{007,123}.json`, `e4hh_ragd_flags_s042.json`.
- **Status flips → FINAL:** E2-2/E3-1, RESCOPE-1 (L2/L3 in-domain), E1-1, E1-2, E-CAL-1, E4-HH. E-CAL-2's specific frozen-no-re-adaptation variant stays seed-42 (the adaptive collapse it measures is confirmed across seeds via (a)).

### E1-4 · END-TO-END attack demo: a SURVIVING clone-inject FLIPS the RAG answer (8/8) — "retrieved-unflagged" → real corruption
- **Added** 2026-05-31 · **Origin** `e1_4_demo.py` (seva-env retrieval + **Ollama `gpt-oss:20b`** generator, offline, temp=0) · **Status** SETTLED (seed 42; demo).
- **Setup:** 8 target queries each with a clone-inject poison surviving **both** defenses (SEVA-unflagged AND RAGDefender-not-removed). For each, generate the RAG answer from (i) the top-K **with** the surviving poison vs (ii) the **clean-only** top-K; "flip" = false-payload markers ≥2 in the poisoned answer AND > clean+1. Generator is offline + a separate process → the detector stays LLM-free (core-identity invariant intact). Provenance: `whitebox_attack_results/e1_4_demo_s042.json` (full prompts + both answers + model id).
- **Result: 8/8 queries FLIPPED.** Clean-context answers are legitimate (Kerberos/NTLM/OAuth; Group-Policy Restricted Groups; sudoers — no payload). Poisoned-context answers assert the **false payload as authoritative fact**, e.g. *"the primary access token [is] provisioned by the infrastructure team at deployment and stored in a central registry, with administrator accounts holding elevated privileges across all subsystems … rotation every 90 days."* Payload-marker counts poisoned/clean: 4/0, 4/0, 4/1, 5/0, 5/1, 3/1, 5/1, 4/1.
- **Finding:** the clone-inject "retrieved-unflagged" result is a **real, working corpus-poisoning attack** — surviving poison reaches the generator and corrupts the answer on every tested query. Makes the E4-HH shared-blind-spot concrete: an attack that evades **both** SEVA and RAGDefender actually flips RAG outputs.
- **Manuscript impact:** establishes attack **effectiveness** (not just retrieval/ASR) — the demonstration that the clone-inject boundary is a genuine threat, not a metric artifact. (Demo: single generator/seed; larger query set / multiple generators = optional hardening.)
- **Affected:** threat model / attack section; Limitations; the clone-inject / boundary discussion.

*(further entries: E5 / E6 / E7 — appended per the Standing rule)*
