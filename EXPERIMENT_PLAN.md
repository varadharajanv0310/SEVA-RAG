# SEVA-RAG — Experiment Plan for Q1 Acceptance (rev. 5 — final planning revision)

**Status:** DRAFT for author review (final planning rev; author will review → approve). Planning only — no experiments run, no code/data changed, no paper drafted.
**Author context:** solo undergraduate, no faculty co-author, single RTX 5080. Budget unit ≈ one 100k 3-seed multitier run = 30–90 min.
**Venue framing:** *Realistic target* = respected Q1 journal, **IJIS / JISA tier**. *Reach* = **TDSC / TIFS**. Labels: **[mandatory-realistic]**, **[mandatory-reach-only]**, **[optional]**.
**Section 4 decisions:** Q1–Q6 all settled; RAGDefender mechanism verified (see §0.5).
**Provenance invariant:** every comparative number in the revised paper traces to a results file produced on *this* machine.

---

## ★ CORE-IDENTITY INVARIANT (overrides everything — do not trade away for a better number)

SEVA's identity is a **fully LLM-free, lightweight, local, sub-30 ms-at-detection, weak-hardware-deployable (MacBook-Air-class, offline-capable) defense that also scales to industry GPUs.** Offline *attack* generation and reproducing an LLM-free *baseline* may use any tool. But **no** experiment or code/corpus/calibration change may: **(a)** make the **detector call an LLM/external API at detection time**; **(b)** add a **signal/dependency that breaks CPU-only / weak-hardware operation**; **(c)** **regress sub-30 ms detection latency.** If any change would require touching the detector in a way that risks (a)/(b)/(c), **STOP and flag.** The detector (`text_features`, `_score`, `_snr_weights`, `cluster_coh`, 10 signals) stays **byte-frozen** in every experiment. Watch-items: **E5** (ensemble τ = one offline constant), **E6** (encoder stays bge-class), **E7** (verify sub-30 ms at 500k). Reproducing **RAGDefender** does not breach the invariant (it is itself LLM-free, separate code path).

---

## 0. Context snapshot (self-contained)

**SEVA.** LLM-free RAG poisoning detector. Embeds docs with `BAAI/bge-large-en-v1.5` (1024-dim), CPU `faiss.IndexHNSWFlat`, 10 signals/doc → SNR-weighted "A-score," τ calibrated to `FPR_TARGET=0.0069`. Decisive signal **`cluster_coh`** = mean cosine to K=5 nearest **corpus** neighbours, **precomputed doc-level** (`seva_benchmark_4060.py:~325-360`). Templated poison clusters (≈0.99); diverse clean does not (≈0.73).

**Terminology.** "**Limitation N**" = paper Discussion. "**layer L1/L2/L3**" = detector tier.

**Verified baseline (RTX 5080, seeds 42 & 7; backup `seva_results_5080_baseline_backup_20260529/`).** `cluster_coh` gap 1/5/10% = +0.2569/+0.2587/+0.2654. Layer-L3 ASR: seed 42 → 5%=17.6%,10%=15.2%; seed 7 → 5%=13.6%,10%=20.8%. Grand-mean ASR ≈3.88–4.00%, DocFPR ≈0.77–0.85%. 5080≡4060 per-seed.

**Limitations (author-confirmed):** 1 white-box embedding attack; 2 domain-contrast confound; 3 L3 irreducible floor (calibration-dependent); 4 calibration sensitivity at L3 (≈37% spread); 5 two-point scaling; 6 no head-to-head; 7 single encoder.

---

## 0.5 — SEVA's CONFIRMED novelty vs the closest competitor (the positioning — read before E4-HH)

**Verified (RAGDefender, ACSAC 2025 full text; SecAI-Lab/RAGDefender, MIT; LLM-free):** (i) **per-query / post-retrieval** — it clusters and computes concentration *within the retrieved top-k set*, not corpus-level; (ii) **density-estimating** — its first stage estimates **Nadv** (number of adversarial passages in the retrieved set). RAGDefender is the architecturally closest system to SEVA, and it is the **opposite design on both axes.**

**SEVA's headline novelty (confirmed, and independent of out-performing on raw ASR):**
> **Unsupervised, density-agnostic, doc-level (corpus-anchored, retrieval-manipulation-immune) LLM-free poisoning detection with universal-FPR calibration.**

- **Doc-level vs per-query:** SEVA scores each doc against its *corpus* neighbours *before any query* → immune to query-time retrieval manipulation, and effective when poison is **sparse per query** (the realistic low-density regime). Per-query methods need a concentrated cluster inside the retrieved window.
- **Density-agnostic vs density-estimating:** SEVA calibrates one universal τ to a fixed FPR with **no contamination-rate input**; RAGDefender estimates Nadv.

**Explicitly NOT the headline:** the 10-signal SNR composite (additive; `cluster_coh` dominates, the extra 8 are low-SNR/evadable) and the L1/L2/L3 layering (an evaluation framework). **Lead with the doc-level/density-agnostic property; demonstrate it via E4-HH's low-density-regime comparison — not with "more signals."**

---

## Section 1 — Limitation triage

| Limitation | Class | Reviewer-impact reasoning | Experiment & label |
|---|---|---|---|
| **Lim 1 — White-box embedding attack** | (a) | Strongest attack on the core claim; Table VIII strawman → desk-reject-class. | **E1 + E1b — [mandatory-realistic]** |
| **Lim 2 — Domain-contrast confound** | (a) | Separation may measure domain, not poisoning → reject. | **E2 — [mandatory-realistic]** |
| **Lim 6 — No head-to-head (RAGDefender)** | (a) | RAGDefender = closest LLM-free competitor; head-to-head now *showcases* SEVA's confirmed doc-level/density-agnostic advantage in the low-density regime (§0.5). Highest-value experiment after E1. | **E4-HH — [mandatory-realistic]** |
| **Lim 6 — AV Filter / RobustRAG** | (b) reframe | Both need decoder/LLM access → same-operating-point head-to-head genuinely N/A; cite-for-reference. | (part of **E4-HH** write-up) |
| **Lim 4 — Calibration sensitivity at L3** | (a) | 37% spread; paper recommends ensemble but doesn't do it. Cheap. | **E5-CAL — [mandatory-realistic]** |
| **CF-008 / CF-007 / CF-009 — eval integrity** | (a) | CF-008 deflates FPR (reviewer distrusts all FPR); CF-007 thin adv set + caps E1 CIs; CF-009 false "3 seeds." | **E3 — [mandatory-realistic]** |
| **Lim 7 — Single encoder** | (a)/(c) | Encoder-conditional geometry; reach-only per Q6. | **E6-ENC (e5-large-v2) — [mandatory-reach-only]** |
| **Lim 5 — Two-point scaling** | (a) | Curve-fitting on 2 points; reach-only per Q6. | **E7-SCALE (500k) — [mandatory-reach-only]** (1M [optional]) |
| **Lim 3 — L3 irreducible floor** | (b)+(c), partial (a) via E5 | Calibration-dependent (0% on M4); disclose, don't cherry-pick. | rides on **E5-CAL** |
| **CF-006/005/001/004, W-002/007** | (b) | Text-only reframes (see R). | **R — [mandatory-realistic]** (free) |

---

## Section 2 — Experiment specifications

> Per experiment: **Method / Corpus & config / Confirms-if / Refutes-if / Compute / Failure branch / Invariant check.**

### E1 — Embedding-aware white-box attack + cost-of-attack curve (Lim 1) — [mandatory-realistic]

**Method (Q2).** Backbone = **algorithmic embedding-space optimization** (worst-case white-box): per poison doc, discrete-token optimization minimizing K-NN inter-poison cohesion s.t. a **retrievability constraint** (cos to target-query embedding ≥ floor); effort `b` = iterations (0/25/100/400). Cheap end = **offline-LLM diversification**; `b` = candidates/doc (1/4/16/64). New `whitebox_attack_seva.py`, subclassing the bench; detector byte-untouched.
**Statistical rigor.** **Wilson 95%** per point; n=250×3=750 → ±2.7 pp at p≈0.17. Frontier needs adjacent points ≳4–5 pp apart; increase **attempts** (CF-007) before seeds; seeds count only after CF-009. State CI-resolved vs within-noise.
**Corpus & config.** Final E2 corpus; 100k; 5%; seeds [42,7,123]; layer L1.
**Confirms/Refutes.** Descriptive curve; necessity tested in E1b. **Compute.** ~3–5 GPU-h.
**Failure branch.** Cheap config hits L1 ASR ≥~25% + clean-band cohesion → reframe to cost-imposition; cheap *and* costless → escalate.
**Invariant check.** Attack-side only. ✓

### E1b — Necessity vs correlation of retrievability ⊥ low-cohesion (Lim 1 hinge) — [mandatory-realistic]

**Method.** Jointly optimize poison to maximize min cos(doc, target-query q) **and** minimize mean pairwise cos; map the achievable Pareto frontier vs the geometric bound for the cap `{v:⟨v,q⟩≥r*}`.
**Confirms NECESSARY-if:** min achievable inter-poison cohesion at r* > clean band (~0.73). **Refutes (CORRELATED)-if:** cohesion ≤ clean band with cos(doc,q) ≥ r*.
**Compute.** <1 GPU-h. **Failure branch.** Only-correlated → "self-defeating" becomes setup-specific. **Invariant check.** Attack-side only. ✓ **Gates reach add-ons (Q6); feeds the E1×E4-HH interaction.**

### E2 — Domain-confound corpus + re-baseline (Lim 2) — [mandatory-realistic]

**Method.** Option A, **APPROVED A1 = Security Stack Exchange/SO Q&A** with a **mandatory clean-cohesion pre-check** (≤0.80 before full build; dedup SE near-duplicates). **Fallback A2 = arXiv cs.CR**. Regenerate poison to the same domain; re-run multitier.
**Corpus & config.** New clean+poison; 100k; 1/5/10%; seeds [42,7,123]. **Full re-embed.**
**Confirms-if.** Gap persists in-domain; L1 ASR≈0 at FPR≈target; `kw_density`/TTR separation survives → near-duplication, not topic. **Refutes-if.** Gap collapses → domain artifact.
**Compute.** 1–3 GPU-h. **Failure branch.** Narrow to "templated/near-duplicate; degrades when in-domain *and* diverse"; lean on E1b. **Invariant check.** Corpus data only. ✓

### E3 — Evaluation-integrity fixes (CF-008/007/009) — [mandatory-realistic]

**Method.** Query-construction edits (`~498-522`): independent held-out benign set; ≥50 unique adv templates; seed benign pool with `self.cal_seed`. Re-verify constants + frozen detector. **Folded into the E2 baseline run.**
**Confirms-if.** FPR ≈≤1–2%, ASR unchanged. **Refutes-if.** FPR jumps → coupling artifact. **Failure branch.** Honest FPR; transparent re-tune / ROC. **Invariant check.** Eval-side only. ✓

### E4-HH — Head-to-head vs RAGDefender, designed around the architectural regime split (Lim 6) — [mandatory-realistic] — DECISION Q4 (settled)

**Verified (see §0.5):** RAGDefender is **per-query/post-retrieval** and **density-estimating (Nadv)**, LLM-free, MIT. SEVA's **doc-level + density-agnostic** distinction is REAL — confirmed, not cosmetic. The "RAGDefender requires LLM" framing is FALSE and prohibited. AV Filter + RobustRAG keep the LLM-free-boundary softening (true for them).

**Design principle (the key move).** Compare **in the regime where the architecture bites.** RAGDefender's home turf = high adversarial-per-window ratios (k=5, 1×–6× — most of the retrieved window poisoned). SEVA's native regime = **low corpus density (1–10%)**, where a query's top-k is mostly clean (**0–1 poison docs**). RAGDefender's per-query concentration + Nadv estimation structurally needs a concentrated cluster in the window; when poison is **sparse-per-query**, there is no cluster (Nadv→0, misses) or it over-fires on legitimately-similar clean docs.

**Method.** Reproduce SecAI-Lab/RAGDefender; run BOTH on the **final E2 corpus + our poison**, **sweeping the per-query adversarial ratio** from RAGDefender's high-ratio home turf down to SEVA's sparse-per-query realistic regime; plot ASR/FPR for both; identify the crossover. **Headline = the low-per-query-density end (SEVA's native point).** Also run both **under the E1 white-box attack** (interaction below).

**Headline result (target, grounded in confirmed architecture so it holds even at ASR parity in RAGDefender's home regime):** *"The SOTA per-query defense assumes poison is concentrated in the retrieval window; under realistic low-density corpus poisoning that assumption fails (RAGDefender degrades), while SEVA's corpus-level doc-anchored detection holds."*

**Honest flip-side (REQUIRED — report it).** A regime plausibly exists where RAGDefender's per-query view **beats** SEVA: poison **diverse corpus-wide** (low doc-level cohesion → SEVA misses) but **concentrated for a target query** (high per-query concentration → RAGDefender catches). If found, report it — same architectural coin → **complementary coverage**, strengthens credibility.

**E1 × E4-HH interaction (key analysis).** A successful E1 attack on SEVA (corpus-diffuse + query-retrievable poison) lands *exactly* in RAGDefender's catch zone. So the honest framing of a successful white-box attack is **complementary / defense-in-depth** (SEVA owns low-density corpus poisoning; per-query owns query-targeted-diffuse poison), not "SEVA dominates." If E1b instead shows diverse+retrievable is unreachable (necessity), neither advantage materializes and SEVA's robustness story is clean. **This intersection is the single most informative experiment in the plan.**

**Corpus & config.** Final E2 corpus; per-query-ratio sweep + 1/5/10% density; seeds [42,7,123]; baseline poison + E1 attack.
**Confirms-if.** RAGDefender degrades at low per-query density while SEVA holds → headline confirmed (independent of raw-ASR parity). **Refutes-if.** RAGDefender holds at low per-query density too → architectural advantage doesn't translate empirically; fall back to the confirmed *conceptual* distinction + density-agnostic-calibration claim; report honestly.
**Compute & time-box.** LLM-free, light; GPU ≈ <1 h; integration **TIME-BOXED ~1–2 days**; fallback = honest non-reproduction ("RAGDefender cited, not reproduced") — never the false LLM framing.
**Invariant check.** RAGDefender LLM-free, separate code path; SEVA detector untouched. ✓

### E5-CAL — Ensemble calibration (Lim 4; partial Lim 3) — [mandatory-realistic]

**Method.** Ensemble τ (mean/median/bootstrap across seeds) per layer; re-evaluate Phase 4; report L3 ASR variance reduction (37%→<15% target); test floor lowering (Lim 3). Offline τ-injection.
**Corpus & config.** Reuses E2/E3 embeddings + `p3_v6.2_s*.json`; **Phase 4 only.** **Compute.** <1 GPU-h.
**Confirms-if.** Spread shrinks without FPR inflation. **Refutes-if.** No reduction → Lim 4 stands.
**Integrity note.** Never present M4's 0% floor as "better." **Invariant check (watch).** Ensemble τ = one offline scalar/layer applied once. ✓

### E6-ENC — Second encoder (Lim 7) — [mandatory-reach-only] (Q3: e5-large-v2)

Swap to `intfloat/e5-large-v2` (1024-dim, no `EMB_DIM` change); compare gap + L1 ASR/FPR. **Compute** 1–2 GPU-h. **Confirms-if** gap persists; **Refutes-if** encoder-specific. **Invariant check.** bge-class size → latency/weak-HW preserved; confirm CPU latency sub-30 ms. ✓

### E7-SCALE — Third scaling point (Lim 5) — [mandatory-reach-only] (1M [optional]; Q5)

Add **500k** (1M only if RAM/disk precheck passes); check τ/FPR follow O(1/√N). **Compute** 500k ≈ 2–4 GPU-h (~2 GB embeddings OK); **1M RAM/disk-pressured → check first.** **Invariant check.** Measure per-query latency at 500k; **confirm sub-30 ms** (positive scaling result); regress → STOP/flag. ✓

### R — Reframes (CF-001/004/005/006, W-002/007; Lim 3/5/6 wording) — [mandatory-realistic, ~0 GPU]

"argue" not "prove"; drop "hardware-agnostic" → 5080≡4060 + MPS caveat; 50 iters; within-density ~25×; disclose `K_FETCH=20`/`NORM_PERCENTILE=90`; Lim 3 calibration framing; Lim 6 wording = RAGDefender reproduced (regime-split result) + AV Filter/RobustRAG cited-for-reference; **lead novelty with §0.5, not the multi-signal composite.**

---

## Section 3 — Dependency graph & sequencing

**Upstream:** E3 fixes + E2 corpus → everything numeric after. **Depends on final E2 corpus:** E1, E1b, E4-HH, E5-CAL, E6-ENC, E7-SCALE. **Depends on E1 outputs:** E4-HH comparison (2) (under-attack) + the E1×E4-HH interaction. **Phase-4-only:** E5-CAL. **CPU-side / parallel:** E4-HH integration (1–2 day time-box), R.

**"One run or one-by-one?"** E2+E3 = one re-baseline run. E1 = a sweep. E4-HH = a per-query-ratio sweep (CPU-side). E5-CAL = Phase-4-only. E6/E7 = one re-embed each.

**Optimal order:**
1. **(no GPU)** All **R** + begin **E4-HH integration** (stand up RAGDefender; start the 1–2 day time-box early).
2. **(Step 1)** E3 fixes + A1 corpus; **clean-cohesion precheck** (gate ≤0.80, else A2); re-verify constants + frozen detector.
3. **(Step 2 — ONE run)** New baseline: multitier 100k, 3 seeds → Tables I/II/V/VI/VII. **Gate:** gap + FPR.
4. **(Step 3)** **E5-CAL** (Phase-4-only); **E4-HH ratio sweep** on baseline poison (CPU-side).
5. **(Step 4 — sweep)** **E1 + E1b**; then **E4-HH under the E1 attack** + the **E1×E4-HH interaction analysis** (the headline).
6. **★ DECISION GATE after E1b (Q6):** if **E1b = strong necessity AND faculty co-author secured** → reach add-ons; **else STOP at realistic, submit IJIS/JISA.**
7. **(Reach only)** **E6-ENC**, then **E7-SCALE** 500k (RAM-check first; 1M optional).

**Budget:** **Realistic GPU** ≈ **6–9 GPU-hours** + **~1–2 calendar-days CPU-side RAGDefender integration in parallel.** **Reach add-on** ≈ **+3–6 GPU-hours.**

---

## Section 4 — Decisions (all settled)

- **Q1 — APPROVED:** A1 (Security SE/SO Q&A) + ≤0.80 clean-cohesion precheck; fallback A2 (arXiv cs.CR); A3 (RFC/NIST/CWE) rejected.
- **Q2 — DECIDED:** E1 backbone = algorithmic embedding-space optimization; cheap end = offline-LLM diversification.
- **Q3 — DECIDED:** E6 encoder = `e5-large-v2`; reach-only.
- **Q4 — DECIDED + VERIFIED:** Reproduce RAGDefender (per-query, density-estimating, LLM-free, MIT) as a regime-split head-to-head; AV Filter/RobustRAG softened/cited; false LLM framing prohibited; time-box 1–2 days → honest non-reproduction fallback.
- **Q5 — DECIDED:** 500k third point; 1M optional contingent on RAM/disk precheck.
- **Q6 — DECIDED:** Provisional stop at [mandatory-realistic] → IJIS/JISA. Reach add-ons only if E1b = strong necessity AND faculty co-author secured. Revisable after E1b.

---

## Section 5 — Honest acceptance read (pre-conditions & residuals)

The plan front-loads the experiments most likely to come out against the paper (E1/E1b, E2) and now *showcases* a confirmed architectural novelty (E4-HH). Conditional estimate in Section 6.

**Residual weaknesses even if everything is favorable (disclose):**
1. **`cluster_coh` security is empirical, not proven** — rests on E1+E1b's low-diversity claim; favorable E1b confirms it geometrically *in our setup/encoder*, not as a theorem.
2. **SEVA's advantage is regime-specific, not universal** (the flip side of the confirmed novelty). Doc-level detection owns the low-density corpus regime; per-query methods own the query-targeted/corpus-diffuse regime. The honest framing is **complementarity / defense-in-depth**, which is credible but caps any "SEVA is THE answer" claim and invites a "you may need both" response. (This is a *bounded, confirmed* contribution — far healthier than rev-4's "is there novelty at all?" risk.)
3. **Doc-level detection assumes a static corpus + corpus-wide precompute** — more expensive than per-query and tied to the static-injection threat model.
4. **Single corpus family + (without E6) single encoder.**
5. **Layer-L3 floor is calibration-dependent.**
6. **Two-/three-point scaling.**

**Non-experimental levers (zero GPU):** faculty/senior co-author before submission (biggest leverage; Q6 ties the reach attempt to it); lead with the reproducibility asset (5080≡4060, committed per-seed results, provenance discipline).

---

## Section 6 — Acceptance probability on the favorable branch (re-estimated with novelty CONFIRMED)

**Assumption:** full **[mandatory-realistic]** set executes — E1+E1b, E2, E3, E4-HH (RAGDefender reproduced, regime-split), E5-CAL, R — **and favorable** (CI-resolved cost frontier; **E1b confirms necessity**; in-domain separation holds; honest FPR low; **RAGDefender degrades at low per-query density while SEVA holds**; ensemble calibration shrinks the L3 spread). Reach-only items not assumed.

**(a) Realistic — IJIS / JISA-tier: ~60–73%.**
Up from rev-4 (58–72%): the **novelty-exposure risk is retired** — SEVA's contribution is now a *confirmed* architectural difference (doc-level, density-agnostic) demonstrable in its native regime, so the head-to-head *showcases* rather than risks. On the favorable branch this is a complete, honest, well-differentiated paper for this tier. **Dominant swing factor reverts to the E1b necessity result** (principled vs setup-specific robustness — the core thesis). **Secondary swing:** whether the E4-HH ratio sweep empirically shows the low-density-regime win (high-confidence given the confirmed architecture, but the experiment must demonstrate it) and how the E1×E4-HH interaction resolves (clean SEVA robustness via E1b necessity, vs honest complementarity).

**(b) Reach — TDSC / TIFS, mandatory-realistic only favorable: ~33–48%.**
The confirmed novelty + a regime-split head-to-head against the SOTA LLM-free competitor helps materially at this tier. Still docked for the two open reach-only gaps (single encoder, two-point scaling) + author standing. Filling E6-ENC + E7-SCALE *and* adding a faculty co-author (Q6 gate) lifts toward ~48–60%.

**Single biggest residual weakness on the fully favorable branch (both venues): `cluster_coh`'s security is empirical, not proven** — even with SEVA owning the low-density regime against the closest competitor, the robustness argument is "effective poison must be low-diversity, and that's what we detect," confirmed geometrically *in our space/encoder/setup* (E1b), not as a theorem and not encoder-general. **Close second: the regime-specific bound (residual #2)** — SEVA is provantly stronger where per-query methods are blind, but the converse regime exists, so the honest ceiling is "best-in-class for low-density corpus poisoning," not "universally dominant." The paper should adopt both framings explicitly; with the §0.5 novelty confirmed and the RAGDefender regime-split in hand, that honesty reads as rigor, not concession.

---

*End of plan (rev. 5 — final planning revision). No code/data/experiments produced. Awaiting author review → approval. Do not commit until approved.*
