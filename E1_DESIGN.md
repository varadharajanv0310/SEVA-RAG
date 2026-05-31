# E1_DESIGN.md — Pre-registered design for E1 (cost-of-attack curve) + E1b (necessity)

*Pre-registration. Written 2026-05-30; **corrected 2026-05-31 per author approval** (binding-case fix,
band quantification, cluster_coh-only verdict, seed semantics). This document fixes the retrievability
floor, the attack-strength ladder, and the exact numeric NECESSARY-vs-CORRELATED rule **before** any
attack code is written or any result is seen. Committed to git before `whitebox_attack_seva.py` exists.
Anchor numbers trace to `seva_v6_2_results_100k_secqa_p050_s042.json` (the in-domain gate).*

---

## 0. Why this experiment, and why E1b is load-bearing

Step-3 (in-domain E2): `cluster_coh` confound-kill **CONFIRMED** (gap +0.235…+0.247, SNR↑), but the
multi-signal **adaptive**-robustness claim **collapsed** — dropping the domain-confounded `kw_density`
backstop (L2/L3) sends in-domain ASR to **44/61/73%** at 1/5/10%. The paper's surviving, defensible
claim is therefore **the geometric core alone**: an LLM-free, domain-independent, doc-level signal
(`cluster_coh`) detecting templated/near-duplicate poisoning. **E1/E1b now *decide* whether that core
survives a direct attack with no linguistic backstop.**

Sharpening datapoint: the in-domain gate poison (`poison_corpus_diverse.json`, 1,920 surface variants
per base doc across OAuth/JWT/API-key/session/service-account framings) **still has cluster_coh ≈
0.9909** (5%, s42). Surface/text diversification *already failed* to break doc-level cohesion. Open
question: can an adversary optimizing **in embedding space** succeed — and is success even
**geometrically possible** while the poison stays retrievable *and* attack-effective?

- **E1b (LOAD-BEARING — geometric necessity):** Is "retrievable + attack-effective" ⊥ "low cohesion" a
  **fundamental geometric incompatibility** (→ the core is *principled*, robust by necessity), or a
  **breakable correlation** in our corpus/encoder?
- **E1 (EMPIRICAL CONFIRMATION — realizable cost curve):** Attacker effort `b` vs achieved cohesion and
  ASR (Wilson 95% CIs per point), against the **strongest reasonable white-box adversary**, confirming
  or refuting E1b's geometric verdict with *real* poison documents.

---

## 1. Threat model, and the E1 vs E1b methodological split

White-box, worst-case **static corpus-injection** adversary: knows the encoder
(`BAAI/bge-large-en-v1.5`, frozen), the full clean corpus, the detector + every constant (K=5,
K_FETCH=20, FPR_TARGET=0.69%, SNR weighting, `_compute_doc_coh`), the calibration procedure, and the
detector's gradients. Goal: inject poison that is **retrieved** into the top-K for target queries and
**evades** the geometric core by suppressing `cluster_coh` — *without* any linguistic backstop (the
adversary is assumed to have neutralized `kw_density`/`avg_sent_len`, the L2/L3 regime the Step-3
finding made operative). Budget: ≤ the evaluated poison ratio (1/5/10%).

**The two experiments answer different questions and so use different attacker models — deliberately:**
- **E1b = geometric feasibility (UPPER BOUND).** Necessity is a *geometric* claim, so E1b grants the
  attacker the **best case**: poison vectors placed directly on the unit sphere (synthetic embeddings),
  unconstrained by text realizability. This is an **upper bound on any attacker's capability**. If even
  this best-case geometry cannot achieve low cohesion + attack-effectiveness, the result is **strong
  necessity** (any realizable text attack is weaker). This matches EXPERIMENT_PLAN §E1b ("map the
  achievable Pareto frontier vs the geometric bound for the cap").
- **E1 = realizable attack (LOWER BOUND / cost).** E1 produces **real poison documents** (cheap end:
  offline-LLM diversification; expensive end: discrete-token optimization) embedded by the frozen
  encoder — measuring what an attacker can *actually* realize and at what cost. We do **not** report E1
  cost-curve ASR from free-floating embeddings (that would be an unfair-to-the-defender strawman in the
  realizability direction).

If E1b says CORRELATED (best-case geometry permits evasion), E1 then tests whether *real text* can
realize it. If E1b says NECESSARY, E1 confirms the cost curve stays bounded.

---

## 2. Invariants & hard constraints (PROJECT_STATE §6)

1. **Detector byte-frozen.** `text_features`, `_score`, `_snr_weights`, `_compute_doc_coh`,
   `cluster_coh`, all constants — **never** modified. The detector scores whatever text/embeddings the
   attack provides, verbatim.
2. **`whitebox_attack_seva.py` subclasses `SEVABench`** (same pattern as `adaptive_attack_seva.py`:
   `load_seva_module(seed)` sets `sys.argv` to pass `--clean-corpus`/`--benign-queries`/`--corpus-tag`;
   `make_*_bench_class` overrides **only** `phase1` (inject attack poison), `phase2`
   (set `self.pe[:P]` to optimized poison embeddings / re-embed attack text + rebuild index), and
   `_ck`/`_shared_ck`/`rf` for isolation). It **calls** the frozen `_compute_doc_coh`/`phase3`/`phase4`/
   `_score`. No existing file is edited.
3. **Detector path stays LLM-free.** Offline-LLM use only in the E1 cheap end (candidate generation,
   attack-side). The detector never calls an LLM/API.
4. **CPU/weak-HW + sub-30 ms** not at risk (attack-side only); Phase-4 latency still reported.
5. **Honesty clause (author-mandated).** If E1b returns **CORRELATED**, it is reported as a
   **limitation**, plainly, in `PAPER_EDITS_LOG.md` and the paper — never softened or buried. The
   honest reframe is **complementarity / defense-in-depth** (§10).
6. **Provenance.** Every number traces to a file produced on this machine. New artifacts under
   `whitebox_attack_results/` (gitignored; backed up).

---

## 3. The implementation seam (so "detector frozen" is verifiable)

`seva_benchmark_4060.py` reference points (read-only — hooked by the subclass, never edited):
- **Poison injection:** `phase1` replaces `self.corpus[0:P]["text"]` + stores `self.hashes` (`:498-503`).
- **Poison embedding:** `phase2` loads the **shared clean embedding cache** (`_shared_ck("p2_pe.npy")`),
  re-embeds the P poison texts, writes `self.pe[:self.P] = pe_poison`, builds `IndexHNSWFlat`
  (`:596-623`). The subclass override either re-embeds attack *text* (E1 cheap) or writes optimized
  poison *embeddings* directly into `self.pe[:P]` (E1 expensive / E1b synthetic), then rebuilds the
  index. **Reusing the shared clean cache keeps clean embeddings byte-identical to the gate.**
- **The detected signal:** `_compute_doc_coh` (`:334-389`) computes, for each doc i, the **mean pairwise
  cosine among i's K=5 nearest neighbors in the full corpus index** (clean+poison), cached to
  `p2_doc_coh.npy`. **Frozen — called, never altered.** This is the quantity the attack must suppress
  and the quantity the decision rule is stated in.
- **Detection & ASR:** `phase4` (`:903-995`) retrieves K_FETCH=20 → exact-rerank top K=5; a poison doc
  in top-K is an **attempt**; if its layer score ≤ τ it is **unflagged → success**. Benchmark ASR =
  successes / attempts over the 50 targeted queries.

---

## 4. PRE-REGISTERED (a): the retrievability floor r*

For each targeted query q, let `d5(q)` = the cosine of the **5th-most-similar clean corpus doc** to q
(the top-K=5 displacement threshold). A poison doc is **retrievable for q** iff
> **r\*(q) = max( d5(q), 0.60 ).**

**Justification (steelman floor, not a strawman defender-win):** `d5(q)` is the *literal* threshold to
enter the retrieved top-K — below it the poison is never retrieved, so calling it "evaded" is vacuous.
`0.60` is the retrievability the **already-effective baseline poison achieves**
(`generate_poison_corpus.py` targets query cosine ≥ 0.60; the gate poison *is* retrieved, atk_att=138 at
5%/s42). So the attacker must stay **at least as retrievable as poison that already works** — neither
hobbled nor handed a free pass. **Locked value:** `r*(q)` is computed deterministically from the frozen
clean corpus + encoder in the probe (Step A) and frozen before any sweep; the operating scalar
`r*_op = median_q d5(q)` (expected ≈ 0.60–0.72 for bge) feeds the decision; the frontier additionally
sweeps `r* ∈ {r*_op, 0.65, 0.70, 0.75}`.

---

## 5. PRE-REGISTERED (b): the attack-strength ladder `b` (E1)

Both ends emit **real poison text** scored by the frozen detector.

| `b` | End | Method |
|---|---|---|
| **0** | baseline anchor | Gate poison (`poison_corpus_diverse.json`). cluster_coh≈0.99; ASR = gate. |
| 1 / 4 / 16 / 64 | **cheap (offline-LLM)** | Per doc, generate `b` candidate rephrasings with an **offline** LLM (attack-side), keep the one minimizing `cluster_coh` s.t. `cos(doc,q) ≥ r*`. Strengthens the existing `adaptive_attack_seva.py` topic-frame/token-dropout heuristic via LLM generation + embedding-space selection. |
| 25 / 100 / 400 | **expensive (discrete-token, white-box)** | Per doc, **discrete-token optimization** (HotFlip / greedy-coordinate-gradient over the frozen encoder) for `b` iterations, **objective = minimize the doc's `cluster_coh`** s.t. `cos(doc,q) ≥ r*`. Full encoder gradients; attacks the exact detection signal. |

**Steelman statement (pre-registered).** *`b=400` discrete-token is the **strongest reasonable
white-box adversary**: full white-box access to the frozen encoder/detector, optimizing the **exact**
detection signal as its objective, constrained **only** by retrievability, with a large iteration
budget. If `cluster_coh` survives it, the robustness is **earned**; if it falls, we report the fall
honestly.* We publish the full b-ladder, achieved cohesion at each rung, and the crossing point (if
any) — **no silent caps**.

---

## 6. PRE-REGISTERED (c): operating points — what the verdict binds on *(APPROVED)*

- **The necessity VERDICT binds on `cluster_coh`-ONLY** (weights = {cluster_coh: 1.0}; an input-side
  weights dict — detector code untouched). This is the geometric thesis in isolation: one signal,
  one threshold (`τ_coh`).
- **L1 / L2 / L3 ASR are reported as context, not part of the rule.** **L3** (no kw_density, no
  avg_sent_len) is the **operational corollary** — the closest *multi-signal* regime to the bare
  geometric core — and we discuss it as such, but it does **not** enter the necessity verdict (binding
  necessity on L3 would conflate **one-signal necessity** with **six-signal robustness**). **L1 is
  never bound on** (it holds only via the domain-confounded `kw_density`, per Step-3).

---

## 7. PRE-REGISTERED (d): the EXACT NUMERIC E1b decision rule *(APPROVED, binding-case corrected)*

**Binding metric — cohesion of an *attack-effective* poison set (the n=1 degeneracy is excluded):**
> **C_min = the lowest mean inter-poison cohesion achievable by a poison set that STILL REACHES TARGET
> ASR at the retrievability floor.**

Operationally, `C_min = min` over poison configurations satisfying **both** (i) the retrievability floor
`cos(p_i,q) ≥ r*(q)` and (ii) the **target-ASR gate** (below), of the **mean `cluster_coh` (frozen
`_compute_doc_coh`) over the poison docs**. In the attack-effective regime the poison docs' nearest
corpus neighbors are predominantly other poison, so this mean equals the **mean inter-poison cohesion**.
The poison set size / per-query count `n` is a **free search variable** (≤ the density budget), **not a
fixed parameter** — the optimizer may use any `n` it likes; the target-ASR gate is what makes the metric
meaningful.

**Why this fixes the binding case.** A lone poison doc (`n=1`) has cluster_coh in the clean band *by
construction* (its 5 nearest neighbors are clean), so a naive "min cohesion" rule would trivially return
CORRELATED for an attack **too weak to corrupt anything**. The **target-ASR gate** rejects such
degenerate weak-but-evasive configurations: only poison sets that **actually work as attacks** can count
as a refutation. We therefore **instrument (cohesion, retrievability, ASR) JOINTLY at every attack
point** — a point counts toward `C_min` only if it is simultaneously low-cohesion **and** retrievable
**and** attack-effective.

**Target-ASR gate (pre-registered).** A configuration "reaches target ASR" iff — against the
`cluster_coh`-only detector at the retrievability floor — it places ≥1 **unflagged** poison doc in the
top-K for **≥ 0.8 · R_base** of the 50 targeted queries, where **R_base** = the fraction of targeted
queries whose top-K contains ≥1 poison doc under the **baseline (b=0) attack with detection OFF** (the
attack's intrinsic effectiveness ceiling; measured in the probe, expected ≈ 100% at 5%). The `0.8`
factor requires the evasive attack to **retain ≥80% of baseline retrieval reach** — i.e., be diverse
*without* surrendering the reach needed to corrupt answers. (We also report the benchmark's doc-level
ASR for continuity.)

**Band (FPR-aligned, the detector's own threshold).** `τ_coh(s)` = the `cluster_coh`-only detection
threshold produced by the **frozen** Phase-3 `calibrate()` routine at FPR_TARGET = 0.69% on seed s's
clean calibration set (≈ the 99.31st percentile of clean `cluster_coh`). A poison doc evades the
geometric core **iff** its `cluster_coh ≤ τ_coh`.

**Decision (stated now, before any result):**
> **NECESSARY** iff **C_min > max_s τ_coh(s)** over seeds s ∈ {42, 7, 123} — the *worst-case
> calibration threshold*. (We use `max_s τ_coh` rather than per-seed because **C_min is
> seed-independent geometry** while **τ_coh is the only seed-varying quantity** — see §7a.)
>
> **CORRELATED** iff **C_min ≤ max_s τ_coh(s)** — an attack-effective, retrievable poison set exists
> whose cohesion reaches the detection band → the geometric separation is a breakable correlation.
>
> **NECESSARY (strong)** as a special case: if **no** configuration passes the target-ASR gate at the
> floor at all, the retrievability constraint *alone* defeats the attack — report distinctly.

**Reported alongside (context, not a second rule):** the **margin** `C_min − max_s τ_coh`; the same
comparison against the **clean 95th percentile** (a lower, more defender-favorable reference — your
original example); and the full joint (cohesion, retrievability, ASR) frontier. If `C_min` lands in
`(P95, max_s τ_coh]` the verdict is **borderline / setup-specific** and reported as such — no overclaim.

**Anchors (so the bar is legible now):** gate clean `cluster_coh` = 0.7526 ± 0.0398, poison = 0.9909
(5%/s42) ⇒ `τ_coh` ≈ 0.85–0.88 (computed exactly in the probe). The attack must drive an
*attack-effective* poison set's cohesion from ≈0.99 **below ≈0.86** while holding retrievability — E1b
asks if that is geometrically reachable; E1 measures the cost if so.

---

## 7a. Seed semantics — what the 3-seed rule actually bounds *(per author; confirmed in code)*

The three seeds {42, 7, 123} vary **only** the benign-query sampling and the cal/eval split — **not**
the poison, clean corpus, or embeddings. Confirmed against `seva_benchmark_4060.py`:

| Element | Seed-varied? | Code |
|---|---|---|
| Clean corpus (first N of A1) | **No** (deterministic) | `phase1:422-431` |
| Poison set (`poison_corpus_diverse.json`, injected `corpus[0:P]`) | **No** (deterministic) | `phase1:497-503` |
| Embeddings / FAISS index / `doc_coh` | **No** (deterministic) | `phase2`, `_compute_doc_coh` |
| SNR-derived weights | **No** — Phase-3 SNR sample uses fixed `SNR_SAMPLE_SEED=99` | `:83`, `:681` |
| Targeted queries (the 50) | **No** | `phase1:558-559` |
| Benign-query selection (2000 of pool) | **Yes** — `rng(cal_seed)` | `phase1:560-564` |
| Cal/eval split (60/40) | **Yes** — `rng(cal_seed)` | `_split_queries:325` |

**Consequence:** the 3-seed rule bounds **calibration variance only** (it moves `τ`, FPR, and
ASR-via-`τ`). The attack geometry — and therefore **`C_min` — is seed-invariant**, so it is computed
**once** and compared to the **worst-case `max_s τ_coh(s)`**. (This is exactly why the rule is phrased
that way.)

**Practical note (off the probe's critical path).** The background seeds-7/123 runs reused seed-42's
cached `p1_query.json` (the per-tier cache dir is not seed-tagged), so they are **calibration-split-only
variants, not independent seeds** — not valid for τ_coh(7)/τ_coh(123). The **probe uses τ_coh(42)** for
its directional answer. The **final E1b verdict's `max_s τ_coh(s)`** needs valid τ_coh for all three
seeds, obtained from corrected per-seed calibration runs (delete each tier's `p1_query.json` between
seeds → re-select benign sample with `rng(seed)`; embeddings/`doc_coh` are reused → ~1 min/tier). Per
author: **not run now** (off critical path); scheduled at E1b-finalization.

---

## 8. PRE-REGISTERED (e): E1 cost-curve metrics + Wilson CIs

Per ladder rung `b` × layer ∈ {cluster_coh-only (binding), L1, L2, L3 (context)}, 5% density, seeds
{42,7,123}:
1. **Achieved `cluster_coh`** of the attack poison: mean + **fraction below τ_coh**.
2. **ASR** with **Wilson 95% CI**: `n` = poison-in-top-K attempts (CF-007: boost attempts via all 50
   targeted queries + extra unique adversarial templates so adjacent `b` points are ≳4–5 pp apart;
   **state** CI-resolved vs within-noise per pair).
3. **Retrievability check:** fraction of attack poison with `cos(doc,q) ≥ r*` (flags any point that
   "won" by dropping retrievability — invalid).
4. **Latency** (Phase-4 mean/p95) to confirm sub-30 ms untouched.

---

## 9. The coarse single-seed E1b feasibility probe — RUN FIRST, then STOP at the gate

Purpose: a **directional** NECESSARY-vs-CORRELATED answer in **< 30 min GPU, seed 42 only**, reusing the
gate's cached embeddings (no re-embed), with **(cohesion, retrievability, ASR) instrumented jointly**.
Best-case **geometric** attacker (synthetic poison embeddings = upper bound, §1).

Steps in `whitebox_attack_seva.py` (probe mode):
- **A. Lock constants from frozen artifacts.** Reuse the gate's cached clean embeddings + build a clean
  HNSW index; encode the 50 targeted queries (frozen bge). Compute `d5(q)` → `r*_op = median_q d5(q)`.
  Compute the clean `cluster_coh` distribution and **`τ_coh(42)`** via the frozen FPR-0.69%
  `calibrate()` on cluster_coh-only. Compute **`R_base`** (baseline poison retrieval reach, detection
  off).
- **B. Best-case geometric sweep, JOINT triple per point.** Grid over (angular spread σ, per-query count
  `n`): synthesize retrievable poison embeddings in each query's cap and inject into the index;
  for each config record **(mean cluster_coh, retrievability = fraction cos≥r* / retrieval reach, ASR =
  cluster_coh-only per-query success % [+ L1/L2/L3 doc-ASR context])**. This traces the achievable
  region and exposes the tension: spreading poison to lower cohesion vs keeping enough reach to pass the
  target-ASR gate.
- **C. C_min^probe + directional verdict.** `C_min^probe` = lowest mean cohesion among configs that pass
  the target-ASR gate (≥0.8·R_base) at `r*_op`. Compare to `τ_coh(42)`:
  - `C_min^probe` comfortably **above** τ_coh(42) → **necessity-leaning** → proceed to full E1b + E1.
  - `C_min^probe` **at/below** τ_coh(42) → **correlated-leaning** → proceed (E1's cost curve now decides
    cheap-vs-expensive) and pre-warn the author the honest-limitation/complementarity branch (§10) is
    likely.
  - **STOP and report** `(cohesion, retrievability, ASR, directional NECESSARY/CORRELATED)` to the
    author before the full sweep. (Probe `τ_coh(42)` is directional; the binding verdict later uses
    `max_s τ_coh(s)` from corrected 3-seed calibration, §7a.)

---

## 10. Full sequence, decision branches, E1×E4-HH link

**Sequence:** probe (§9, GATE) → **E1b full** (geometric frontier, 3-seed τ_coh, NECESSARY/CORRELATED)
→ **E1 cheap** (LLM ladder) → **E1 expensive** (discrete-token ladder) → cost-curve figure (Wilson CIs)
→ E4-HH interaction.

**Branches (pre-registered):**
- **NECESSARY + E1 confirms** (even b=400 can't push an attack-effective set below τ_coh): robustness
  **earned and principled** — strongest outcome; upgrade R-1 wording (still "show/argue," not "prove").
- **CORRELATED but expensive** (crosses τ_coh only at high b, or only by sacrificing
  retrievability/relevance): reframe to **cost-imposition** ("evasion costs N× and degrades the attack").
- **CORRELATED and cheap** (LLM end crosses τ_coh with retained reach): report **honestly as the central
  limitation**; reframe to **complementarity / defense-in-depth** — such poison is corpus-diffuse +
  query-retrievable = exactly RAGDefender's per-query catch zone (the E1×E4-HH interaction, "the single
  most informative experiment").

---

## 11. Compute, outputs, provenance

- **Compute:** probe < 0.5 GPU-h; E1b full ≈ 1 GPU-h; E1 cheap ≈ 1–2 GPU-h; E1 expensive ≈ 3–5 GPU-h.
  Reuses cached clean embeddings (no 100k re-embed). Total ≈ 5–8 GPU-h.
- **Outputs:** `whitebox_attack_results/` — `probe_s042.json` (r*_op, τ_coh(42), R_base, joint frontier,
  C_min^probe, directional verdict), `e1b_frontier_s*.json`, `e1_costcurve_s*.json`, `summary.md`.
  Gitignored; backed up.
- **Paper:** append E1/E1b entries to `PAPER_EDITS_LOG.md` (PROVISIONAL→FINAL); finalize RESCOPE-1 / R-1
  per the branch that obtains. No manuscript prose edited.

---

## 12. Locked decisions (approved 2026-05-31) & immediate next actions

**Locked (author-approved):**
1. Verdict binds on **`cluster_coh`-only**; report L1/L2/L3; **L3 = operational corollary, not in the
   rule; never L1**.
2. Band = **`τ_coh` (FPR-aligned)**; rule = **NECESSARY iff `C_min > max_s τ_coh(s)`** over {42,7,123};
   report P95 as context.
3. **`C_min` = lowest mean inter-poison cohesion of a poison set that still reaches target ASR
   (≥0.8·R_base) at the retrievability floor** — `n` is a free search variable; **n=1 degeneracy
   excluded** via the target-ASR gate; **(cohesion, retrievability, ASR) instrumented jointly**.
4. Retrievability floor `r*(q)=max(d5(q),0.60)`; b-ladder 0 / {1,4,16,64} / {25,100,400} with b=400 the
   pre-registered steelman.
5. Seeds bound **calibration variance only**; `C_min` is seed-invariant geometry; seeds 7/123 **not
   re-run now** (off critical path).

**Next actions (this session):** commit this file → build `whitebox_attack_seva.py` (probe mode) →
run the <30-min seed-42 probe → **STOP at the probe gate** and report
`(cohesion, retrievability, ASR, directional NECESSARY/CORRELATED)`.

---

## 13. PROBE RESULT (seed 42, 2026-05-31) + CORRECTED interpretation *(supersedes the §9 directional wording)*

**Numbers (`whitebox_attack_results/probe_s042.json`):** τ_coh(42)=**0.8442** (clean P95 0.8130);
r*_op=**0.7231** (d5 median 0.718); R_base=**64%**. Best-case *synthetic free-on-sphere* spread poison:
**100% retrievable, 100% ≤ τ_coh, mean cohesion 0.59–0.70, ASR-loose 100%, ASR-majority ≤94%** →
C_min(geometric)=**0.5908**. Validity (`whitebox_validity.py`): combined real-corpus 5-NN cohesion
**0.5866** ≈ probe 0.5908; poison-only shortcut 0.5103 → the probe used the COMBINED REAL corpus path;
the +0.076 lift over the pure cone floor is the clean-neighbour contribution.

**CORRECTED INTERPRETATION (replaces the earlier "necessity refuted / minus for the paper").** The
CORRELATED result was **foregone**: pure-geometric necessity holds iff **r\*² > τ_coh**, and
**0.72² = 0.52 ≪ 0.84**, so it fails by inspection for any r* < √0.84 ≈ 0.917. The probe merely
**rediscovered the r\*² cone floor**; free-on-sphere placement **ignores the encoder manifold** and so
**did NOT test the real claim**.

> **Reworded verdict: pure-geometric necessity is UNCLAIMABLE (r\*² ≪ τ_coh); manifold realizability is
> the open, decisive question. This is NOT evidence SEVA is weak.**

The only realizable poison we have (the 0.99 set) is *surface-diverse but semantically identical* — a
**weak predictor either way**. The decisive test is whether **realizable text** can be pushed below
τ_coh — that is E1 (below).

## 14. E1 cheap-end PRE-REGISTRATION *(fixed BEFORE any cheap-end numbers)*

**Attack objective — SEMANTIC diffusion.** Generate poison that is **genuinely semantically diverse**
subject to **(a) retrievable** (cos(doc,q) ≥ r*(q)) **and (b) payload preserved** (the malicious claim
survives). Surface-only variation (the 0.99 set) is explicitly *not* the attack.

**Realizability target (crisp):** push **real, retrievable, on-target** poison from cohesion **0.99
below τ_coh = 0.84** — it need NOT reach the 0.59 geometric floor.

**Cheap-end ladder (no GPU optimization):** offline-LLM semantic generation; budget `b` =
candidates/slot ∈ {1, 4, 16, 64} with cohesion-based selection (keep the lowest-cohesion candidate that
stays retrievable + payload-preserving).

**Joint metric per budget point (reported together):** `(mean cluster_coh, retrievability,
ASR-majority)` where **ASR-majority** = ≥⌈K/2⌉=3 of the top-K are *unflagged* poison (cluster_coh-only @
τ_coh) on **≥ target = 0.8·R_base** of the 50 targeted queries.

**3-WAY DECISION RULE (pre-registered):**
- **cheaply-evadable:** cheap LLM rephrasing (small budget) → cohesion ≤ **max_s τ_coh(s)** AND
  ASR-majority ≥ target.
- **cost-imposition:** only the **b=400 discrete-token steelman** reaches it.
- **empirically-robust:** even **b=400** cannot realize cohesion ≤ τ_coh at target ASR in text.

**Honesty clause:** cheap evasion → report **complementarity** (E1×E4-HH: such poison is corpus-diffuse
+ query-retrievable = RAGDefender's per-query catch zone). Do **NOT** escalate to the steelman to
"manufacture robustness" once cheap evasion succeeds.

**Provenance (attack inputs).** Save the LLM-generated poison set **and** the generator (script + model
id) to disk as a versioned artifact — the provenance invariant applies to attack inputs, not only
results. Cheap-end run compares to τ_coh(42)=0.8442 directionally; the binding comparison uses
max_s τ_coh(s) (3-seed calibration, deferred); the 0.84-vs-0.99 margin makes the directional read sound.
