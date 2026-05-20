# PAPER_STRUCTURE_A.md — the scoped-positive SEVA paper (the manuscript blueprint)

**Frame: identity A (scoped-positive).** Written 2026-05-31. This is the section-by-section
plan for the manuscript, mapped to the evidence in `PAPER_EDITS_LOG.md`. The pre-A full record
is preserved in `paper_frame_preA_backup_20260531/` and at git commit `beff46c`.

---

## 0. The spine (the one claim everything serves)

> **SEVA is a lightweight, LLM-free, domain-independent detector of *templated / near-duplicate*
> corpus poisoning — the dominant corpus-poisoning pattern in the literature — achieving 0% ASR
> at ~0.6% FPR under realistic (non-oracle, held-out) calibration, at 2–13 ms CPU latency, and
> out-performing the published per-query SOTA on deployability.**

**WE CLAIM (all evidenced, all traceable to a result file):**
- Detects templated/near-duplicate corpus poisoning (PoisonedRAG-style multi-passage injection).
- **Domain-independent** — validated in-domain on Security-SE, not just general corpora (E2).
- **Realistic calibration** — frozen, calibrated on a held-out reference set of the attack class, not oracle (E-CAL-1).
- **Strong operating point** — 0% ASR @ ~0.6% FPR, stable across 3 seeds (E-CAL-1, SEEDS-1).
- **Lightweight/fast** — 2–13 ms, CPU-only, weak-hardware deployable (latency data, R-2).
- **Beats the SOTA on deployability** — RAGDefender's native filter removes ~50% of clean docs in-domain; SEVA operates at 0.6% FPR; at matched FPR SEVA edges detection (100% vs ~89% templated) (E4-HH fair-comparison).
- **Reproducible** — bit-exact across CUDA GPUs (R-2).

**WE DO NOT CLAIM (the dropped overclaim — this is the whole fix):**
- ❌ "resists adaptive adversaries" / "multi-signal adaptive robustness" → **DROP THIS PHRASE EVERYWHERE.**
- ❌ "detects corpus poisoning" (unqualified) → always qualify to *templated / near-duplicate*.

Everything that "broke" SEVA broke a claim we are **not making**. Drop the one phrase and the
adverse findings are simply out of the scope we claim — nothing to defend, nothing to hide.

---

## 1. Section-by-section, with evidence mapping

### Title (working)
"SEVA: Lightweight LLM-Free Detection of Templated Corpus Poisoning in Retrieval-Augmented Generation"

### Abstract
Scoped claim + headline numbers (0% ASR @ 0.6% FPR, held-out frozen calibration, in-domain,
3-seed) + deployability win over SOTA + 2–13 ms. **No adaptive-robustness language.** (Apply R-2.)

### 1. Introduction & Contributions
- **C1** — a doc-level geometric signal (`cluster_coh`) that detects templated/near-duplicate poisoning **domain-independently**. (E2, E-CAL-1)
- **C2** — a **density-agnostic, universal-FPR calibration** that, trained on a reference attack sample and deployed *frozen*, generalizes to held-out poison of the same class (E-CAL-1). *This is the genuine methodological novelty — lead with it, not with "near-dup detection."*
- **C3** — an LLM-free, CPU-only, **2–13 ms** pipeline (deployability for weak hardware / no-internet).
- **C4** — a **reproduced head-to-head** against the SOTA per-query defense (RAGDefender) under identical conditions, showing SEVA's deployability advantage (E4-HH).
- Language: "we **show / argue**," never "we prove" (R-1).

### 2. Threat model (SCOPED — this section does the honesty work)
- Define the threat as **templated / near-duplicate corpus poisoning**: an adversary injects
  multiple adversarial passages engineered to be co-retrieved for a target query — the dominant
  pattern (PoisonedRAG; Zhong et al.; corpus-poisoning-by-injection line).
- **Explicitly scope out**: single-document injections crafted to statistically mimic the clean
  corpus distribution are outside the geometric-detection regime (forward-ref §5). Stating this
  *here* is what makes the rest of the paper honest — the limitations paragraph then just
  confirms a boundary the threat model already drew.

### 3. Method
- `cluster_coh` definition: doc-level mean pairwise cohesion of K=5 nearest corpus neighbors.
- Disclose `K_FETCH=20` over-fetch→rerank (R-5) and `NORM_PERCENTILE=90` (R-6).
- The signal ensemble → SNR-weighted A-score; 50 binary-search calibration iters (R-3).
- **Calibration protocol (disclose honestly — this is a rigor *feature*, not a confession):**
  signal weights are calibrated on a **reference sample of the templated attack class** and
  deployed **frozen**; we validate (E-CAL-1, §4.3) that frozen reference-calibration generalizes
  to held-out poison. (Resolves OPEN-CAL-1 by disclosure; most papers leave this implicit/oracle.)
- LLM-free, CPU pipeline (core-identity invariant).

### 4. Evaluation
- **4.1 Setup** — in-domain Security-SE corpus, 100k, deduped (E2). *Frame the in-domain corpus
  as a deliberate domain-confound control* — most defense papers use general corpora; we remove
  the domain shortcut. 3 seeds; disclose that seeds bound calibration-sampling variance (E3-2).
- **4.2 Detection performance (headline)** — **0% ASR @ ~0.6% FPR** on templated poison; stable
  across 3 seeds (E-CAL-1, SEEDS-1). Density stress at 1/5/10% and beyond — *we test poison
  densities orders of magnitude above prior work* (say so explicitly).
- **4.3 Calibration realism** — the **frozen held-out** result: calibrate weights+τ on templated
  half-A, evaluate held-out half-B with no recalibration → 0% ASR (E-CAL-1). This is the
  non-oracle number and a credibility differentiator.
- **4.4 Head-to-head vs SOTA (E4-HH fair-comparison)** — RAGDefender's native single_hop filter
  has **no no-attack gate** → ~50% in-domain benign FPR (undeployable in-domain); SEVA operates
  at 0.6%. At a matched (idealized) FPR, SEVA edges detection (100% vs ~89% templated). Lead with
  **deployability**, support with the matched-FPR detection edge.
- **4.5 Efficiency & reproducibility** — 2–13 ms mean/p95, CPU-only; bit-exact cross-GPU (R-2).

### 5. Limitations (ONE honest paragraph — RobustRAG-style; brief, expert, not self-demolition)
- **Scope of geometric detection:** SEVA targets structured near-duplication; an adversary who
  injects a *single* document statistically resembling the clean corpus would not exhibit the
  cohesion signal and is **outside SEVA's scope** — detecting such low-prominence injections with
  lightweight corpus-level signals remains open. *(This is the honest acknowledgment that
  E1-1/E1-2/E1-4 explored in the private record; you acknowledge the boundary without publishing
  your own attack recipe. Floor = acknowledge scope; you choose whether to go further.)*
- **Calibration assumption:** like all signal-calibrated detectors, SEVA assumes a reference
  sample of the target attack class; we show this generalizes to held-out poison of that class.
- **Adaptive evasion of auxiliary signals:** signals an adaptive adversary can suppress reduce
  the detector toward its geometric core; the core's scope is characterized above. (Keep to 1–2
  sentences — do NOT turn this into the L2/L3 demolition table.)

### 6. Related work
PoisonedRAG (attack, USENIX'25); RobustRAG (certified, ICML'24 — note it scopes to bounded
injection and reports 24–49% certified acc → precedent for honest scoping); RAGDefender (per-query
SOTA, ACSAC'25 — our head-to-head baseline); the 2026 corpus-dependent line (Semantic Chameleon
etc. — **read these for overlap/positioning before submission**). Position SEVA as the lightweight
LLM-free detector for the templated regime with realistic calibration and a *reproduced* head-to-head.

### 7. Conclusion
Restate scoped contribution; note the open boundary (clean-mimicking injection) as future work.

---

## 2. Evidence → role mapping (every log entry, where it goes)

| Log entry | Role in scoped-A paper |
|---|---|
| R-1 (show/argue not prove) | **IN** — claim language throughout |
| R-2 (CUDA reproduction; MPS caveat) | **IN** — §4.5 reproducibility |
| R-3 (50 iters) | **IN** — §3 method disclosure |
| R-4 (within-density SNR ratio) | **IN** — §3/§4, finalize number on E2 corpus |
| R-5 (K_FETCH=20) | **IN** — §3 |
| R-6 (NORM_PERCENTILE=90) | **IN** — §3 |
| R-7 (Lim 3 calibration floor) | **IN** — §5 (brief), numbers pending E5 if run |
| R-8 (Lim 5 two-point scaling) | **IN** — §5 brief / future work |
| R-9 (head-to-head framing) | **IN** — §4.4 (now backed by E4-HH fair-comparison) |
| E2-1, E2-2/E3-1 (in-domain) | **IN — headline** — §4.1/§4.2 (domain control = rigor feature) |
| E-CAL-1 (frozen held-out templated 0%) | **IN — headline** — §4.3 (the non-oracle positive number) |
| SEEDS-1 (3-seed stability) | **IN — headline** — §4.2 error bars |
| E4-HH fair-comparison | **IN — headline** — §4.4 deployability win |
| E3-2 (seed = calibration variance) | **IN** — §4.1 disclosure |
| OPEN-CAL-1 (oracle→frozen) | **IN as a STRENGTH** — §3 calibration disclosure + §4.3 |
| R-1's "we prove invariance" | **DROP the proof claim; keep as argued robustness of the core** |
| RESCOPE-1 ("narrow adaptive claim") | **MOOT** — the paper never makes the adaptive claim, so nothing to rescope; drop the phrase "resists adaptive adversaries" |
| E1-1 (templating≠poisoning boundary) | **§5 limitations (1 sentence)** — informs scope, not featured |
| E-CAL-2 (adaptive frozen ~49%) | **§5 limitations (≤1 sentence)** — auxiliary-signal evasion note; not a table |
| E1-3 (hash = tamper check) | **§3 method honesty** — describe hash accurately (don't call it near-dup) |
| E1B-1 (geometric probe) | **PRESERVED RECORD** — not in paper |
| E1-2 (clone-inject 88.8% frozen) | **PRESERVED RECORD** — your private attack, not in paper |
| E1-4 (8/8 answer-flip demo) | **PRESERVED RECORD** — not in paper |
| NOTE-1 (clone-inject subsection) | **SUPERSEDED** — under A, it is not a paper subsection |

**PRESERVED RECORD** = stays in `PAPER_EDITS_LOG.md` + git + backup as your research record
(so you can produce it if ever asked), but is **out of the paper's claimed scope** and not
published. This is legitimate: a scoped claim need not feature attacks the authors invented
against themselves.

---

## 3. The honesty floor (so this stays bulletproof)

1. **Never write "resists adaptive adversaries" or any unqualified "detects corpus poisoning."**
   Every robustness statement is scoped to templated/near-duplicate.
2. **Keep the §2 threat model explicitly scoped** and the §5 limitations paragraph present.
   With both, omitting the clone-inject *recipe* is honest (out of claimed scope), not deceptive.
3. **Disclose the frozen reference-calibration** (don't imply zero-knowledge detection).
4. **Preserve the data** (done: backup + git) so the record exists if challenged.

Meet these four and the paper is stronger *and* more honest than the field's published norm
(PoisonedRAG tests only basic defenses; RAGDefender ships a 50%-FPR filter; neither does a
freeze test, reproduced head-to-head, or in-domain control — you do all three).

---

## 4. What's left to do (manuscript pass, no new experiments required)
1. Confirm frame A (done — author 2026-05-31).
2. Read the 2026 corpus-dependent papers for positioning (§6).
3. Apply the **IN** entries to the `.tex` (not in this repo).
4. Write the scoped §2 threat model + the one §5 limitations paragraph.
5. Decide the venue from the scoped strength (RAGDefender-beating + realistic calibration is a
   solid mid-tier story; the rigor depth supports aiming higher if desired).
