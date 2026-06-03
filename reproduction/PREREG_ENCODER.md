# PRE-REGISTRATION — Encoder-Generalization of `cluster_coh`
**Commit this file (with `encoder_config.py` + `encoder_xrun.py`) BEFORE reading any result.**

## The question (go/no-go for an encoder-generalization claim)
Does `cluster_coh` detect the **geometry of templating** (the attack), or merely the
manifold of **one encoder** (`bge-large-en-v1.5`)? If the templated-poison-vs-clean
separation survives — density-invariantly, at the non-oracle operating point — under a
**different encoder used correctly**, the paper may claim encoder-generalization. If it
materially weakens, the claim is **scoped to what holds, or dropped**. This is the single
result that gates a Tier-1 reach, so it is set up to give the claim its fair, best shot
**and** to be read honestly.

## Fixed vs varied
- **FROZEN — detector:** `cluster_coh` (mean cosine to `K=5` nearest corpus neighbours),
  hard gate (flag iff `coh > τ`), **non-oracle** calibration (`τ = (100 − 0.69)` percentile
  of **clean** coh), `K=5`, `K_FETCH=20`, HNSW `M=32`/`efC=200`, `FPR_TARGET=0.0069`.
  The runner calls the **same** `seva_xplat_common` functions the validated scale run used.
- **FROZEN — corpus + poison TEXT:** canonical hash-gated (corpus `28ec3811…f8a9` + per-doc
  fingerprint; poison `4f7ee3f3…733c`). Only the **embeddings** are encoder-new.
- **FROZEN — machine:** 5080 / CUDA for every encoder. Hardware is tested separately by the
  bge cross-platform runs; **do not cross the encoder and hardware axes.**
- **VARIED — the embedding encoder only.** `e5-large-v2` first (different lineage); then
  `gte-large` **iff** e5 passes.
- **HONEST per-encoder calibration:** `τ` re-derived **non-oracle** on that encoder's **clean**
  coh. **Correct per-encoder convention** (`encoder_config.py`): `cluster_coh` is a **symmetric**
  similarity/clustering signal, so **e5 uses the `"query: "` prefix on ALL texts** (its
  symmetric-task rule); bge/gte use none. Mis-prefixing is the canonical false-negative trap;
  it is pinned correct in config and checked by the sanity gate + the bge reproduction gate.

## Scale
`N = 100,000` only · 3 densities (1/5/10%) × 3 seeds (42/7/123). **10k is deliberately
excluded:** 10k×1% (P=100) sits at the known small-absolute-count boundary — an axis
**orthogonal** to the encoder question — and including it would confound the comparison.

## Pre-registered metrics (per density, 3-seed)
- `gap = poison_coh_mean − clean_coh_mean` — absolute scale is **encoder-dependent**.
- `SNR = gap / clean_coh_std` — **scale-normalized** separation → the fair cross-encoder measure.
- `ASR` at the non-oracle `τ` — operational: does the gate still **catch** templating?
- `DocFPR` (grand-mean held-out) — calibration sanity.
- density-invariance — `gap` range across {1,5,10}% as a fraction of the mean gap.

## Pre-registered verdict — DECIDED NOW, before any e5 number is seen
Read the e5 result against these bands **as-is**. Do not move them. (The runner computes this
verdict itself into the JSON.)

**PASS — encoder-generalization SUPPORTED (e5 carries the claim) — ALL of:**
- **P1.** `ASR = 0%` on all 9 conditions (zero templated-poison evasions at the non-oracle `τ`).
- **P2.** `SNR ≥ 3.0` at every density (clear separation; bge ref ≈ 5.9–6.1).
- **P3.** density-invariant: `gap` range across {1,5,10}% ≤ **25%** of the mean gap.
- **P4.** calibration sane: grand-mean held-out `DocFPR ≤ 1.5%` (≈ 2× the 0.69% target).

**WEAK — PARTIAL; rescope honestly; STOP (do NOT run gte) — any of:**
- `0% < ASR ≤ 10%` on some condition; or
- `1.5 ≤ SNR < 3.0` at some density; or
- `gap` range `> 25%` of the mean (separation present but density-drifting).
→ Paper scopes to: *"strong under bge; under e5 the templating geometry is [weaker /
density-dependent] — encoder-generalization is partial."*

**FAIL — encoder-generalization NOT supported — any of:**
- `ASR > 10%` on any condition; or
- `SNR < 1.5` at any density (separation collapses → the signal was largely bge-specific); or
- calibration cannot hold (grand-mean `DocFPR > 3%` or `τ` degenerate).
→ **Drop** the encoder-generalization claim; scope `cluster_coh` to bge-large; publish the
negative honestly as a limitation (the gate is encoder-sensitive).

*Reference shape (CONTEXT, not a goalpost): bge 100k ≈ gap 0.236–0.245, SNR 5.9–6.1, ASR 0%,
DocFPR 0.67–0.93%.*

## Harness-correctness gate (independent of the e5 outcome)
Before trusting **any** e5 number, run the **same** runner with `--encoder bge` and confirm it
**reproduces** the known bge 100k result (`result_scale100k.json`): gap within **±0.01** of
0.236 / 0.241 / 0.245, `ASR = 0%` on all 9, density-invariant. This proves the runner is
correct **independently of whether e5 looks good** — so a weak e5 cannot be dismissed as a
harness bug, and an e5 PASS cannot be a harness fluke. If bge does **not** reproduce → the
harness is wrong; **fix it**; that is not an e5 result.

## Re-run rule (anti–goalpost-moving)
One configured run per encoder (3 seeds). A re-run is permitted **only** to correct a
**documented misconfiguration identified independently of the result's favorability** — i.e. a
failed sanity check (wrong dim, un-normalized, prefix not applied, degenerate/broken
embeddings) or a failed bge reproduction gate. A correctly-configured, sanity-passing,
bge-validated encoder result is **read straight and recorded, whatever it says.** Re-running a
clean result because it is unfavorable is **forbidden** and voids this pre-registration.

## Conditional branches (the path)
- **e5 PASS** → run `gte-large` next (3rd lineage) toward an "encoder-invariant" claim.
- **e5 WEAK / FAIL** → **STOP.** Report straight. Rescope the paper before any further run.
- The **2M-scale** run is a separate later experiment; **not** part of this.

## Private-negative discipline
A WEAK/FAIL on a hash-matched corpus + sanity-passing, bge-validated encoder is a **real
finding**, reported straight to the operator with the JSON — not buried, not re-run-until-
favorable. The honest win is a true confirmation that carries the claim; a confirmation
manufactured by a soft setup is worse than a clean negative.
