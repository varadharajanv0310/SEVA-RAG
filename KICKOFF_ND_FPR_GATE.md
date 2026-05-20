# KICKOFF_ND_FPR_GATE.md — run the clone-inject defense gate (steps 1–2 only)

*Created 2026-05-31. Paste-ready prompt for a FRESH Claude Code session to execute the cheap
go/no-go gate for the near-duplicate (`s_nd`) defense against clone-inject. The whole idea lives
or dies on one number — `s_nd`'s clean FPR on Security-SE — so we measure ONLY that before
spec'ing the full gauntlet.*

---

## 1. The prompt — copy everything in the block as your first message

```
You are resuming the SEVA-RAG project — an LLM-free RAG corpus-poisoning detector being prepared
for publication. You have NO memory of prior sessions; everything is on disk. Work from
D:\SEVA-RAG\SEVA-RAG.

=== STEP 0 — ORIENT (read in order, fully) ===
1. PROJECT_STATE.md            (resume pointer; note §4 env, §5 corpus/cache paths)
2. PAPER_STRUCTURE_A.md        (the LOCKED paper frame: scoped-positive A)
3. PAPER_EDITS_LOG.md          (read the FRAME-A block at top; entries E1-1, E1-2, E4-HH, E-CAL-1
                                for clone-inject context; and the ND-PROPOSAL note)

=== CONTEXT — what we're doing and why ===
- FRAME (LOCKED): SEVA is a scoped-positive paper — a lightweight, LLM-free, CPU/MacBook-deployable
  detector of TEMPLATED / near-duplicate corpus poisoning: 0% ASR @ ~0.6% FPR (frozen held-out,
  3-seed), sub-2 ms on Apple Silicon / ~12-15 ms on CUDA, beating the per-query SOTA (RAGDefender)
  on deployability.
- THE ONE BOUNDARY: a low-prominence "clone-inject" attack (clone a real in-corpus doc, inject a
  small payload) evades SEVA's cluster_coh AND RAGDefender, and flips RAG answers 8/8 (E1-1, E1-2,
  E4-HH, E1-4). This is the paper's single biggest weakness.
- THE IDEA WE'RE TESTING: clone-inject's defining property is that the poison is a NEAR-DUPLICATE
  of a document already in the corpus (that is how it inherits a clean cluster_coh neighborhood AND
  retrievability). cluster_coh = MEAN PAIRWISE cohesion of the 5 nearest neighbors, so it misses the
  lone twin. BUT the clone's cosine to its SINGLE nearest corpus neighbor (the host) is ~0.98-0.99 —
  a dead giveaway. So a new signal s_nd = MAX cosine to nearest corpus neighbor should catch it.
  s_nd is LLM-free, ~free (reuses the index SEVA already builds), CPU, <30 ms -> preserves core
  identity. It is principled: templated poison = CLUSTERED near-dups (cluster_coh), clone-inject =
  LONE near-dup twin (s_nd) -> one unified "injected near-duplication" story; and it is a
  CORPUS-LEVEL signal the per-query SOTA structurally cannot replicate.
- THE WHOLE IDEA LIVES OR DIES ON ONE NUMBER: s_nd's CLEAN FPR on Security-SE. The clone sits at
  ~0.99; two genuinely-different security questions sit at ~0.7-0.9. If a threshold (~0.96) that
  catches the clones has clean FPR <= ~0.69% -> GREEN (signal alive, spec the full gauntlet later).
  If Security-SE is full of legitimate near-duplicates that push the FPR up -> RED (dead on arrival;
  keep scoped-A with clone-inject as a one-line limitation). Cheap decisive gate first; full
  commitment only if it passes.

=== YOUR TASK — run STEPS 1-2 ONLY, then STOP ===
Do NOT spec or run the full E-ND gauntlet (no ROC protocol design, no frozen re-validation, no
adaptive attack). Just these two cheap measurements, reusing CACHED embeddings (never re-embed the
100k corpus):

STEP 1 — Detection: does s_nd flag the clone-inject poison?
- Find the clone-inject poison docs (+ host doc ids) in whitebox_attack_results/ (from
  whitebox_attack_seva.py squeezegen; see squeeze_retrieval_s042.json). If the exact clone
  texts/hosts are not recoverable, regenerate a small set the documented way: clone the
  most-retrievable Security-SE docs + inject the payload at LOW prominence (1 rep, matching E1-1).
- Cached clean corpus embeddings (100k x 1024, normalized): seva_checkpoints_4060_100k_secqa_*/
  p2_pe.npy + index p2_faiss.index (also in seva_results_5080_secqa_backup_20260530/). Embed ONLY
  the (few hundred) clone docs with bge-large (BAAI/bge-large-en-v1.5, normalize_embeddings=True).
- For each clone, compute s_nd = MAX cosine to the CLEAN corpus embeddings (clone is NOT in that
  set, so no self-match). Report the clone s_nd distribution (expect ~0.98-0.99).

STEP 2 — FPR (THE decisive number): s_nd's clean false-positive rate
- For each CLEAN Security-SE doc, compute s_nd_clean = cosine to its nearest OTHER clean corpus doc
  (query the cached index for top-2, drop self, take top-1 cosine). One index sweep over 100k.
- Report the clean s_nd distribution (mean/median/p90/p99/max) and the ROC: for thresholds
  tau_nd in {0.93,0.95,0.96,0.97,0.98}, report (clone-catch-rate from Step 1, clean-FPR).
- HEADLINE NUMBER: at the LOWEST tau_nd that catches ~100% of clones, what is the clean FPR?
  Compare to the 0.69% target.
- Check build_a1_corpus.py for the corpus dedup method/threshold — it sets the clean s_nd ceiling
  and thus the FPR floor; state how it affects interpretation.

GATE VERDICT (report, do NOT decide the paper):
- GREEN: a tau_nd catches ~all clones at clean FPR <= ~0.69% (or acceptably low) -> recommend
  spec'ing the full E-ND gauntlet.
- RED: no tau_nd separates clones from clean at acceptable FPR -> signal dead on arrival; recommend
  keeping scoped-A with clone-inject as a limitation.
- AMBER: partial separation -> report the tradeoff curve; author decides.

=== STANDING RULES (HARD) ===
1. Core-identity invariant: LLM-free at detection, CPU/weak-hardware, <30 ms. s_nd satisfies this
   (reuses the index). Introduce nothing that needs an LLM/GPU/internet at detection.
2. Detector byte-frozen: s_nd is a NEW signal for a NEW experiment — compute it in a SEPARATE
   standalone script (like the whitebox tooling); do NOT modify seva_benchmark_4060.py's existing
   signals/constants. The frozen baseline stays the ablation.
3. Reuse cached embeddings — NEVER re-embed the 100k corpus (paths above).
4. Provenance: save results to a JSON (e.g. whitebox_attack_results/nd_gate_s042.json); every number
   traces to a file.
5. Log as you go: when the gate lands, update the ND-PROPOSAL note in PAPER_EDITS_LOG.md to an
   ND-GATE result entry.
6. Gates are the author's: STOP after Step 2 with the numbers; do NOT start the full gauntlet.
7. Git: experiments live on exec/step2-e2-corpus (LOCAL). Commit locally; do NOT push (author is
   deciding the remote). Nothing merges to main.

=== ENV (conda NOT on PATH) ===
& "C:\Users\varad\miniconda3\Scripts\conda.exe" run --no-capture-output -n seva python ...
bge-large = BAAI/bge-large-en-v1.5, normalize_embeddings=True. (See PROJECT_STATE §4.)

Report at the gate: clone s_nd distribution; clean s_nd distribution + the ROC; the headline
FPR-at-clone-catching-threshold; and GREEN/RED/AMBER. Then STOP.
```

---

## 2. How to proceed
1. Open a fresh session in `D:\SEVA-RAG\SEVA-RAG`. Paste the block above as the first message.
2. Expect: it reads 3 docs, writes a small standalone script, reuses cached embeddings, runs two
   cheap measurements (~minutes to ~1 hour), and reports the FPR gate. No corpus re-embed.
3. **Red flag — interrupt:** if it starts modifying `seva_benchmark_4060.py`, re-embedding the
   corpus, or spec'ing/running the full gauntlet (adaptive attack, frozen re-validation) before the
   FPR number is in, stop it. The task is the cheap gate ONLY.
4. **The decision is yours** once it reports GREEN/RED/AMBER.

## 3. The gate decision rule (what GREEN/RED mean for the paper)
- **GREEN** (clone-catching τ_nd has clean FPR ≤ ~0.69%): the signal is alive → spec the full
  `E-ND` gauntlet (ROC, frozen re-validation that templated 0% + FPR + latency hold with `s_nd`,
  and the adaptive attack on the augmented detector). Potential payoff: clone-inject flips from the
  paper's biggest weakness to a headline strength.
- **RED** (no clean separation at acceptable FPR): Security-SE has too many legitimate near-dups →
  the signal is dead on arrival → keep the scoped-A paper; clone-inject stays a one-line §5
  limitation. **No loss vs. the current strong position.**
- **AMBER:** report the tradeoff; the author weighs the FPR cost against closing the boundary.

## 4. Fallback
Deep pre-context detail (if the on-disk docs don't suffice): the prior advisory-session transcript
documents the full E1→E4-HH→framing reasoning. The on-disk docs (`PROJECT_STATE.md`,
`PAPER_STRUCTURE_A.md`, `PAPER_EDITS_LOG.md`) are canonical and should be enough.
