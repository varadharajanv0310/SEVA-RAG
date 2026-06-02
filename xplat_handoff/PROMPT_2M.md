=========================  PASTE INTO A FRESH CLAUDE CODE CHAT ON THE 5080 (executor session)  =========================
(This is the 2M-document SCALE experiment — the one run deferred from the encoder / cross-platform work.
It runs OVERNIGHT while the operator sleeps, so it must survive disconnection and never die silently.
Don't change the science. If a FATAL fires, log it, write it to the result, exit non-zero, and stop.)

## Why this run exists
The paper currently states (Limitations, "Corpus-scale validation") that behaviour at the
multi-million-document scale is **extrapolated from the index's O(log N) structure, not measured.**
This run MEASURES it, at 2,000,000 documents (or the largest honest in-domain scale available).

## Goal — three things, in priority order
1. **Latency at scale** — confirm per-query latency stays *flat / logarithmic* (O(log N)) at 2M,
   validating the deployment claim at production scale. (This is the primary deferred claim.)
2. **Calibration FPR at scale** — the THIRD point of the O(1/√N) calibration-convergence law
   (Observation 2; currently 10k + 100k). Does the non-oracle Doc-FPR keep converging toward the
   0.69% target at 2M?
3. **Detection at scale** — does the `cluster_coh` gap stay density-invariant and the gate hold
   0% templated ASR when the corpus is 20× larger? (Observation 1's K=5 mechanism predicts yes.)

## Hard rules (same discipline as every prior run)
- **Detector byte-frozen.** Reuse the frozen `seva_xplat_common` primitives (`doc_coh_full`,
  `retrieve_topk`, the non-oracle τ, `K=5`/`K_FETCH=20`/`M=32`/`FPR_TARGET=0.0069`). Do NOT modify the
  detector. Build the 2M runner by **extending `scale_xrun.py`** (it already does
  build → hash-gate → embed → per-density index+coh → non-oracle τ → grid); the only changes are
  corpus scale, **resumability**, and the 2M corpus build.
- **Corpus-integrity gate.** The 2M corpus must be built DETERMINISTICALLY and hash-gated: emit an
  order-sensitive SHA-256 + a per-doc fingerprint, exactly as the 100k canonical (`28ec3811…`) is
  gated; record both in the result + a fingerprint file so the run is reproducible. Pin the source
  data revision.
- **In-domain only; honest scale.** Build the largest REPRODUCIBLE in-domain (technical Q&A —
  Security / Stack-Exchange-family) corpus you can, up to 2M, deduplicated with the same dedup as the
  100k. **If the in-domain source caps below 2M, run at the achievable scale (e.g. 1M) and SAY SO —
  do NOT pad with out-of-domain documents to hit 2M** (that would break the calibration/gap
  comparison with the 10k/100k points). Document the true N.
- **Poison:** regenerate deterministically (`xplat_poison_gen`), injected at `corpus[0:P]` for
  `P = density × N` at 1/5/10%. If the generator's variant cross-product caps below the largest P
  (10% of 2M = 200k), run the densities it can fill and document the cap — do not fabricate poison.
- **Pre-register before reading.** Commit the registered expectations (below) BEFORE looking at any
  number. Read the result STRAIGHT against them; a miss is a real finding (**private negative — report
  it with the JSON, do not bury, do not re-run-until-favorable**). A re-run is allowed only to fix a
  documented crash/misconfig, never to chase a number.
- **2M is the ONLY run here.** No tuning, nothing else.

## Pre-registered expectations (commit before reading)
- **Latency:** per-query mean stays within ~2× the 100k figure (≈13–16 ms on the 5080) — flat to
  logarithmic in N, NOT linear. A linear blow-up would refute the O(log N) deployment claim.
- **Detection:** `cluster_coh` gap density-invariant (range ≤ 0.05, every gap > 0.15) and 0% templated
  ASR on every condition, as at 100k.
- **Calibration:** non-oracle grand-mean Doc-FPR stays near the 0.69% target and its deviation does
  not grow with N (consistent with Obs 2's O(1/√N) convergence; 100k deviated 0.016%).
- **Private negative** = any of: latency super-logarithmic; gap collapses or ASR > 0; FPR diverges
  from target. Report straight.

## Overnight safety (the operator is ASLEEP — the run MUST survive)
- **Resumable from checkpoints.** Every expensive phase writes its artifact to disk and is skipped if
  present on restart: (a) the 2M corpus + its hash/fingerprint; (b) embeddings **chunked, each chunk
  saved as it completes** (a crash resumes mid-embed, not from zero); (c) per density: the FAISS index
  and the `doc_coh` array, each saved after it finishes; (d) per-(density,seed) grid rows appended to a
  progress JSON as they complete. **Re-running the SAME command resumes from the furthest checkpoint.**
- **Run detached + captured logs.** Launch under `nohup`/background so it survives SSH/terminal
  disconnection; redirect stdout+stderr to a timestamped log file; flush per chunk. Print a
  **heartbeat** (timestamp + phase + progress %) at least every few minutes so a stall is visible —
  **no silent death.**
- **Fail loud.** On any FATAL (corpus hash mismatch, OOM, poison cap) write the reason to the log AND
  the result JSON and exit non-zero — never emit a partial "success".
- **Memory.** 2M embeddings (fp32) ≈ 8 GB; the HNSW index at 2M ≈ 9 GB → budget ~25–30 GB system RAM.
  **Check available RAM FIRST**; if short, memory-map the embeddings / reduce batch and LOG the
  mitigation. The detector math stays frozen regardless.
- **Runtime prediction FIRST (before the full launch):** time a **200k-document micro-build** (embed +
  one HNSW index + one `doc_coh` pass), then extrapolate (embed ∝ N; HNSW build ∝ N log N; coh ∝ N log N)
  to 2M × 3 densities, and **PRINT your predicted wall-clock + the dominant term BEFORE starting the
  overnight run. State it to the operator. If it predicts ≫24 h, say so before launching.**

## Steps
0. **Pre-register** (commit the expectations above). No push.
1. **RAM/disk check + the 200k micro-benchmark → print YOUR OWN 2M runtime estimate** (above).
2. **Build** the 2M (or largest in-domain) corpus deterministically; emit + record its order-sensitive
   SHA-256 + fingerprint; pin the source revision.
3. **Embed** (chunked, resumable, logged) → per-density **index + doc_coh** (saved) → non-oracle τ
   (60/40 split, seeds 42/7/123) → score the grid → measure per-query latency (same protocol as
   `tab:xplat`: encode 1 query + FAISS retrieve + gate-check).
4. **Emit `result_2M.json`:** `env`; `corpus{N, canonical_sha256, fingerprint_ok, source_revision}`;
   `detector{frozen, K…}`; `poison{hash, per-density P or cap}`; `grid` (gap/SNR/ASR/DocFPR per
   density×seed); `latency{mean, p95, ref_100k}`; `scale_summary{grand_mean_docfpr_pct vs 0.69%,
   vs 10k/100k}`; and a **self-verdict** against the pre-registered expectations.
5. **Commit** the prereg, `result_2M.json`, the runner, the fingerprint, and the log (force-add JSONs;
   **no push**). **Report:** predicted vs actual runtime, the three pre-registered readings
   (latency / detection / calibration) STRAIGHT, and any private negative.

## My (prompt-author) runtime estimate, for your planning
**~12–20 h wall-clock on the 5080**, dominated by (a) embedding 2M docs (~3–5 h, linear in N) and
(b) the three FAISS HNSW index builds + three K=5 coherence passes at 2M (~8–14 h combined; the HNSW
build at 2M with M=32 / efC=200 is the largest and least-certain term). Corpus build/dedup adds ~1–2 h.
Treat this as a planning band, not a promise — **your 200k micro-benchmark extrapolation (Step 1) is the
number to trust.**

## If the poison generator or in-domain source caps below 2M
The **latency + calibration** results (the primary O(log N) / Obs-2 goals) still stand at whatever N you
reach — report them and note the true N and the detection densities actually achieved. Don't widen
scope; don't tune; 2M (or the largest honest in-domain N) only.
=========================  END  =========================
