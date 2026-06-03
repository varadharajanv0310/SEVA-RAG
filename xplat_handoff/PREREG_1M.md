# PRE-REGISTRATION — 1M-document corpus-scale run (deferred from the cross-platform work)
**Commit this file BEFORE reading any result.** 2026-06-03. 5080 / CUDA. In-domain technical SE.

## Why
The paper's Limitations ("Corpus-scale validation") currently EXTRAPOLATES multi-million-doc behaviour
from the index's O(log N) structure. This run MEASURES it at **N = 1,000,000** in-domain documents.

## Goal (priority order)
1. **Latency at scale** — per-query latency stays flat/logarithmic (O(log N)) at 1M (primary deferred claim).
2. **Calibration FPR at scale** — 3rd point of the O(1/√N) convergence law (have 10k, 100k); does the
   non-oracle Doc-FPR keep converging toward 0.69% at 1M?
3. **Detection at scale** — cluster_coh gap density-invariant and 0% poison-evasion at 10× scale (K=5 mechanism).

## Frozen / varied
- **Detector byte-frozen** — `seva_xplat_common` (`doc_coh_full`, `retrieve_topk`, non-oracle τ, K=5,
  K_FETCH=20, HNSW M=32/efC=200, FPR_TARGET=0.0069). NOT modified. The 1M runner EXTENDS `scale_xrun.py`
  (corpus scale + resumability + the 1M build only).
- **Corpus hash-gated** — the 1M corpus is built deterministically (pinned source revision, pinned site
  list, SEED=42, same `x[:300]` dedup as the 100k) and emits an order-sensitive SHA-256 + a per-doc
  fingerprint; both recorded in the result + a fingerprint file → reproducible. The runner FATALs on a
  hash mismatch on resume.
- **Poison** — regenerated deterministically (`xplat_poison_gen`), injected at `corpus[0:P]`, P=density×N.

## Documented scope (facts known before any result — NOT padding, NOT tuning)
- **Corpus source:** Security-SE alone caps ~103k. To reach 1M honestly IN-DOMAIN, the corpus is the
  **technical Stack-Exchange family** (IT / sysadmin / security / programming) — a PINNED site set
  (security, serverfault, superuser, askubuntu, unix, softwareengineering, dba, + networkengineering /
  codereview / crypto for margin), same filter/dedup/seed as the 100k. This is broader than the
  security-only 10k/100k points (which were a tighter domain) — **disclosed**; it is in-domain technical
  Q&A, NOT out-of-domain padding. The true N is whatever the pinned set yields after dedup, capped at 1M.
- **Poison density cap:** the generator caps at **96,000** unique variants. At 1M: **1% (P=10k)** and
  **5% (P=50k)** fill; **10% (P=100k > 96k) CANNOT fill** → the 10% density is SKIPPED and documented.
  Detection at 1M is therefore measured at **1% and 5%** only (vs 1/5/10% at 100k).

## Pre-registered expectations (fixed before results)
- **Latency:** per-query mean within ~2× the 100k figure (~13–16 ms on the 5080) — flat→logarithmic in N,
  NOT linear. A linear blow-up would refute the O(log N) deployment claim.
- **Detection:** cluster_coh gap density-invariant (range ≤ 0.05 across the *fillable* densities, every
  gap > 0.15) and 0% poison-evasion on every condition, as at 100k.
- **Calibration:** non-oracle grand-mean Doc-FPR stays near 0.69% and its deviation does NOT grow with N
  (Obs-2 O(1/√N); 100k deviated 0.016%).

**Private-negative rule:** any of — latency super-logarithmic; gap collapses or poison-evasion > 0; FPR
diverges from target — is a real finding, reported straight WITH the JSON, not buried, not re-run-until-
favorable. A re-run is permitted only to fix a documented crash/misconfig, never to chase a number.

## Overnight safety
Resumable from checkpoints (corpus+hash; chunked embeddings; per-density index+coh; per-(density,seed)
grid rows appended to progress JSON). Detached + timestamped log + heartbeat every few min. Fail-loud:
any FATAL → log + result JSON + non-zero exit, never a partial "success". Memory: mmap the 1M embeddings
(~4 GB) so peak stays ~8–9 GB of the 31 GB / 14 GB-free system. Runtime estimate from a 200k
micro-benchmark is printed BEFORE the full launch; if it predicts ≫ 24 h, stop and report first.

1M is the ONLY run. No tuning, nothing else. (2M remains a separate later experiment.)
