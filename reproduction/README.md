# SEVA cross-machine handoff — package index

Turnkey package to run SEVA experiments **#1–#4** on two external machines (a **4060/CUDA** box and
an **Apple M4** box), each driven by a non-expert who only **pastes → enters → waits → returns one
JSON**. The goal: a **hash-verified identical in-domain corpus** on all three machines (5080 / 4060 /
M4), the **current cluster_coh hard gate** (never the v7.1.3 composite), under **frozen/non-oracle**
calibration.

## The experiments
- **#1 (M4) / #3 (4060):** in-domain hard-gate grid — 3 densities (1/5/10%) × 3 seeds (42/7/123) →
  cluster_coh **gap, SNR, ASR, DocFPR** per condition.
- **#2 (M4) / #4 (4060):** in-domain hard-gate **per-query latency** + device backend (MPS vs CPU vs CUDA).

## How identity is guaranteed (not just "same script + seed")
1. **Pinned inputs** (`MANIFEST.json`): exact HF dataset commit revisions for the Security-SE source,
   exact `bge-large-en-v1.5` encoder revision, seed=42, Python 3.11, the frozen constants
   (K=5, K_FETCH=20, FPR_TARGET=0.0069, …), and the exact templated poison (`poison_corpus_diverse.json`,
   shipped — not regenerated).
2. **Local rebuild + re-embed** from those pinned inputs (the 100k corpus file is *not* shipped).
3. **Order-sensitive canonical gate (auto-enforced):** the canonical corpus is **pre-pinned** — an
   ORDER-SENSITIVE SHA-256 (per-doc text hash taken in corpus order; captures the set **and** order)
   plus a `corpus_fingerprint.txt` doc-by-doc check. Each machine rebuilds and the runner **STOPs**
   (naming the first divergent doc index) unless it matches the canonical corpus the paper used. Order
   matters because poison replaces `corpus[0:P]`. Verified: `build_corpus_xplat.py` at the pinned
   commits reproduces the canonical corpus byte-identically in set and order.

## Files
| File | Role |
|---|---|
| `seva_xplat_common.py` | shared: device detect (cuda/mps/cpu), env+cache report, corpus hash, **resumable/stderr embedding**, the **VERBATIM frozen** cluster_coh + retrieval math, constants |
| `build_corpus_xplat.py` | cross-platform, HF-revision-pinned rebuild of the exact in-domain corpus; emits the canonical hash |
| `hardgate_xrun.py` | **the one turnkey runner**: env → ignore old wikitext cache → build → hash-gate → embed once → 3×3 grid + latency → `result_<label>.json` (resumable) |
| `xplat_poison_gen.py` | the **deterministic** templated-poison generator (50 base docs × variant cross-product; no randomness) — regenerates the exact 10k poison in memory + hash-checks it, so no 6.9MB file ships |
| `corpus_fingerprint.txt` | 6.6 MB; ordered per-doc SHA-256 of the canonical clean corpus — the doc-by-doc gate the runner verifies the rebuild against (and localizes any divergence) |
| `MANIFEST.json` | the pins + the **pre-pinned** canonical corpus hash/reference + the pre-registered PASS checks |
| `requirements_install.md` | reuse-or-install the Python env, per platform |
| `PROMPT_4060.md` / `PROMPT_M4.md` | the exact paste for each machine's Claude Code (M4 is fully self-contained) |
| `OPERATOR_GUIDE.md` | **YOUR runbook** (5080): build bundle+reference, send files, collect JSON, **validity gate**, diverge-vs-confirm |
| `PREREGISTRATION.md` | registered verdicts + the private-negative rule |

## Order of operations (see OPERATOR_GUIDE.md for detail)
1. **5080 (you):** copy in the poison file → run `hardgate_xrun.py --label 5080` → fill `MANIFEST.json`
   (reference hash, pinned revisions, versions). 2. **Zip + send** the folder + the matching prompt to
   each person. 3. **They** paste, run, return `result_<label>.json`. 4. **You** run the validity gate
   (OPERATOR_GUIDE Step E) before folding anything in.

## Non-negotiables
Identical hash-verified in-domain corpus · current hard gate only · frozen/non-oracle calibration ·
never reuse the wikitext cache (fresh in-domain, reuse within a machine) · pre-register before reading
· private-negative reportable not auto-published · chunked/resumable/stderr embeds · turnkey for two
non-experts · operator validity gate before trusting any returned number.
