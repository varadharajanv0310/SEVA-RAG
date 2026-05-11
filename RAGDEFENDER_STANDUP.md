# RAGDefender Stand-up — Step 1 recon (does it run, what does it need)

**Status:** ✅ Cloned, stood up, smoke-tested. **NOT yet integrated against our corpus** (that waits for the final E2 corpus, per the execution gate). This file is the committed recon record; the RAGDefender repo + its env live **outside** this git repo.

## Locations (all outside the SEVA-RAG git repo)
- **Repo:** `D:\SEVA-RAG\RAGDefender` (cloned from `github.com/SecAI-Lab/RAGDefender`, MIT, **v0.2.0**). Packaged tool: `pyproject.toml`, PyPI readme, `tests/`, `examples/`, `docs/`, `claims/`.
- **Env:** separate conda env **`ragdefender`** (python 3.11) — **deliberately NOT the frozen `seva` env**; SEVA's detector/env untouched.
- **Smoke script:** `D:\SEVA-RAG\ragdefender_smoke.py`; setup log `D:\SEVA-RAG\ragdefender_setup.log`.

## What it requires
- Deps (lightweight, CPU): `torch, transformers, numpy, pandas, tqdm, scikit-learn, sentence-transformers`. **No LLM / decoder / generation deps.** No GPU required ("RAGDefender itself runs on CPU tensors; the embedder is the only model loaded").
- Default embedder `all-MiniLM-L6-v2` (~80 MB, auto-downloads). **Configurable to any HF id, incl. `BAAI/bge-large-en-v1.5`** → E4-HH can use SEVA's exact encoder.
- No datasets needed for basic operation (NQ/HotpotQA/poisonedrag are only for reproducing *RAGDefender's own* paper numbers, which we do not need).

## Architecture — confirmed from source (matches the E4-HH premise)
- **Per-query / post-retrieval:** API is `defender.defend(query, R=[...retrieved passages...])` → operates on the retrieved set, not the corpus. ✓
- **Density-estimating:** "Stage 1 estimates how many passages were poisoned (Nadv); Stage 2 picks which indices to drop." ✓
- **LLM-free, deterministic, CPU.** ✓
- `task_type` required: `single_hop` (clustering; NQ/MS MARCO) vs `multi_hop` (concentration; HotpotQA). No auto-detect. For our short factual adversarial queries, **`single_hop`** is the likely match — confirm at integration via `docs/algorithm.md`.
- API surface for E4-HH: `defend(query, R, return_indices=True)` → `(survivors, removed_indices)`; `Evaluator.evaluate(test_data, attack_method, task_type)` over `{query, retrieved_docs, poisoned_indices}` dicts → precision/recall/F1.

## Smoke test (passed)
Input 4 passages (2 correct "Paris", 2 adversarial "Lyon" at idx 1,3) → **survivors = the 2 Paris passages; removed indices = [1, 3]**. Behaves correctly out of the box.

## Effort estimate for E4-HH (later, on the E2 corpus)
- **Stand-up:** trivial (done, ~5 min).
- **Integration:** **~0.5–1 day** — adapter from our E2 eval output `(query, top-k retrieved, known-poison-indices)` → RAGDefender `Evaluator` format; configure `embedder="BAAI/bge-large-en-v1.5"`, `task_type="single_hop"`; run the **per-query adversarial-ratio sweep** (the regime-split headline). Comfortably within the 1–2 day time-box; **low risk of needing the non-reproduction fallback.**
- Read `docs/algorithm.md` (implementation-vs-paper note) before integration to configure faithfully.

## Benign warnings observed
- HF symlinks unsupported on Windows → caching works in degraded mode (more disk). Optional fix: enable Developer Mode / `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- Unauthenticated HF downloads → works, just rate-limited. Optional: set `HF_TOKEN`.

## Invariant compliance
RAGDefender is LLM-free and lives in a separate env + separate code path. Reproducing it does **not** touch SEVA's frozen detector or the core-identity invariant.
