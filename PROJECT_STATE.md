# PROJECT_STATE.md — READ FIRST (resume pointer for a fresh session)

*Last updated 2026-05-30. Single source of truth for project state, standing rules, env, and next actions.*

## 0. What this is
SEVA‑RAG: an LLM‑free RAG corpus‑poisoning detector. We are hardening the paper for a Q1 venue (**realistic = IJIS/JISA; reach = TDSC/TIFS**) by executing **`EXPERIMENT_PLAN.md` (rev. 5, approved)**.

## 1. ⚠ CURRENT GIT BRANCH (critical)
- **Active branch: `exec/step2-e2-corpus`** (pushed to origin). The integrated benchmark + all Step‑2/3 work lives here, **NOT merged to main**.
- `main` has: RTX‑5080 env migration, `EXPERIMENT_PLAN.md` rev.5, Step‑1 docs.
- A fresh session MUST: `git -C D:\SEVA-RAG\SEVA-RAG checkout exec/step2-e2-corpus` to see the integrated `seva_benchmark_4060.py` and the in‑domain results.
- **Branch strategy (standing rule):** docs/notes → main; **one feature branch per major execution step**; commit+push **before AND after** each step.

## 2. Where we are (status)
- Phases A–F (RTX 5080 env migration): **DONE**. Verified 5080 ≡ 4060 per‑seed (exact). Backup: `seva_results_5080_baseline_backup_20260529/`.
- Post‑migration verification (V‑Steps 1–6): **DONE**.
- `EXPERIMENT_PLAN.md` rev. 5: **APPROVED** (limitation triage, E1…E7, venue framing, honest acceptance read).
- **Step 1** (R reframes + RAGDefender stand‑up): **DONE** → `PAPER_REFRAMES_R.md`, `RAGDEFENDER_STANDUP.md`, `PAPER_EDITS_LOG.md`.
- **Step 2** (E2 in‑domain corpus + E3 query fixes): **DONE** on this branch.
- **Step 3** (in‑domain baseline, seed 42): **DONE — GATE REACHED.** → `STEP3_GATE_RESULTS.md`, `STEP3_FINDING_AND_DECISION.md`.
- **PENDING AUTHOR DECISION:** GO to E1/E1b with rescoped framing (see `STEP3_FINDING_AND_DECISION.md`). **Do NOT start E1/E1b/E5 until the author decides.**

## 3. The gate outcome in one paragraph
cluster_coh confound‑kill **CONFIRMED** (gap holds in‑domain, SNR higher). BUT kw_density SNR collapsed 38→~7 in‑domain (was partly domain‑confounded), and the **adaptive L2/L3 ASR exploded to 44–73%** (vs ~0–17% WikiText) — the composite's adaptive robustness was propped up by kw_density. L1 (all signals) still 0% ASR. → E1/E1b are now **decisive**; the multi‑signal adaptive claim must be **narrowed**. Full numbers in `STEP3_GATE_RESULTS.md`.

## 4. Environment & invocation (conda NOT on PATH — key gotcha)
- conda: `C:\Users\varad\miniconda3\Scripts\conda.exe` (NOT on PATH in any shell).
- **Run in seva env:** `& "C:\Users\varad\miniconda3\Scripts\conda.exe" run --no-capture-output -n seva python ...`
- seva env: **torch 2.11.0+cu128** (CUDA 12.8), **faiss‑cpu 1.7.4**, python 3.11.15, numpy 1.26.4, sentence‑transformers 3.0.1. GPU = RTX 5080 (Blackwell sm_120). Defined by `environment_5080.yml`.
- **conda env create/remove needs a ToS override** (conda 26.x ToS‑gates Anaconda defaults; do NOT accept the ToS): `$env:CONDA_CHANNELS='conda-forge'; $env:CONDA_DEFAULT_CHANNELS='conda-forge'` then call conda. `conda env create` does not accept `--override-channels`.
- **RAGDefender** (for E4‑HH): separate env `ragdefender`; repo at `D:\SEVA-RAG\RAGDefender` (outside git, MIT, v0.2.0); smoke‑tested. Per‑query + density‑estimating + LLM‑free (confirmed). `task_type` NOT locked — confirm `single_hop` vs `multi_hop` via its `docs/algorithm.md` at integration. The "RAGDefender requires an LLM" framing is FALSE.

## 5. The in‑domain corpus + EXACT run command
- A1 corpus (security SE Q&A, 100k): `D:\SEVA-RAG\a1_corpus\clean_corpus_security.json` (outside git; regenerate via `build_a1_corpus.py`).
- Held‑out benign queries (CF‑008): `D:\SEVA-RAG\a1_corpus\benign_queries_security.json` (3000 question titles).
- **Exact in‑domain run command** (seed 42 done; run 7 & 123 next on GO):
  ```
  & "C:\Users\varad\miniconda3\Scripts\conda.exe" run --no-capture-output -n seva python -u seva_benchmark_4060.py --multitier --mtcorpus 100000 --poison-ratio 0.01 --cal-seed <SEED> --clean-corpus "D:\SEVA-RAG\a1_corpus\clean_corpus_security.json" --benign-queries "D:\SEVA-RAG\a1_corpus\benign_queries_security.json" --corpus-tag secqa
  ```
  (run from `D:\SEVA-RAG\SEVA-RAG`). WikiText baseline = run WITHOUT those 3 args. Each multitier run ≈ **48 min** (each tier full‑embeds 100k ~13 min; security docs are longer than WikiText, so slower than the 33‑min WikiText run). `--poison-ratio` is ignored under `--multitier` (runs 1/5/10%); `--corpus-tag secqa` isolates caches/results from the WikiText baseline.
- In‑domain result files: `seva_v6_2_results_100k_secqa_p{010,050,100}_s042.json` (repo root, gitignored). In‑domain caches: `seva_checkpoints_4060_100k_secqa_*` (gitignored). Both backed up in `seva_results_5080_secqa_backup_20260530/`.
- ⚠ **Known quirk (not data loss):** `seva_multitier_summary.json` (repo‑root convenience file) is **NOT** corpus‑tagged, so a WikiText re‑run and a secqa re‑run overwrite each other's repo‑root copy. Harmless — the per‑tier `*_secqa_*`/WikiText JSONs ARE tagged and are the source of truth; the WikiText summary is preserved in `seva_results_5080_baseline_backup_20260529/result_jsons/` and the secqa one in `…secqa_backup_20260530/result_jsons/secqa_multitier_summary.json`. (If E1 tooling ever needs the untagged summary, read the per‑tier JSONs instead.)

## 6. NON‑NEGOTIABLE standing rules (accumulated across the project)
1. **Detector byte‑frozen:** never modify `text_features`, `_score`, `_snr_weights`, `cluster_coh`/`_compute_doc_coh`, or the constants (K=5, K_FETCH=20, FPR_TARGET=0.0069, SNR_MIN_ABS=0.5, SNR_LOG_CAP=1.0, NORM_PERCENTILE=90, BATCH_SIZE=32, TARGETED_Q=50, BENIGN_Q default 2000). If a change seems to need it → **STOP and flag.** (My Step‑2 edits touched ONLY argparse + `__init__` cache/result naming + phase‑1 corpus loading + query construction; detector verified byte‑frozen by diff.)
2. **Core‑identity invariant (overrides any better number):** the detector stays **LLM‑free at detection, CPU/weak‑hardware‑deployable, sub‑30 ms**. Offline attack/baseline tooling may use LLMs; the detector may not. Any change risking (a) LLM/API at detection, (b) CPU/weak‑HW break, (c) sub‑30 ms regression → **STOP and flag.**
3. **All paper edits batched to the END via `PAPER_EDITS_LOG.md`.** Do **NOT** edit manuscript prose. The paper `.tex` is **not** in this repo (only the PDF in `C:\Users\varad\Downloads\SEVA_Paper_v7_1_3.pdf`). Append a dated entry whenever an experiment changes a claim/number; commit+push the log.
4. **Provenance invariant:** every comparative number in the paper must trace to a results file produced on this machine.
5. **Git checkpoint discipline:** commit+push before AND after each step; large data artifacts stay out of git (`.gitignore` covers `*.json`/`*.log`/`seva_checkpoints*`; the in‑domain result JSONs are gitignored — back them up, don't commit).
6. **Gates are the author's:** stop at each gate and present numbers; the author makes go/no‑go. Do not run past a gate without the decision.
7. In THIS execution phase the author authorized me to run `git commit`/`push` (per‑step discipline). (An earlier *verification* session was "no git by me" — superseded for execution.)

## 7. File map (committed on `exec/step2-e2-corpus`)
- `PROJECT_STATE.md` — THIS FILE (read first).
- `EXPERIMENT_PLAN.md` — master plan rev. 5 (E1…E7, triage, venue, acceptance read).
- `STEP3_GATE_RESULTS.md` — authoritative Step‑3 gate numbers.
- `STEP3_FINDING_AND_DECISION.md` — the finding + pending decision (read for paper status).
- `PAPER_EDITS_LOG.md` — authoritative manuscript‑edit ledger (R‑1…R‑9, E2‑2/E3‑1, rescoping). Apply at the end.
- `PAPER_REFRAMES_R.md` — Step‑1 R reframes spec.
- `RAGDEFENDER_STANDUP.md` — RAGDefender recon (E4‑HH).
- `precheck_cohesion.py`, `build_a1_corpus.py` — corpus tooling.
- `seva_benchmark_4060.py` — the benchmark (detector frozen; +E2/E3 input‑side plumbing).
- `KNOWN_ISSUES.md` — original code/paper discrepancy audit (CF‑/W‑ items).
- Backups (NOT git): `seva_results_5080_baseline_backup_20260529/` (WikiText), `seva_results_5080_secqa_backup_20260530/` (in‑domain).

## 8. NEXT ACTIONS (on author GO)
1. Author decides per `STEP3_FINDING_AND_DECISION.md` (recommended: GO to E1/E1b with rescoped framing).
2. (Optional, for tables) run seeds **7, 123** in‑domain (command in §5) → full 3‑seed in‑domain tables + L2/L3 seed‑stability.
3. **E1** (algorithmic embedding‑space white‑box attack on cluster_coh) + **E1b** (necessity: is retrievability ⊥ low‑inter‑poison‑cohesion fundamental or only correlated?) per `EXPERIMENT_PLAN.md` §E1/E1b — now **decisive**. New module `whitebox_attack_seva.py` (subclass the bench; detector frozen). Wilson 95% CIs on each cost‑curve point.
4. Then **E4‑HH** (reproduce RAGDefender on the in‑domain corpus; regime‑split low‑per‑query‑density headline), **E5‑CAL** (ensemble calibration). Reach: **E6‑ENC** (e5‑large‑v2), **E7‑SCALE** (500k) — gated on E1b=necessity AND a faculty co‑author.
5. Keep appending paper‑affecting numbers to `PAPER_EDITS_LOG.md`; checkpoint commit+push around each step.
