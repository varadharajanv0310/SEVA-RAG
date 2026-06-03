=========================  PASTE INTO A FRESH CLAUDE CODE CHAT ON THE 5080 (repo root)  =========================
(You're on the 5080, in the SEVA repo `D:\SEVA-RAG\SEVA-RAG`, with the `seva` CUDA env. Run ONE
experiment family, read it straight, report. Don't change the science. If a STOP/FATAL fires, report it
and wait.)

## What this is
The go/no-go test for an **encoder-generalization** claim: does `cluster_coh` detect the **geometry of
templating** (the attack), or just **bge-large's manifold**? You will run the FROZEN SEVA hard gate with a
**different encoder** (e5-large-v2) on the **same hash-verified corpus**, and read the result against a
**pre-registered** bar. This gates a Tier-1 reach, so it must be set up right AND read honestly.

## The three files are already in `xplat_handoff/` (read them, don't rewrite them)
- `PREREG_ENCODER.md` — the pre-registered PASS/WEAK/FAIL bands + the read-it-straight discipline.
- `encoder_config.py` — the encoder registry + correct-by-construction embedding. **Key fact:** `cluster_coh`
  is a *symmetric* doc–doc similarity signal, so **e5 uses the `"query: "` prefix on ALL texts** (its
  symmetric-task convention); bge/gte use none. This is pinned correct already.
- `encoder_xrun.py` — the runner: it IS `scale_xrun.py` at N=100000 with only the embed swapped for the
  encoder-parameterized one. Every line of the **detector** (`doc_coh_full`, `retrieve_topk`, non-oracle τ,
  the grid math) is the same frozen `seva_xplat_common` code — nothing in it touches the detector.

## Hard rules (do not deviate)
- **Detector frozen.** Do not edit `seva_xplat_common.py`, the detector math, or any constant
  (K=5, K_FETCH=20, FPR_TARGET=0.0069, M=32). Only the encoder changes.
- **Same corpus, gated.** The runner STOPs (FATAL) unless the corpus is canonical `28ec3811…` + fingerprint
  and poison is `4f7ee3f3…`. Only the embeddings are encoder-new (cached per-encoder; never reuse another
  encoder's `.npy`).
- **5080 only.** Don't touch the hardware axis (that's tested separately).
- **Read it straight.** The runner computes the pre-registered verdict into the JSON. Do **not** move the
  bands. Do **not** re-run a clean, sanity-passing, bge-validated encoder to chase a better number — that's
  forbidden by the prereg. A re-run is allowed ONLY to fix a *documented* misconfiguration (a failed sanity
  check or a failed bge reproduction gate).

## STEPS

**0) Pre-register (commit BEFORE any result).** From the repo root:
```
git add xplat_handoff/PREREG_ENCODER.md xplat_handoff/encoder_config.py xplat_handoff/encoder_xrun.py
git commit -m "prereg(encoder-generalization): bands + runner + encoder config (before results)"
```
Commit locally, **no push**. (This timestamps the prereg ahead of the numbers.)

**1) Env + syntax.** Use the existing `seva` CUDA env. From `xplat_handoff/`:
```
python -c "import numpy,torch,sentence_transformers,faiss,huggingface_hub; print('cuda', torch.cuda.is_available())"   # expect True
python -m py_compile encoder_xrun.py encoder_config.py
```
e5/gte download on first use (a few hundred MB each); allow it. Install nothing unless an import fails.

**2) HARNESS-CORRECTNESS GATE — reproduce bge first (this validates the runner independently of e5).**
```
python encoder_xrun.py --encoder bge
```
~45–70 min (embeds 100k with bge, runs the 3×3 grid; resumable — re-run the same command if interrupted).
Then compare `result_encoder_bge.json` to the existing `result_scale100k.json`:
- gap per density within **±0.01** of **0.236 / 0.241 / 0.245**, ASR **0%** on all 9, density-invariant.
- If it reproduces → the runner is correct; proceed. **If it does NOT reproduce → the harness is buggy
  (NOT the encoder). STOP, report, fix the harness.** Do not run e5 on a harness that can't reproduce bge.

**3) THE TEST — e5.**
```
python encoder_xrun.py --encoder e5
```
Watch the `SANITY` line it logs early (before the long embed):
- `dim` = 1024, `normalized_ok` true, `retrieval_sane` true, and `sample_doc_input` literally begins with
  **`query: `** (confirms the prefix is applied). If sanity FATALs (exit 4), the encoder is misconfigured —
  report it; that's a fixable config issue, not an e5 result.
~45–70 min. Resumable.

**4) Read it straight + commit.** Open `result_encoder_e5.json`. The `verdict.result` is the runner's
self-call (PASS / WEAK / FAIL) against `PREREG_ENCODER.md`. Validity-gate it first (all must hold):
`corpus.hash_match` true **and** `fingerprint_check.ok` true; `poison.hash_match` true;
`detector.composite_used` false; `encoder.doc_prefix` == `"query: "`; `sanity.dim_ok/normalized_ok/
retrieval_sane` true; `n_docs` 100000. Then commit the results (they're git-ignored, so force-add):
```
git add -f result_encoder_bge.json result_encoder_e5.json
git commit -m "encoder-generalization: bge reproduction + e5 result (read straight vs prereg)"
```
No push.

**5) Report + branch.** Send back, straight:
- bge reproduction: gap-per-density vs 0.236/0.241/0.245, ASR — did it reproduce? (yes/no)
- e5: `verdict.result`, and the measured `gap_per_density`, `snr_min`, `asr_max_pct`, `gap_range_rel`,
  `grand_mean_docfpr_eval_pct`.
- The branch (from `PREREG_ENCODER.md`):
  - **e5 PASS** → stop and ask the operator before running `python encoder_xrun.py --encoder gte`
    (the 3rd-lineage follow-up toward an "encoder-invariant" claim).
  - **e5 WEAK / FAIL** → **STOP.** Report straight with the JSON. Do not run gte. The paper rescopes.
- The 2M-scale run is NOT part of this.

Don't analyze beyond the verdict, don't tune anything, don't run anything else.
=========================  END  =========================
