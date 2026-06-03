# OPERATOR_GUIDE.md — coordinating the two external runs (for YOU, on the 5080)

You are the coordinator. Two non-experts will each run one machine (a **4060/CUDA** box and an
**Apple M4** box). Their entire job is: **paste → enter → wait → send back one JSON.** Everything
else is handled by the bundle. This guide is your runbook: build the bundle + reference, send the
right files, collect each JSON, and **verify it before you trust it.**

The whole point is a **hash-verified identical in-domain corpus on all three machines**, the
**current cluster_coh hard gate** (never the v7.1.3 composite), under **frozen/non-oracle**
calibration. The 4060/M4 **rebuild** the corpus from pinned inputs and **re-embed locally**; a
canonical SHA-256 gate catches any divergence before any number is trusted.

---

## ORDER (externals-first is fully gated)
The clean-corpus identity is **pre-pinned and auto-enforced**: `MANIFEST.json` carries the
ORDER-SENSITIVE canonical hash of the exact corpus the paper used (`28ec3811...`), plus
`corpus_fingerprint.txt` (per-doc). Both were VERIFIED on the 5080 — `build_corpus_xplat.py` at the
pinned commits reproduces that corpus **byte-identically in set AND order**. So:
- The 4060/M4 **gate against the canonical corpus immediately**: the runner auto-STOPs (FATAL) if
  their rebuilt clean corpus diverges in set, order, or content, and reports the first divergent doc
  index. They do NOT need to wait for the 5080.
- You run the 5080 last; it re-verifies against the same canonical hash + fingerprint. (All three
  `corpus.canonical_sha256` will be equal, or that machine FATAL'd before producing numbers.)

## STEP A — Bundle + (eventually) the reference

**A1. The bundle is already complete** — every file is small text/code; **no big data file**. The
templated poison is REGENERATED on each machine by the shipped deterministic generator
(`xplat_poison_gen.py`, 50 base docs × variant cross-product, no randomness) and hash-checked against
the canonical hash in `MANIFEST.json` — so nothing 6.9 MB travels.

Bundle contents (verify all present in `xplat_handoff/`):
- `seva_xplat_common.py` · `build_corpus_xplat.py` · `hardgate_xrun.py` · `xplat_poison_gen.py` (code)
- `corpus_fingerprint.txt` (6.6 MB; ordered per-doc fingerprint of the canonical clean corpus — the doc-by-doc gate)
- `MANIFEST.json` (source/encoder revisions + the canonical corpus hash are **pre-pinned**; you fill only the numeric `reference` fields in A4) · `PREREGISTRATION.md`
- `PROMPT_4060.md` · `PROMPT_M4.md` · `requirements_install.md` · `README.md`

**A2. Run the reference on the 5080** (in the `seva` conda env, from inside `xplat_handoff/`):

```
& "C:\Users\varad\miniconda3\Scripts\conda.exe" run --no-capture-output -n seva ^
  python hardgate_xrun.py --label 5080 --out ./a1_corpus_xplat --cache ./seva_cache_secxplat
```

This builds the fresh in-domain corpus, embeds it, runs the 3×3 grid + latency, and writes
`result_5080.json` + `a1_corpus_xplat/build_provenance.json`. (≈ an hour; the embed is the long part.)
Confirm `result_5080.json → verdict.result == "CONFIRMS"` before going further — if the **5080**
doesn't confirm, fix that first; do not ship a broken reference.

**A3. Pin the source + encoder revisions** into `MANIFEST.json`:
- From `a1_corpus_xplat/build_provenance.json → source_repos`, copy each resolved commit SHA into
  `MANIFEST.pinned_source_revisions` (replace the two `null`s).
- Set `MANIFEST.pinned_encoder_revision` to the bge commit SHA. Get it with:
  `python -c "from huggingface_hub import HfApi; print(HfApi().model_info('BAAI/bge-large-en-v1.5').sha)"`

**A4. Fill the reference block** in `MANIFEST.json → reference` from `result_5080.json`:
- `corpus_canonical_sha256` ← `result_5080.json → corpus.canonical_sha256` (**the gate the others check**)
- `gap_per_density` ← `verdict.gap_per_density`
- `templated_asr_pct` ← should be `0`
- `docfpr_range_pct`, `latency_mean_ms` ← from the grid / `latency.mean_ms`
- `package_versions_5080` ← from `result_5080.json → env`
- set `produced_unix`.

Now `MANIFEST.json` is the single source of truth: pinned inputs + the reference hash + the
pre-registered PASS checks. **This filled manifest ships unchanged to both machines.**

> Why this is enough to guarantee identity: the corpus is a deterministic function of
> (source-data revision, seed=42, Python's stable Mersenne-Twister shuffle). We pin the first,
> the script fixes the second, and Step C pins Python to 3.11. The canonical SHA-256 then
> *proves* identity — anything unpinned that still diverges trips the hash gate before any run is trusted.

---

## STEP B — Send each person their files

Zip the **whole `xplat_handoff/` folder** (with the filled `MANIFEST.json` and the copied
`poison_corpus_diverse.json`) and send the **same zip to both** people (it's ~7 MB + tiny scripts —
the 100k corpus is NOT in it; it's rebuilt locally). Tell each person only:
> "Unzip this anywhere. Open your Claude Code chat in that folder. Paste the message I'm sending
> next, press Enter, and let it run (could be an hour or two). When it finishes it prints a file
> name — send me that file."

- To the **4060** person: also send the contents of **`PROMPT_4060.md`** as the paste.
- To the **M4** person: also send the contents of **`PROMPT_M4.md`** as the paste.

(That's the only difference between them: which prompt text you paste-send.)

---

## STEP C — What each pastes, and when

Each person pastes the single prompt you sent them into their Claude Code chat **once**, presses
Enter, and waits. The prompt instructs *their* Claude Code to: set up the env (pinned versions),
verify the bundle, run `hardgate_xrun.py`, and report the result file. No further input from them.
Order doesn't matter; they can run in parallel.

---

## STEP D — The one JSON each sends back

- 4060 person returns **`result_4060.json`**
- M4 person returns **`result_M4.json`**

Each must contain (the runner produces exactly this): `machine_label`, `env` (incl. `backend`),
`corpus.{canonical_sha256, reference_sha256, hash_match, n_docs}`, `detector.{type, composite_used}`,
`embedding_backend`, `grid` (9 cells: density, seed, gap, snr, asr_pct, docfpr_benign_retrieval_pct,
…), `latency.{backend, mean_ms, p95_ms}`, and `verdict.{gap_per_density, gap_density_invariant,
templated_asr_zero, result}`. If a run hit the hash gate it contains a top-level `FATAL` instead of
a full grid — that's an expected, designed stop (see Step G).

---

## STEP E — The validity gate (run this on every returned JSON BEFORE folding it in)

A result is **valid to fold in only if ALL of these hold**. Check them in order:

1. **Identical corpus + poison (auto-gated).** `corpus.hash_match == true` **and**
   `corpus.fingerprint_check.ok == true` — the rebuilt clean corpus matched the canonical
   ORDER-SENSITIVE hash **and** the doc-by-doc fingerprint. The runner auto-STOPs (FATAL) otherwise, so
   a returned *full* result already means the clean corpus is byte-identical (set + order + content) to
   the paper's. Also confirm `poison.hash_match == true`. (Eyeball that `corpus.canonical_sha256` ==
   `28ec38114ee64e6010ec489d01e6d3ee13d9b3758fd30a169c99ed078732f8a9` on all three.)
2. **Hard gate, not composite.** `detector.type == "cluster_coh_hard_gate"` **and**
   `detector.composite_used == false`.
3. **In-domain, not wikitext.** `corpus.n_docs == 100000`; the canonical hash matching (#1) already
   guarantees it's the in-domain Security corpus. Sanity-glance `preexisting_cache_report` — it
   should *report* an old wikitext `.npy` but the run must have used the fresh `secxplat` cache
   (embedding ran; `embedding_backend` is set, not `"cached"`-from-wikitext).
4. **Numbers plausible.** Per density, `gap ∈ ~[0.20, 0.27]` and seed-invariant (≈ equal across the
   3 seeds); `asr_pct == 0` on all 9; `docfpr_benign_retrieval_pct` < ~1.5%; SNR ~5–6.5. Latency
   sane for the platform (4060 tens of ms; M4 tens of ms — and `latency.backend` reported).
5. **Self-verdict.** `verdict.result == "CONFIRMS"`.

If 1–5 all pass → **fold it in** (record it as the platform's confirming run). If any fail → Step G.

---

## STEP F — Pre-registered pass/fail (decide the reading BEFORE you look)

Registered expectation (also in `PREREGISTRATION.md` and inside each result's `verdict`):
- **PASS** = hash matches **and** the cluster_coh gap is density-invariant (range ≤ 0.05, every gap
  > 0.15) **and** templated ASR is 0% on all 9 conditions.
- Reference shape: gap ~ +0.235…+0.247, SNR ~5.8–6.0, ASR 0%, DocFPR ~0.4–0.9% (seed-dependent).

Because it's pre-registered, you read the externals honestly: a confirming run is the expected
cross-platform replication; a non-confirming run on a **hash-matched** corpus is a real finding, not
something to massage.

---

## STEP G — Diverges vs confirms: what to do

- **Hash MISMATCH (`hash_match == false` / `FATAL`).** The corpus is not identical → the run is
  meaningless for comparison. Cause is almost always source-data or Python drift. Fix: confirm the
  person used the **pinned** `MANIFEST.json` you filled (revisions present) and **Python 3.11**;
  have them delete `a1_corpus_xplat/` and re-run (the embed/grid resume; only the corpus rebuilds).
  Do **not** fold in. This is the gate working as designed — a divergent corpus caught *before* any
  number is trusted.
- **CONFIRMS (passes Step E).** Fold it in. This is the cross-platform replication you wanted:
  identical corpus → density-invariant gap → 0% templated ASR, on a third (and fourth) platform.
- **DIVERGES on a hash-MATCHED corpus** (gap collapses, or ASR > 0, or DocFPR wild). This is a
  **genuine new problem** — same corpus, same detector, different platform, different result. It is
  a **private negative**: report it to me with the JSON, do **NOT** auto-publish or quietly fold it
  in. It would mean either a real platform-dependence in the geometry (a finding worth understanding)
  or an encoder/version mismatch (check `env` vs `MANIFEST.package_versions_5080` and
  `encoder.revision`). Investigate the cause before any claim leans on the cross-platform result.

---

## Fixed constraints (the spine — do not relax any of these)
- Identical **hash-verified** in-domain corpus across 5080 / 4060 / M4.
- **Current cluster_coh hard gate only** — never the v7.1.3 composite.
- **Frozen / non-oracle** calibration (τ from clean coh at FPR_TARGET; no poison/oracle).
- **Do not reuse the wikitext cache** — build fresh in-domain, reuse within a machine (the runner
  tags its cache `secxplat` and ignores any pre-existing `.npy`).
- **Pre-register before reading**; a **private negative is reportable, not auto-published.**
- **Chunked / resumable / stderr** embedding (a long embed can't silently die or restart from zero).
- **Turnkey paste-and-run** for two non-experts; this guide + the validity gate are yours.
