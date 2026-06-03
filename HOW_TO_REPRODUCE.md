# How to reproduce SEVA

Every headline number is reproducible from pinned inputs. Identity is guaranteed not by
"same script + seed" alone but by **hash-gated, deterministic** corpus and poison: the corpus
is rebuilt from pinned HuggingFace dataset revisions and verified against a canonical
**order-sensitive SHA-256** (order matters because poison replaces `corpus[0:P]`), and the
poison is regenerated deterministically and hash-checked. A runner **stops** (naming the first
divergent document) if the rebuild does not match.

## Documented hashes (verify identity against these)

| Artifact | SHA-256 |
|---|---|
| 100k in-domain corpus (order-sensitive) | `28ec38114ee64e6010ec489d01e6d3ee13d9b3758fd30a169c99ed078732f8a9` |
| Templated poison (10,000 docs, ordered) | `4f7ee3f368cc6aae82180df261f4ee60bbd1f02b0834a4c4be72615ba68a733c` |
| 1M multi-site corpus (order-sensitive) | `317eb43c337c1970c4d80e14f8eb2a9f785b75b1cbac780c620d05fe765e98f4` |

Encoder: `BAAI/bge-large-en-v1.5` @ `d4aa6901d3a41ba39fb536a557fa166f842b0e09`. Pinned dataset
revisions are in `reproduction/MANIFEST.json`.

---

## 0. Environment

```bash
conda env create -f environment.yml      # or: conda create -n seva python=3.11 -y && pip install -r requirements.txt
conda activate seva
```

The primary runs used Python 3.11.15, torch 2.11.0+cu128, numpy 1.26.4, faiss 1.7.4,
sentence-transformers 3.0.1, huggingface-hub 0.24.2. A CUDA GPU is recommended (≈ 8 GB VRAM is
enough at 100k); CPU and Apple MPS also reproduce identical detection, only slower.

---

## 1. Primary in-domain result (`tab:main`, `tab:coh`, `tab:agg`)

The canonical, hash-gated path — corpus build, deterministic poison, frozen gate, 3 densities
× 3 seeds — is a single runner:

```bash
cd reproduction
python hardgate_xrun.py --label local
```

This (a) rebuilds the 100k Security-SE corpus from the pinned revisions and **gates** it to
`28ec3811…`; (b) regenerates the exact 10k templated poison in memory and checks it against
`4f7ee3f3…`; (c) embeds once with bge-large; (d) scores the 1/5/10% × seed{42,7,123} grid under
non-oracle calibration. Output `result_local.json` should reproduce:

- **poison-evasion = 0%** on every condition; clean coherence ≈ 0.75, poison ≈ 0.99;
- gap **+0.235 → +0.247** across density (SNR ≈ 5.8–6.0), density-invariant;
- document-FPR near the 0.69% target (grand mean ≈ 0.56%).

The committed reference grid is in [`results/in_domain/`](results/in_domain/).

---

## 2. High-encounter Wilson bound (`tab:main`, `tab:percond`)

```bash
cd reproduction
python hienc_ci.py
```

A dedicated frozen-gate run over **25,000** templated encounters → **0 evasions**, tightening
the 95% Wilson upper bound to **0.0154%** (`result_hienc_ci.json`).

---

## 3. Calibration scaling — Observation 2 (`subsec:calib`)

```bash
cd reproduction
python scale_xrun.py            # emits result_scale10k.json, result_scale100k.json
python scale1m_xrun.py          # 1M point (see §5)
```

Grand-mean held-out Doc-FPR converges toward 0.69% as the clean calibration corpus grows:
**0.765% (10k) → 0.674% (100k) → 0.701% (1M)** — deviation 0.075% → 0.016% → 0.011%.

---

## 4. Encoder-invariance — Observation 3 (`tab:encoder`)

```bash
cd reproduction
python encoder_xrun.py --encoder bge   # then: --encoder e5 ; --encoder gte
```

Re-embeds the identical hash-verified corpus with three independent lineages, each in its
correct symmetric convention (bge/gte: no prefix; e5: `query:` on all texts). All three:
**0% poison-evasion**, density-invariant gap, SNR preserved (5.2–6.7). Results:
`result_encoder_{bge,e5,gte}.json`.

---

## 5. Million-document scale (`tab:scale`)

```bash
cd reproduction
python build_1m_corpus.py       # deterministic 10-site technical-SE corpus, gated to 317eb43c…
python scale1m_xrun.py
```

`N = 1,000,000`: **0% poison-evasion**, gap +0.245/+0.249 at 1%/5%, **15.0 ms** mean latency
(retrieval + gate ≈ 0.4 ms — sub-millisecond at 10× the corpus), non-oracle Doc-FPR 0.70%.
Result: `result_1M.json` (the 1M corpus is a *distinct, larger* corpus from the cross-platform
100k — different hash).

---

## 6. Cross-platform reproduction (`tab:xplat`)

Run the same turnkey runner on each machine; the corpus hash-gate guarantees a byte-identical
in-domain corpus across backends:

```bash
cd reproduction
python hardgate_xrun.py --label 5080      # RTX 5080 (CUDA)
python hardgate_xrun.py --label 4060      # RTX 4060 (CUDA)
python hardgate_xrun.py --label M4        # Apple M4 (MPS)
```

Detection is identical across CUDA and Apple-Silicon backends; the two independently
re-embedded external machines agree on the coherence gap to **< 5×10⁻⁷** (several cells
bit-identical). Committed: `result_4060.json`, `result_M4.json`. See
`reproduction/PROMPT_4060.md` / `PROMPT_M4.md` for the exact per-machine procedure and
`reproduction/PREREGISTRATION.md` for the registered PASS criteria.

---

## 7. Cross-domain (released PoisonedRAG), lexical fragility, head-to-head

These use the broader experiment scripts at the repository root (results in
`whitebox_attack_results/`):

| Claim (table) | Script | Result file |
|---|---|---|
| Cross-domain catch on released PoisonedRAG, NQ + HotpotQA (`tab:xdomain`, `tab:roc`) | `pr_xgate.py` | `whitebox_attack_results/pr_xgate_s042.json`, `pr_xgate_hotpotqa_s042.json` |
| Geometric core vs 10-signal composite (`tab:core`) | `cheap_must1.py`, `pr_gate.py` | `cheap_must1_s042.json`, `pr_gate_s042.json` |
| Composite under feature-neutralization, 49–57% (`tab:core`) | E-CAL-2 (`cheap_must1.py`) | `ecal2_s042.json` |
| Head-to-head vs RAGDefender, matched-FPR (`tab:h2h`) | `e4hh_fair.py`, `e4hh_ragdefender.py` | `e4hh_fair_s042.json`, `e4hh_s042.json` |
| Diversity-injection adaptive attack, SEVA holds 0% | `adaptive_attack_seva.py` | `adaptive_attack_results/summary.md` |

RAGDefender is reproduced in a **separate** conda environment (it is LLM-free but lives outside
this repo's frozen detector); see `RAGDEFENDER_STANDUP.md`.

---

## Mapping every table to its source

See [RESULTS.md](RESULTS.md) for the complete claim → script → result-file → hash map.
