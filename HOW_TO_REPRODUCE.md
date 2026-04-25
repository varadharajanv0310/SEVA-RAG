# SEVA-RAG — End-to-End Reproduction Guide

This document walks an independent reviewer from a fresh clone to the paper's Table IV / Table VI /
Table VIII numerical cells. Follow the steps in order; each phase caches its output so a failed run
can be resumed from the last completed phase.

---

## Prerequisites

- NVIDIA GPU with ≥ 8 GB VRAM (paper: RTX 4060 Laptop GPU, CUDA 12.4)
- Python 3.11, conda (or venv)
- ≥ 20 GB free disk space
- Internet access for first run (downloads WikiText-103 and BGE-large-en-v1.5 from HuggingFace)

---

## Step 0 — Clone and install

```bash
git clone https://github.com/varadharajanv0310/SEVA-RAG.git
cd SEVA-RAG

# Create and activate environment
conda create -n seva python=3.11 -y
conda activate seva

# PyTorch + CUDA 12.4
pip install torch==2.3.1+cu124 torchvision==0.18.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# FAISS GPU (conda is the most reliable source for GPU builds)
conda install -c pytorch faiss-gpu=1.7.4 cudatoolkit=12.4 -y

# Remaining packages
pip install -r requirements.txt
```

Verify CUDA is visible:

```bash
python -c "import torch, faiss; print(torch.cuda.get_device_name(0)); print(faiss.get_num_gpus())"
# Expected: your GPU name, then "1"
```

---

## Step 1 — Generate the poison corpus

`generate_poison_corpus.py` uses only the Python standard library — no GPU needed.

```bash
python generate_poison_corpus.py
# Output: poison_corpus_diverse.json  (≈ 10 000 docs, < 10 s)
```

Verify:

```bash
python -c "import json; d=json.load(open('poison_corpus_diverse.json')); print(len(d), 'docs')"
# Expected: 10000 docs
```

---

## Step 2 — Edit driver path constants (one-time)

`run_seva_v6_smoke.py` and `run_seva_v6_100k.py` each have three path constants at the top that
must be set to values valid on your machine:

```python
PYTHON = r"/path/to/conda/envs/seva/bin/python"  # or full path to python.exe on Windows
BENCH  = r"/path/to/SEVA-RAG/seva_benchmark_4060.py"
CWD    = r"/path/to/SEVA-RAG"
```

On Linux/macOS:
```bash
which python   # inside the seva conda env
```

On Windows:
```
where python   # inside the seva conda env
```

---

## Step 3 — Smoke test (10k corpus, ≈ 30 min)

Runs 3 densities × 3 seeds × 3 layers at 10k corpus to verify the pipeline end-to-end before
committing 6 hours to the full run.

```bash
python run_seva_v6_smoke.py
```

**Pass criteria** (all must hold):
- Mean ASR ≤ 22% across all tiers and layers
- Mean FPR ≤ 3% across all tiers and layers
- No tier reports `"skipped": True` (indicates Phase 3 calibration failure)

Checkpoints are written to `seva_checkpoints_4060_10k_p*/`.

---

## Step 4 — Full 100k production run (≈ 6 h)

```bash
python run_seva_v6_100k.py
```

This runs:
- Corpus sizes: 100 000
- Poison ratios: 1%, 5%, 10%
- Cal/eval seeds: 42, 7, 123
- Three-layer threat model: L1, L2, L3

Result JSONs are written to:
```
seva_checkpoints_4060_100k_p001/seva_v6_2_results_100k_p001_s042.json
seva_checkpoints_4060_100k_p001/seva_v6_2_results_100k_p001_s007.json
seva_checkpoints_4060_100k_p001/seva_v6_2_results_100k_p001_s123.json
seva_checkpoints_4060_100k_p050/seva_v6_2_results_100k_p050_s*.json
seva_checkpoints_4060_100k_p100/seva_v6_2_results_100k_p100_s*.json
```

Each JSON contains `L1`, `L2`, `L3` sub-objects with `asr`, `fpr`, `tau`, `weights`, `signal_stats`,
and `counts` fields corresponding to Table IV, Table VI, and Table II of the paper.

---

## Step 5 — Adaptive attack (≈ 2 h at 5% density)

```bash
python adaptive_attack_seva.py
```

Requires the 100k 5% checkpoints from Step 4 to exist
(`seva_checkpoints_4060_100k_p050_adaptive/`).

Results are written to `adaptive_attack_results/` (Table VIII of the paper). The committed
[adaptive_attack_results/summary.md](adaptive_attack_results/summary.md) shows the expected output.

---

## Reading committed Table IV results

The nine production result files are committed to the `results/` directory:

```
results/seva_v6_2_results_100k_p010_s{007,042,123}.json   ← 1% density
results/seva_v6_2_results_100k_p050_s{007,042,123}.json   ← 5% density
results/seva_v6_2_results_100k_p100_s{007,042,123}.json   ← 10% density
```

Read them with:

```python
import json, glob, numpy as np

results = {}
for path in sorted(glob.glob("results/seva_v6_2_results_100k_*.json")):
    d = json.load(open(path))
    ratio_pct = int(d["poison_ratio"] * 100)
    tag = f"{ratio_pct}% s{d['cal_seed']}"
    results[tag] = {layer: {"asr": d[layer]["asr"], "fpr": d[layer]["doc_fpr"],
                             "tau": d[layer]["tau"],
                             "lat_mean": d[layer]["latency"]["mean"],
                             "lat_p95":  d[layer]["latency"]["p95"]}
                    for layer in ("L1", "L2", "L3")}

for tag, v in sorted(results.items()):
    print(tag, {l: f"ASR={v[l]['asr']:.2f}% FPR={v[l]['fpr']:.3f}%" for l in v})
```

---

## Phase cache structure (for resume/debugging)

| File | Phase | Contents |
|------|-------|----------|
| `p1_corpus.json` | 1 | Background corpus doc ids + text |
| `p1_query.json` | 1 | Targeted + benign queries |
| `p2_pe.npy` | 2 | BGE-large corpus embeddings |
| `p2_doc_coh.npy` | 2 | Precomputed per-doc cluster_coh |
| `p3_v6.2_s{seed}.json` | 3 | τ, SNR weights, signal stats, cal_doc_ids |
| `seva_v6_2_results_*.json` | 4 | ASR, FPR, latencies, confusion matrix |

Delete any of these files to force recomputation from that phase onward.
Use `--reset` flag to clear all caches for a given density:

```bash
python seva_benchmark_4060.py --multitier --mtcorpus 100000 --reset
```

---

## Seed semantics

The three reported seeds `[42, 7, 123]` control the **60/40 cal/eval partition** of the 2000 benign
queries (via `_split_queries` in `seva_benchmark_4060.py`). All three seeds share the same 2000
benign queries; they differ only in which 1200 are used for Phase 3 calibration vs which 800 are
held out for Phase 4 evaluation. Mean ± std across seeds therefore quantifies sensitivity of τ to
the calibration partition, not full experimental independence.

---

## Expected output ranges (100k, paper Table IV)

| Condition | ASR (mean ± std) | FPR (mean ± std) |
|-----------|-----------------|-----------------|
| L1, 1% | 0.00 ± 0.00% | 0.73 ± 0.17% |
| L1, 5% | 0.00 ± 0.00% | 0.63 ± 0.07% |
| L1, 10% | 0.00 ± 0.00% | 0.95 ± 0.34% |
| L2, 1% | 0.00 ± 0.00% | 0.74 ± 0.08% |
| L2, 5% | 0.53 ± 0.92% | 0.62 ± 0.05% |
| L2, 10% | 0.00 ± 0.00% | 0.95 ± 0.34% |
| L3, 1% | 1.07 ± 0.92% | 0.72 ± 0.18% |
| L3, 5% | 16.27 ± 2.31% | 0.67 ± 0.19% |
| L3, 10% | 17.07 ± 3.23% | 0.89 ± 0.17% |

Latency on RTX 4060 (full pipeline, Phase 4 per-query): **43 ms mean, ≈ 53 ms p95**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ERROR: CUDA not available` | No CUDA GPU / wrong PyTorch build | Reinstall torch with `+cu124` index |
| `ModuleNotFoundError: faiss` | FAISS not installed | Use `conda install -c pytorch faiss-gpu` |
| Phase 3 `STOP` — no positive-SNR signal | Corpus too small or wrong poison file | Verify `poison_corpus_diverse.json` exists and has 10k docs |
| Result JSON not found after run | `.gitignore` blocks `*.json` | The `!poison_corpus_diverse.json` exception in `.gitignore` only unblocks that file; result JSONs are still excluded and must be read locally |
| Run hangs on HuggingFace download | No internet / proxy | Set `HF_DATASETS_OFFLINE=1` after first download |
