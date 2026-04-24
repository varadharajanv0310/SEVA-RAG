# SEVA-RAG: Statistical Embedding Verification for Adversarial RAG Poisoning Detection

[![CI](https://github.com/varadharajanv0310/SEVA-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/varadharajanv0310/SEVA-RAG/actions/workflows/ci.yml)

SEVA detects corpus-poisoning attacks against Retrieval-Augmented Generation (RAG) systems using a
SNR-weighted composite of ten embedding-space signals anchored to a universal FPR target — no density
oracle required. See [HOW_TO_REPRODUCE.md](HOW_TO_REPRODUCE.md) for end-to-end reproduction.

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 4060 (8 GB VRAM) or equivalent CUDA-capable GPU |
| CUDA | 12.4 |
| RAM | ≥ 16 GB |
| Storage | ≥ 20 GB free (WikiText-103 + checkpoints + embeddings) |
| Python | 3.11 |

> **CPU / Apple MPS:** `seva_benchmark_4060.py` exits immediately if `torch.cuda.is_available()` returns
> `False`. CUDA is required to reproduce the paper's numbers.

---

## Repository Layout

```
SEVA-RAG/
├── seva_benchmark_4060.py      # Core benchmark (v6.2.1) — do not modify
├── adaptive_attack_seva.py     # Adaptive attack evaluator
├── generate_poison_corpus.py   # Generates poison_corpus_diverse.json (stdlib only)
│
├── run_seva_v6_smoke.py        # ← START HERE: 10k smoke test (≈ 30 min on RTX 4060)
├── run_seva_v6_100k.py         # Production 100k run (≈ 6 h on RTX 4060)
│
├── diagnose_fpr.py             # Ad-hoc FPR diagnostic utility
├── analyze_dist.py             # Score-distribution analysis utility
├── sweep_thresh.py             # Threshold sweep utility
├── generate_summary.py         # Summary generation utility
├── resume_experiments.py       # Experiment resumption utility
├── run_1pct_tier.py            # Single-tier 1% runner
├── run_50k.py                  # 50k corpus runner
├── run_benchmark.py            # Generic benchmark runner
├── run_experiments.py          # Experiment batch runner
├── run_smoke_test.py           # Minimal smoke runner
│
├── adaptive_attack_results/
│   └── summary.md              # Committed adaptive-attack results (Table VIII)
│
├── legacy/                     # Obsolete v5 scripts — not used by v6.2.1
│   └── README.md
│
├── poison_corpus_diverse.json  # 10k diverse poison docs — generate before running
│                               # (generate_poison_corpus.py; gitignored by default,
│                               #  unblocked by !poison_corpus_diverse.json in .gitignore)
│
├── requirements.txt
├── LICENSE
└── HOW_TO_REPRODUCE.md
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/varadharajanv0310/SEVA-RAG.git
cd SEVA-RAG

conda create -n seva python=3.11 -y
conda activate seva

# PyTorch with CUDA 12.4
pip install torch==2.3.1+cu124 torchvision==0.18.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# FAISS GPU (conda preferred for CUDA compatibility)
conda install -c pytorch faiss-gpu=1.7.4 cudatoolkit=12.4 -y

# Remaining dependencies
pip install -r requirements.txt
```

### 2. Generate the poison corpus

```bash
python generate_poison_corpus.py
# Outputs: poison_corpus_diverse.json  (~10 000 docs, stdlib-only, ~10 s)
```

This file must exist before running the benchmark. It is excluded from git by default but the
`.gitignore` exception (`!poison_corpus_diverse.json`) allows you to commit it once generated.

### 3. Run the smoke test (10k corpus — recommended first step)

```bash
# Edit the two path constants at the top of run_seva_v6_smoke.py:
#   PYTHON = r"<path to conda env python>"
#   BENCH  = r"<path to seva_benchmark_4060.py>"
#   CWD    = r"<repo root>"

python run_seva_v6_smoke.py
```

Expected output: **mean ASR ≤ 22%, mean FPR ≤ 3%** across all tiers and layers.
Runtime: ≈ 30 minutes on RTX 4060.

### 4. Run the full 100k production benchmark

```bash
# Same path edits in run_seva_v6_100k.py
python run_seva_v6_100k.py
```

Runtime: ≈ 6 hours on RTX 4060 (3 seeds × 3 densities × 3 layers).
Results written to `seva_checkpoints_4060_100k_p*/seva_v6_2_results_100k_p*_s*.json`.

### 5. Run the adaptive attack

```bash
python adaptive_attack_seva.py
# Results written to adaptive_attack_results/
```

---

## Key Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `FPR_TARGET` | `0.0069` | Universal FPR calibration target (0.69%) |
| `SNR_MIN_ABS` | `0.5` | Minimum \|SNR\| to include a signal in weights |
| `SNR_LOG_CAP` | `1.0` | Log-cap on per-signal weight |
| `NORM_PERCENTILE` | `90` | Percentile for corpus-derived normalization |
| `K` | `5` | Nearest-neighbor count |
| `K_FETCH` | `20` | HNSW over-fetch count (reranked to top K) |
| `BENIGN_Q` | `2000` | Benign queries (60% cal / 40% eval) |
| `TARGETED_Q` | `50` | Adversarial queries per seed per density |
| Cal/eval seeds | `[42, 7, 123]` | Three independent random seeds |

---

## Threat Model (Three-Layer)

| Layer | Excluded signals | Adversary knowledge |
|-------|-----------------|---------------------|
| L1 (Naive) | None | None |
| L2 (Standard) | `kw_density` | Keyword-density evasion |
| L3 (Compound) | `kw_density`, `avg_sent_len` | + sentence-length evasion |

---

## Committed Results

The adaptive-attack results at 5% density (100k corpus, seeds [42, 7, 123]) are committed at
[adaptive_attack_results/summary.md](adaptive_attack_results/summary.md).

Full Table IV / Table VI numerical cells require running the 100k benchmark and reading the produced
`seva_v6_2_results_100k_p*_s*.json` files. See [HOW_TO_REPRODUCE.md](HOW_TO_REPRODUCE.md) for the
exact commands and expected output ranges.

---

## Hardcoded Windows Paths

`run_seva_v6_100k.py` and `run_seva_v6_smoke.py` contain hardcoded Windows paths (`PYTHON`, `BENCH`,
`CWD`) at the top of each file. Before running on a different machine, replace these three constants
with paths valid for your environment. The benchmark script itself (`seva_benchmark_4060.py`) has no
hardcoded paths.

---

## Citation

```bibtex
@misc{seva2025,
  title   = {SEVA: Statistical Embedding Verification for Adversarial RAG Poisoning Detection},
  author  = {Varadhara Janv},
  year    = {2025},
  url     = {https://github.com/varadharajanv0310/SEVA-RAG}
}
```

---

## License

[MIT](LICENSE)
