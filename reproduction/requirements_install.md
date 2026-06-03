# requirements_install.md — Python env for the cross-platform run

Both external machines already ran the v7.1.3 build, so they almost certainly **already have a
working Python env** with the needed packages and the `bge-large-en-v1.5` model cached. **Prefer
reusing that env.** Only install if something is missing or a major version is off.

## What the run needs (import names → pip package)
- `numpy`  (numpy)            — pin **1.26.x** (matches the 5080 reference; avoids 2.x dtype quirks)
- `torch`  (torch)           — any stable **2.x** for the platform:
    - **4060 / CUDA:** the machine's existing CUDA build of torch (do not reinstall if it works).
    - **Apple M4:** the standard macOS arm64 wheel (`pip install torch`) — it includes the **MPS** backend.
- `sentence_transformers` (sentence-transformers) — pin **3.0.1**
- `faiss`  (faiss-cpu)       — pin **1.7.4** (CPU build on every platform; we never use faiss-gpu)
- `huggingface_hub` (huggingface_hub) — recent (≥0.23)

## Reuse-or-install (the agent runs this)
```
python -c "import numpy,torch,sentence_transformers,faiss,huggingface_hub as h; \
print('numpy',numpy.__version__,'torch',torch.__version__,'st',sentence_transformers.__version__, \
'hf',h.__version__); import torch as t; print('cuda',t.cuda.is_available(),'mps',getattr(t.backends,'mps',None) and t.backends.mps.is_available())"
```
- If that prints versions and a device (cuda **or** mps True) → **you're done, use this env.**
- If an import fails, install only the missing piece, e.g.:
  `pip install "numpy==1.26.4" "sentence-transformers==3.0.1" "faiss-cpu==1.7.4" "huggingface_hub>=0.23"`
  (Do **not** reinstall a working torch — especially the 4060's CUDA build.)

## Determinism pins that matter
- **Python 3.11** — the corpus build uses Python's Mersenne-Twister `random.shuffle`; staying on 3.11
  (same as the 5080) keeps the selection identical. (3.10/3.12 are very likely identical too, but the
  canonical SHA-256 gate is the real guarantee — if it matches, the corpus is byte-identical regardless.)
- **Encoder revision** is pinned in `MANIFEST.json → pinned_encoder_revision`; the model weights are
  identical across machines, so embeddings differ only by hardware float noise (which the cluster_coh
  gap is robust to). FAISS + the gate run on CPU on every platform.
- No internet is needed at detection time; the only downloads are the (pinned) HF datasets for the
  corpus build and the (cached) encoder.
