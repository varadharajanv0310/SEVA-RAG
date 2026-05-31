#!/usr/bin/env python3
"""
encoder_config.py -- encoder registry + correct-by-construction embedding for the
SEVA encoder-generalization experiment.

WHY THIS FILE EXISTS
--------------------
The SEVA detector (cluster_coh hard gate) is FROZEN. The ONLY thing the experiment
varies is the embedding ENCODER. The single largest threat to a valid result is a
FALSE NEGATIVE caused by feeding a new encoder text in the wrong format (missing or
wrong instruction prefix), which silently degrades its embeddings and understates the
method. This module pins, per encoder, the *correct* usage so each encoder gets a
FAIR test. Read the notes -- the conventions differ and they matter.

THE KEY SUBTLETY (do not get this wrong)
----------------------------------------
cluster_coh is a SYMMETRIC document<->document similarity / local-clustering signal
(a doc's mean cosine to its K nearest CORPUS neighbours). So each encoder must be used
in its SYMMETRIC / clustering convention, NOT its asymmetric query->passage retrieval
convention:

  * bge-large-en-v1.5 (BAAI): symmetric similarity uses NO instruction prefix. (The
    "Represent this sentence ..." instruction is for the QUERY side of ASYMMETRIC
    retrieval only.)  -> no prefix.  [matches the frozen bge pipeline already in the paper]

  * e5-large-v2 (intfloat): the model card is explicit -- for SYMMETRIC tasks
    (semantic similarity, clustering, embeddings-as-features) use the "query: " prefix
    for ALL texts; "passage: " is ONLY for the passage side of asymmetric retrieval.
    cluster_coh is symmetric/clustering -> use "query: " on EVERY text (docs AND
    queries). Using "passage: " on the docs, or no prefix, is the canonical e5 misuse
    and would understate the result.  -> "query: " for all.

  * gte-large (thenlper / Alibaba): uses NO instruction prefix for any task.  -> none.

DIM: all three are 1024-d, so the frozen FAISS HNSW params (M=32, efC=200) and
K_FETCH=20 transfer unchanged. The runner asserts the produced dim == DIM.

POOLING: handled automatically by sentence-transformers from each model's own config
(bge=CLS, e5=mean, gte=mean) -- do not override it.
"""
import os
import sys
import numpy as np

# Embedding batch. The detector result is invariant to this (deterministic encoder);
# kept at 32 to match the frozen pipeline and minimise any fp drift vs the bge baseline.
BATCH = 32

ENCODERS = {
    # ---- baseline / harness-correctness encoder (already in the paper) ----
    "bge": {
        "model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "doc_prefix": "",          # symmetric similarity: no instruction
        "query_prefix": "",
        "normalize": True,
        "note": "Baseline. Must reproduce the known bge 100k numbers (result_scale100k.json) "
                "before any new encoder is trusted -- this validates the harness independently "
                "of whether the new encoder looks good.",
    },
    # ---- THE test encoder: different lineage (Microsoft, weakly-supervised contrastive) ----
    "e5": {
        "model": "intfloat/e5-large-v2",
        "dim": 1024,
        "doc_prefix": "query: ",   # symmetric-task convention per the e5 model card: "query: " for ALL texts
        "query_prefix": "query: ",
        "normalize": True,
        "note": "Symmetric/clustering convention -> 'query: ' on EVERY text (docs AND queries), "
                "NOT 'passage: '. cluster_coh is a symmetric doc-doc similarity signal.",
    },
    # ---- follow-up encoder (Alibaba lineage): only if e5 PASSES ----
    "gte": {
        "model": "thenlper/gte-large",
        "dim": 1024,
        "doc_prefix": "",          # gte uses no instruction prefix for any task
        "query_prefix": "",
        "normalize": True,
        "note": "Follow-up toward an 'encoder-invariant' claim (3rd lineage). Run ONLY after e5 PASSES.",
    },
}


def resolve_revision(key):
    """Resolve the current HF commit SHA, to pin the encoder in the result JSON."""
    from huggingface_hub import HfApi
    return HfApi().model_info(ENCODERS[key]["model"]).sha


def load_model(key, device):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ENCODERS[key]["model"], device=device)


def _prefix(texts, role, key):
    cfg = ENCODERS[key]
    p = cfg["doc_prefix"] if role == "doc" else cfg["query_prefix"]
    return list(texts) if not p else [p + t for t in texts]


def embed_enc(key, texts, role, tag, model, cache_root, device, log=sys.stderr):
    """
    Chunked, resumable, stderr-logged embedding with the encoder's CORRECT prefix +
    L2 normalisation (so dot == cosine, matching the frozen detector's geometry).
      key   : encoder key in ENCODERS
      role  : "doc" or "query" -> selects the prefix (both are "query: " for e5)
      tag   : filename tag ("clean"/"poison"/"tq"/"bquery") -- cache is per-encoder, per-tag,
              so one encoder's embeddings NEVER reuse another's.
    Returns float32 [N, DIM], L2-normalised.
    """
    cfg = ENCODERS[key]
    n = len(texts)
    os.makedirs(cache_root, exist_ok=True)
    done_path = os.path.join(cache_root, f"emb_{tag}.npy")
    if os.path.exists(done_path):
        arr = np.load(done_path)
        if arr.shape == (n, cfg["dim"]):
            print(f"[embed_enc] {key}/{tag}: reuse cache {arr.shape}", file=log, flush=True)
            return arr.astype(np.float32)
    pref = _prefix(texts, role, key)
    out = np.zeros((n, cfg["dim"]), dtype=np.float32)
    CHUNK = 2000
    for i in range(0, n, CHUNK):
        part_path = os.path.join(cache_root, f"emb_{tag}_part_{i}.npy")
        if os.path.exists(part_path):
            part = np.load(part_path)
        else:
            part = model.encode(pref[i:i + CHUNK], batch_size=BATCH,
                                normalize_embeddings=cfg["normalize"],
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            np.save(part_path, part)
        out[i:i + part.shape[0]] = part
        print(f"[embed_enc] {key}/{tag}: {min(i + CHUNK, n)}/{n}", file=log, flush=True)
    assert out.shape[1] == cfg["dim"], f"DIM mismatch: produced {out.shape[1]} != configured {cfg['dim']}"
    np.save(done_path, out)
    return out


def sanity_report(key, model, device, sample_docs, sample_queries):
    """
    Coarse guards against a misconfigured encoder (the false-negative direction), run on
    a tiny sample BEFORE the long 100k embed:
      - dim matches the config,
      - embeddings are L2-normalised (norm ~ 1),
      - the prefix is actually applied (the literal input string is shown for eyeball),
      - embeddings are not degenerate/random (in-domain texts cohere clearly above chance).
    The strong correctness proof is the separate bge harness-reproduction gate.
    """
    cfg = ENCODERS[key]
    dpref = _prefix(sample_docs[:8], "doc", key)
    qpref = _prefix(sample_queries[:8], "query", key)
    d = model.encode(dpref, batch_size=BATCH, normalize_embeddings=cfg["normalize"],
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    q = model.encode(qpref, batch_size=BATCH, normalize_embeddings=cfg["normalize"],
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    norms = np.linalg.norm(d, axis=1)
    sims = q @ d.T
    return {
        "dim": int(d.shape[1]),
        "dim_ok": bool(d.shape[1] == cfg["dim"]),
        "emb_norm_mean": float(norms.mean()),
        "normalized_ok": bool(abs(float(norms.mean()) - 1.0) < 1e-2),
        "sample_doc_input": (cfg["doc_prefix"] + sample_docs[0])[:140],
        "sample_query_input": (cfg["query_prefix"] + sample_queries[0])[:140],
        "retrieval_max_cos_mean": float(sims.max(axis=1).mean()),
        "retrieval_sane": bool(float(sims.max(axis=1).mean()) > 0.2),  # random 1024-d ~ 0; >0.2 = not degenerate
    }
