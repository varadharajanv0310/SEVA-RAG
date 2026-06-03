#!/usr/bin/env python3
"""
seva_xplat_common.py -- shared cross-platform helpers for the SEVA cross-machine
hard-gate handoff (5080 reference / 4060 / Apple M4).

Imported by build_corpus_xplat.py and hardgate_xrun.py. Contains:
  * frozen constants (must match seva_benchmark_4060.py byte-for-byte)
  * the FROZEN cluster_coh (doc_coh) FAISS K-NN math -- copied VERBATIM from the
    frozen detector (seva_benchmark_4060._compute_doc_coh / cheap_must1.doc_coh_full /
    build_a1_corpus). DO NOT EDIT -- cross-platform validity depends on it being identical.
  * device auto-detection (cuda -> mps -> cpu) with reporting
  * environment + pre-existing-cache inspection
  * SHA-256 corpus hashing (the corpus-integrity check)
  * chunked / resumable / stderr-logged embedding (so a long embed can't silently die)

Nothing here is platform-specific except pick_device(). FAISS is CPU on every
platform; only the SentenceTransformer encoder uses the GPU/MPS device.
"""
import os, sys, json, time, hashlib, platform, traceback
import numpy as np

# ============================ FROZEN CONSTANTS ============================
# These MUST equal the frozen detector's constants. Source: seva_benchmark_4060.py
# (K=5, K_FETCH=20, FPR_TARGET=0.0069, EMB_DIM=1024, INDEX_M=32, efC=200, BATCH=32).
MODEL_ID      = "BAAI/bge-large-en-v1.5"
K             = 5            # cluster_coh neighbourhood
K_FETCH       = 20          # retrieval over-fetch before rerank-to-top-K
EMB_DIM       = 1024
INDEX_M       = 32          # HNSW M
INDEX_EF      = 200         # HNSW efConstruction
FPR_TARGET    = 0.0069      # 0.69% -- the only calibration knob; non-oracle
BATCH         = 32          # encoder batch size (matches build_a1_corpus.py)
FAISS_THREADS = 1           # determinism: build_a1_corpus used omp_set_num_threads(1)

# experiment grid
DENSITIES = [0.01, 0.05, 0.10]     # 1% / 5% / 10%
SEEDS     = [42, 7, 123]
TARGETED_Q = 50                    # adversarial queries used for retrieval-based ASR
BENIGN_Q   = 2000                  # benign queries sampled per seed for DocFPR

def log(msg):
    """stderr-logged, flushed progress (survives a killed stdout / long runs)."""
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n"); sys.stderr.flush()

# ============================ DEVICE ============================
def pick_device(prefer="auto"):
    """Return (device_str, human_name, backend). cuda -> mps -> cpu. prefer='cpu' forces CPU."""
    import torch
    if prefer == "cpu":
        return "cpu", "CPU (forced)", "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0), "cuda"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", "Apple Silicon (MPS)", "mps"
    except Exception:
        pass
    return "cpu", "CPU (no CUDA/MPS available)", "cpu"

_ENC_REV = None
def set_encoder_revision(rev):
    """Pin the encoder commit (from MANIFEST.pinned_encoder_revision). All subsequent
    load_encoder() calls then fetch EXACTLY that revision -> identical weights across machines."""
    global _ENC_REV
    _ENC_REV = rev or None

def load_encoder(device_str):
    """Load bge-large (PINNED revision if set) on the chosen device; fall back to CPU if the
    device init fails. Returns (encoder, effective_backend)."""
    from sentence_transformers import SentenceTransformer
    kw = {"device": device_str}
    if _ENC_REV:
        kw["revision"] = _ENC_REV
    try:
        return SentenceTransformer(MODEL_ID, **kw), device_str
    except Exception as e:
        log(f"WARNING: encoder failed on device '{device_str}' ({e!r}); falling back to CPU.")
        kw["device"] = "cpu"
        return SentenceTransformer(MODEL_ID, **kw), "cpu"

# ============================ ENV / CACHE REPORT ============================
def env_report():
    import torch
    dev, name, backend = pick_device()
    rep = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", "?"),
        "detected_device": dev, "device_name": name, "backend": backend,
    }
    try:
        import sentence_transformers as st; rep["sentence_transformers"] = st.__version__
    except Exception: rep["sentence_transformers"] = "?"
    try:
        import faiss; rep["faiss"] = getattr(faiss, "__version__", "present")
    except Exception: rep["faiss"] = "MISSING"
    try:
        import huggingface_hub as hh; rep["huggingface_hub"] = hh.__version__
    except Exception: rep["huggingface_hub"] = "?"
    return rep

def inspect_preexisting_cache(search_dirs, max_depth=3, max_files=20000):
    """Report what the machine's PRE-EXISTING caches contain (expected: v7.1.3 wikitext
    embeddings = WRONG corpus, do NOT use). Read-only; never deletes. BOUNDED walk
    (depth + file cap, skips hidden/our-own dirs) so it can never hang on a huge home tree."""
    found = []; seen = 0
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        base_depth = os.path.abspath(d).rstrip(os.sep).count(os.sep)
        for root, dirs, files in os.walk(d):
            depth = os.path.abspath(root).rstrip(os.sep).count(os.sep) - base_depth
            if depth >= max_depth:
                dirs[:] = []  # don't descend further
            # prune hidden dirs, our fresh cache, and known-huge trees
            dirs[:] = [x for x in dirs if not x.startswith(".")
                       and "secxplat" not in x
                       and x not in ("node_modules", "site-packages", "__pycache__", ".git", "huggingface")]
            for fn in files:
                seen += 1
                if seen > max_files:
                    found.append({"note": f"walk capped at {max_files} files"}); return _cache_note(found)
                if fn.endswith(".npy") and ("pe" in fn or "emb" in fn or "p2" in fn):
                    fp = os.path.join(root, fn)
                    try:
                        arr = np.load(fp, mmap_mode="r")
                        low = fp.lower()
                        if "secxplat" in low: tag = "fresh-in-domain (this run)"
                        elif "secqa" in low: tag = "in-domain-tagged (prior secqa run)"
                        else: tag = "WIKITEXT?(v7.1.3 -- IGNORED by this run)"
                        found.append({"path": fp, "shape": list(arr.shape),
                                      "dtype": str(arr.dtype), "guess": tag})
                    except Exception as e:
                        found.append({"path": fp, "error": repr(e)})
    return _cache_note(found)

def _cache_note(found):
    return {"note": "Pre-existing caches below are the machine's OLD state. The v7.1.3 100k "
                    "cache is WIKITEXT embeddings (wrong corpus) -- this run IGNORES them and "
                    "builds a FRESH in-domain cache tagged 'secxplat'.",
            "preexisting": found}

# ============================ HASHING (corpus integrity) ============================
def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_corpus_canonical(corpus_json_path):
    """ORDER-SENSITIVE content hash of the CLEAN corpus: SHA-256 over the per-doc
    SHA-256(text) taken in CORPUS ORDER (doc_0, doc_1, ..., doc_99999), prefixed by the
    count. Captures the exact document SET **and their order**.

    Order is load-bearing: the poison replaces corpus[0:P], so a reordered (but same-set)
    corpus removes DIFFERENT clean docs and yields a different clean subset -> the comparison
    would be invalid. This hash flags ANY set, order, or content divergence in the clean
    corpus. It hashes only d['text'] (not raw JSON bytes), so it is robust to JSON
    whitespace / key-order differences across machines but sensitive to every text change."""
    docs = json.load(open(corpus_json_path, encoding="utf-8"))
    h = hashlib.sha256(); h.update(str(len(docs)).encode()); h.update(b"\n")
    for d in docs:
        h.update(hashlib.sha256(d["text"].encode("utf-8")).hexdigest().encode()); h.update(b"\n")
    return h.hexdigest(), len(docs)

def verify_against_fingerprint(corpus_json_path, fingerprint_path):
    """Doc-by-doc compare of the rebuilt clean corpus to the shipped ordered fingerprint
    (corpus_fingerprint.txt: one SHA-256(text) hex per line, in canonical corpus order).
    Returns the match flag AND the FIRST divergent document index -- so a mismatch is
    localized to an exact doc, turning an opaque 'hashes differ' into 'doc N differs'."""
    docs = json.load(open(corpus_json_path, encoding="utf-8"))
    ref = [ln.strip() for ln in open(fingerprint_path, encoding="utf-8") if ln.strip()]
    first = None
    for i in range(min(len(docs), len(ref))):
        if hashlib.sha256(docs[i]["text"].encode("utf-8")).hexdigest() != ref[i]:
            first = i; break
    ok = (len(docs) == len(ref)) and (first is None)
    return {"ok": bool(ok), "n_corpus": len(docs), "n_fingerprint": len(ref),
            "first_divergent_index": first,
            "divergent_text_preview": (docs[first]["text"][:160] if first is not None else None)}

def sha256_ordered_texts(texts):
    """ORDER-preserving content hash of a text list. Used for the regenerated poison
    (poison doc i = pool[i], so order matters). Identical generator -> identical hash."""
    h = hashlib.sha256(); h.update(str(len(texts)).encode()); h.update(b"\n")
    for t in texts:
        h.update(hashlib.sha256(t.encode("utf-8")).hexdigest().encode()); h.update(b"\n")
    return h.hexdigest()

# ============================ RESUMABLE EMBEDDING ============================
def embed_resumable(texts, out_npy, device_str, chunk=2000, label="embed"):
    """Encode `texts` with bge-large in chunks, persisting each chunk to disk so a
    killed/crashed run RESUMES instead of restarting. Returns (embeddings, effective_backend).

      * chunk dir:  <out_npy>.parts/   (chunk_00000.npy ...) + done.json manifest
      * on resume:  completed chunks are loaded; embedding continues from the first gap
      * assembled:  when all chunks exist, concatenated -> out_npy and parts kept (idempotent)
      * stderr:     per-chunk progress via log()
    """
    if os.path.exists(out_npy):
        log(f"{label}: found assembled {out_npy} -- reusing (no re-embed).")
        return np.load(out_npy), "cached"
    n = len(texts); parts = out_npy + ".parts"; os.makedirs(parts, exist_ok=True)
    n_chunks = (n + chunk - 1) // chunk
    done_path = os.path.join(parts, "done.json")
    done = set(json.load(open(done_path)) ) if os.path.exists(done_path) else set()
    enc, backend = load_encoder(device_str)
    log(f"{label}: {n} texts in {n_chunks} chunks of {chunk} on backend={backend} "
        f"(resuming: {len(done)}/{n_chunks} chunks already done)")
    for ci in range(n_chunks):
        cp = os.path.join(parts, f"chunk_{ci:05d}.npy")
        if ci in done and os.path.exists(cp):
            continue
        s0, e0 = ci * chunk, min((ci + 1) * chunk, n)
        t0 = time.perf_counter()
        emb = enc.encode(texts[s0:e0], batch_size=BATCH, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        np.save(cp, np.ascontiguousarray(emb, dtype=np.float32))
        done.add(ci); json.dump(sorted(done), open(done_path, "w"))
        log(f"{label}: chunk {ci+1}/{n_chunks} [{s0}:{e0}] done in {time.perf_counter()-t0:.1f}s")
    # assemble
    log(f"{label}: assembling {n_chunks} chunks -> {out_npy}")
    full = np.concatenate([np.load(os.path.join(parts, f"chunk_{ci:05d}.npy"))
                           for ci in range(n_chunks)], axis=0).astype(np.float32)
    assert full.shape[0] == n, f"assembled {full.shape[0]} != {n}"
    np.save(out_npy, full)
    return full, backend

# ============================ FROZEN cluster_coh + retrieval ============================
# VERBATIM from the frozen detector. cosine == inner product on L2-normalized embeddings.
def _faiss_index(emb):
    import faiss
    faiss.omp_set_num_threads(FAISS_THREADS)
    idx = faiss.IndexHNSWFlat(EMB_DIM, INDEX_M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = INDEX_EF
    idx.add(np.ascontiguousarray(emb, dtype=np.float32))
    return idx

def doc_coh_full(full):
    """Per-document cluster_coh = mean pairwise cosine among the K nearest CORPUS
    neighbours (excluding self). 0.5 if <2 valid neighbours. FROZEN math."""
    idx = _faiss_index(full)
    coh = np.zeros(len(full), dtype=np.float32)
    for s0 in range(0, len(full), 1024):
        e0 = min(s0 + 1024, len(full))
        _, nbr = idx.search(np.ascontiguousarray(full[s0:e0]), K + 1)
        for bi in range(e0 - s0):
            di = s0 + bi
            val = [int(j) for j in nbr[bi] if j >= 0 and int(j) != di][:K]
            if len(val) >= 2:
                em = full[val]; sm = em @ em.T
                coh[di] = float(sm[np.triu_indices(len(val), k=1)].mean())
            else:
                coh[di] = 0.5
    return coh, idx

def retrieve_topk(idx, full, Q):
    """Over-fetch K_FETCH, rerank by exact cosine, return top-K corpus indices per query.
    FROZEN retrieval (matches cheap_must1.retrieve)."""
    out = []
    for qp in range(len(Q)):
        _, ir = idx.search(np.ascontiguousarray(Q[qp:qp + 1]), K_FETCH)
        ri = [int(x) for x in ir[0] if x >= 0]
        if not ri:
            out.append([]); continue
        sims = full[ri] @ Q[qp]
        topk = [ri[x] for x in np.argsort(-sims)[:K]]
        out.append(topk)
    return out
