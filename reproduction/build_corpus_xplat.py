#!/usr/bin/env python3
"""
build_corpus_xplat.py -- cross-platform, pinned, hash-emitting rebuild of the EXACT
in-domain Security Stack Exchange corpus (A1). Faithful port of build_a1_corpus.py:
the SELECTION LOGIC is byte-identical (same source repos, SEED=42, filters, dedup,
shuffles) so every machine regenerates a byte-identical corpus -- which we then PROVE
with a canonical SHA-256 compared against the 5080 reference.

Differences from build_a1_corpus.py (input-side only; corpus content unchanged):
  * output dir is a CLI arg / relative (no Windows D:\\ path)
  * device auto-detected (cuda/mps/cpu) for the cohesion precheck (no hard cuda:0)
  * HF dataset revisions are PINNED (read from MANIFEST.json) and the resolved commit
    SHAs are recorded, so 'source-data version' is pinned, not just 'same script+seed'
  * emits build_provenance.json: resolved HF commits, encoder revision, counts,
    cohesion, and the canonical corpus hash (the cross-machine integrity check)

Usage:
  python build_corpus_xplat.py --out ./a1_corpus_xplat [--manifest ./MANIFEST.json] [--cpu]
"""
import os, re, gzip, json, random, html, argparse, time, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
import seva_xplat_common as C

# ---- frozen build parameters (VERBATIM from build_a1_corpus.py) ----
TARGET_CORPUS = 100000; N_BENIGN = 3000; MIN_CHARS = 200; SEED = 42; GATE = 0.80
REPO_Q = "flax-sentence-embeddings/stackexchange_title_body_jsonl"
REPO_A = "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl"
FILE_SE = "security.stackexchange.com.jsonl.gz"

def clean(s):
    s = html.unescape(s); s = re.sub(r"<[^>]+>", " ", s); s = re.sub(r"\s+", " ", s); return s.strip()

def load_gz(repo, fn, revision):
    from huggingface_hub import hf_hub_download
    fp = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset", revision=revision)
    rows = []
    with gzip.open(fp, "rt", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def resolved_commit(repo, revision):
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo, revision=revision).sha
    except Exception as e:
        return f"UNRESOLVED({e!r})"

def dedup(texts):
    seen, out = set(), []
    for x in texts:
        k = x[:300].lower()
        if k not in seen: seen.add(k); out.append(x)
    return out

def build(out_dir, rev_q, rev_a, device_str):
    C.log(f"Loading questions (rev={rev_q}) ..."); qrows = load_gz(REPO_Q, FILE_SE, rev_q); C.log(f"  q rows: {len(qrows)}")
    C.log(f"Loading answers (rev={rev_a}) ...");   arows = load_gz(REPO_A, FILE_SE, rev_a); C.log(f"  a rows: {len(arows)}")

    q_items = []
    for r in qrows:
        t = r.get("texts")
        if not isinstance(t, list) or len(t) < 2: continue
        title = clean(str(t[0])); body = clean(str(t[1])); doc = (title + "\n\n" + body).strip()
        if len(doc) >= MIN_CHARS and len(title) >= 10: q_items.append((title, doc))
    a_docs = []
    for r in arows:
        if not isinstance(r, list) or len(r) < 2: continue
        ans = clean(str(r[1]))
        if len(ans) >= MIN_CHARS: a_docs.append(ans)
    C.log(f"  question docs: {len(q_items)} | answer docs: {len(a_docs)}")

    rng = random.Random(SEED); rng.shuffle(q_items)
    benign_queries = [t for (t, _) in q_items[:N_BENIGN]]
    corpus_q       = [d for (_, d) in q_items[N_BENIGN:]]
    pool = dedup(corpus_q + a_docs); rng.shuffle(pool); pool = pool[:TARGET_CORPUS]
    corpus = [{"id": f"doc_{i}", "text": t, "is_poisoned": False} for i, t in enumerate(pool)]
    C.log(f"  FINAL corpus: {len(corpus)} docs | held-out benign queries: {len(benign_queries)}")
    if len(corpus) != TARGET_CORPUS:
        C.log(f"  *** WARNING: corpus size {len(corpus)} != {TARGET_CORPUS} -- source data differs; "
              f"the hash WILL mismatch the 5080. STOP and report to the operator.")

    # cohesion re-check on a sample (device-adaptive); same K-NN math as the detector
    samp = [d["text"] for d in corpus]; rng.shuffle(samp); samp = samp[:8000]
    C.log("Embedding 8k sample for cohesion re-check ...")
    enc, backend = C.load_encoder(device_str)
    pe = np.ascontiguousarray(enc.encode(samp, batch_size=C.BATCH, convert_to_numpy=True,
                                         normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    coh, _ = C.doc_coh_full(pe)
    mc = float(coh.mean())
    C.log(f"  sample cohesion mean={mc:.4f} median={float(np.median(coh)):.4f}  GATE<= {GATE}: "
          f"{'PASS' if mc <= GATE else 'FAIL'}  (backend={backend})")

    os.makedirs(out_dir, exist_ok=True)
    corpus_path = os.path.join(out_dir, "clean_corpus_security.json")
    benign_path = os.path.join(out_dir, "benign_queries_security.json")
    json.dump(corpus, open(corpus_path, "w", encoding="utf-8"))
    json.dump(benign_queries, open(benign_path, "w", encoding="utf-8"))

    chash, ndocs = C.sha256_corpus_canonical(corpus_path)
    prov = {
        "corpus_canonical_sha256": chash, "n_docs": ndocs, "n_benign_queries": len(benign_queries),
        "seed": SEED, "target_corpus": TARGET_CORPUS,
        "source_repos": {REPO_Q: resolved_commit(REPO_Q, rev_q), REPO_A: resolved_commit(REPO_A, rev_a)},
        "source_file": FILE_SE, "encoder": C.MODEL_ID,
        "cohesion_sample_mean": mc, "cohesion_gate": GATE, "cohesion_pass": bool(mc <= GATE),
        "precheck_backend": backend, "python": C.env_report()["python"],
        "built_unix": int(time.time()),
    }
    json.dump(prov, open(os.path.join(out_dir, "build_provenance.json"), "w"), indent=2)
    C.log(f"  SAVED corpus -> {corpus_path}")
    C.log(f"  CANONICAL CORPUS SHA-256 = {chash}   (compare to the 5080 reference!)")
    return prov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./a1_corpus_xplat")
    ap.add_argument("--manifest", default="./MANIFEST.json")
    ap.add_argument("--cpu", action="store_true", help="force CPU for the precheck encode")
    a = ap.parse_args()
    rev_q = rev_a = "main"
    if os.path.exists(a.manifest):
        man = json.load(open(a.manifest))
        pins = man.get("pinned_source_revisions", {})
        rev_q = pins.get(REPO_Q) or "main"
        rev_a = pins.get(REPO_A) or "main"
        C.set_encoder_revision(man.get("pinned_encoder_revision"))   # pin the precheck encoder too
    dev, name, backend = C.pick_device("cpu" if a.cpu else "auto")
    C.log(f"device={dev} ({name}) | source revisions: q={rev_q} a={rev_a}")
    prov = build(a.out, rev_q, rev_a, dev)
    print(json.dumps({"corpus_canonical_sha256": prov["corpus_canonical_sha256"],
                      "n_docs": prov["n_docs"], "cohesion_pass": prov["cohesion_pass"]}, indent=2))

if __name__ == "__main__":
    main()
