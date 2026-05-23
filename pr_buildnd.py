#!/usr/bin/env python3
"""pr_buildnd.py — PR-GATE-2 Part B step 1-2: rebuild the Security-SE clean corpus WITHOUT the lexical
prefix-dedup, report the natural near-dup rate, then embed (bge-large, chunked + resumable).

Restores the real lexical near-duplicates that build_a1_corpus.py's `x[:300].lower()` prefix-dedup had
removed — for the matched-FPR DEDUP COMPARISON ONLY (does NOT replace the main 100k deduped corpus or
any other result). Same source + same pipeline as build_a1_corpus.py EXCEPT dedup() is not applied.
Near-dup rate is measured with the IDENTICAL prefix key (no inflation/deflation). GPU embedding (no LLM
here -> no GPU contention). `python pr_buildnd.py buildonly` = build + report rate, skip embedding.
"""
import os, re, gzip, json, html, random, time, sys
import numpy as np
OUT = r"D:\SEVA-RAG\a1_corpus_nondedup"; os.makedirs(OUT, exist_ok=True)
TARGET = 100000; N_BENIGN = 3000; MIN_CHARS = 200; SEED = 42; CHUNK = 50000
BUILDONLY = len(sys.argv) > 1 and sys.argv[1] == "buildonly"
CORP = os.path.join(OUT, "clean_corpus_security_nondedup.json")
EMB = os.path.join(OUT, "pe_nondedup.npy")
def clean(s): s = html.unescape(s); s = re.sub(r"<[^>]+>", " ", s); s = re.sub(r"\s+", " ", s); return s.strip()
def load_gz(repo, fn):
    from huggingface_hub import hf_hub_download
    fp = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset"); rows = []
    with gzip.open(fp, "rt", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

if os.path.exists(CORP):
    corpus = json.load(open(CORP, encoding="utf-8")); print(f"resumed non-deduped corpus: {len(corpus)} docs")
else:
    print("loading Security-SE source (HF cache from the original build) ...")
    qrows = load_gz("flax-sentence-embeddings/stackexchange_title_body_jsonl", "security.stackexchange.com.jsonl.gz")
    arows = load_gz("flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl", "security.stackexchange.com.jsonl.gz")
    print(f"  q rows: {len(qrows)} | a rows: {len(arows)}")
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
    rng = random.Random(SEED); rng.shuffle(q_items)
    corpus_q = [d for (_, d) in q_items[N_BENIGN:]]
    pool_all = corpus_q + a_docs
    # natural near-dup rate via the IDENTICAL x[:300].lower() prefix key the build had removed
    seen = set(); ndup = 0
    for x in pool_all:
        k = x[:300].lower()
        if k in seen: ndup += 1
        else: seen.add(k)
    print(f"FULL pool: {len(pool_all)} docs | exact-prefix near-dups (would-be-removed by build dedup): {ndup} = {100*ndup/len(pool_all):.2f}% | unique: {len(seen)}")
    rng2 = random.Random(SEED); rng2.shuffle(pool_all); pool = pool_all[:TARGET]      # NON-deduped 100k (no dedup applied)
    seen2 = set(); nd2 = 0
    for x in pool:
        k = x[:300].lower()
        if k in seen2: nd2 += 1
        else: seen2.add(k)
    print(f"NON-deduped corpus ({len(pool)} docs): within-sample prefix near-dups = {nd2} = {100*nd2/len(pool):.2f}%  (vs the deduped corpus's ~0%)")
    corpus = [{"id": f"nd_{i}", "text": t, "is_poisoned": False} for i, t in enumerate(pool)]
    json.dump(corpus, open(CORP, "w", encoding="utf-8")); print(f"saved {CORP}")

if BUILDONLY:
    print("buildonly -> stop before embedding."); sys.exit(0)

# ---- embed chunked + resumable (GPU) ----
import torch
from sentence_transformers import SentenceTransformer
N = len(corpus)
if os.path.exists(EMB):
    pe = np.load(EMB, mmap_mode="r")
    if pe.shape[0] == N: print("embeddings already complete"); sys.exit(0)
PART = EMB + ".part.npy"; DONE = os.path.join(OUT, "_emb_done.txt")
emb = np.zeros((N, 1024), dtype=np.float32); start = 0
if os.path.exists(PART) and os.path.exists(DONE):
    start = int(open(DONE).read().strip()); p = np.load(PART); emb[:start] = p[:start]; print(f"resume embed from {start}/{N}")
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
t0 = time.time()
for s in range(start, N, CHUNK):
    e = min(s + CHUNK, N)
    emb[s:e] = enc.encode([corpus[i]["text"] for i in range(s, e)], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(PART, emb); open(DONE, "w").write(str(e)); print(f"  embedded {e}/{N} ({time.time()-t0:.0f}s)")
np.save(EMB, emb)
if os.path.exists(PART): os.remove(PART)
print(f"saved {EMB}  (shape {emb.shape})")
