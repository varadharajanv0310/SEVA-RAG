"""
Build the A1 in-domain clean corpus (E2, Limitation-2 fix) from Security Stack Exchange.

Outputs (to D:\\SEVA-RAG\\a1_corpus\\ , OUTSIDE the git repo — large data stays out of git):
  - clean_corpus_security.json   : ~100k docs {"id","text","is_poisoned":False}
                                    = security questions (title+body) + best-answer bodies
  - benign_queries_security.json : held-out security question TITLES (CF-008: independent
                                    benign queries NOT in the indexed corpus)

Also re-checks the FINAL combined-corpus clean cohesion against the <=0.80 hard gate
(answers were not in the first precheck). Mirrors the detector exactly. Detector untouched.
Sources: flax-sentence-embeddings/{stackexchange_title_body_jsonl, stackexchange_title_best_voted_answer_jsonl}
         :: security.stackexchange.com.jsonl.gz   (CC-BY-SA).
"""
import os, re, gzip, json, random, html
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

OUT = r"D:\SEVA-RAG\a1_corpus"; os.makedirs(OUT, exist_ok=True)
TARGET_CORPUS = 100000; N_BENIGN = 3000; MIN_CHARS = 200; SEED = 42; GATE = 0.80
K = 5; INDEX_M = 32; INDEX_EF = 200; EMB_DIM = 1024; BATCH = 32

def clean(s):
    s = html.unescape(s); s = re.sub(r"<[^>]+>", " ", s); s = re.sub(r"\s+", " ", s); return s.strip()

def load_gz(repo, fn):
    fp = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset")
    rows = []
    with gzip.open(fp, "rt", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

print("Loading questions ..."); qrows = load_gz("flax-sentence-embeddings/stackexchange_title_body_jsonl", "security.stackexchange.com.jsonl.gz")
print("  q rows:", len(qrows))
print("Loading answers ...");   arows = load_gz("flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl", "security.stackexchange.com.jsonl.gz")
print("  a rows:", len(arows))

q_items = []  # (title, doc_text)
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
print(f"  question docs: {len(q_items)} | answer docs: {len(a_docs)}")

def dedup(texts):
    seen, out = set(), []
    for x in texts:
        k = x[:300].lower()
        if k not in seen: seen.add(k); out.append(x)
    return out

rng = random.Random(SEED); rng.shuffle(q_items)
benign_queries = [t for (t, _) in q_items[:N_BENIGN]]           # held-out question titles
corpus_q       = [d for (_, d) in q_items[N_BENIGN:]]
pool = dedup(corpus_q + a_docs); rng.shuffle(pool); pool = pool[:TARGET_CORPUS]
corpus = [{"id": f"doc_{i}", "text": t, "is_poisoned": False} for i, t in enumerate(pool)]
print(f"  FINAL corpus: {len(corpus)} docs | held-out benign queries: {len(benign_queries)}")

# --- cohesion re-check on FINAL corpus sample (hard gate) ---
samp = [d["text"] for d in corpus]; rng.shuffle(samp); samp = samp[:8000]
print("Embedding sample for cohesion re-check ...")
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
pe = np.ascontiguousarray(enc.encode(samp, batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
faiss.omp_set_num_threads(1)
idx = faiss.IndexHNSWFlat(EMB_DIM, INDEX_M, faiss.METRIC_INNER_PRODUCT); idx.hnsw.efConstruction = INDEX_EF; idx.add(pe)
coh = np.zeros(len(pe), dtype=np.float32)
for s in range(0, len(pe), 512):
    e = min(s + 512, len(pe)); _, nbr = idx.search(np.ascontiguousarray(pe[s:e]), K + 1)
    for bi in range(e - s):
        di = s + bi; valid = [int(j) for j in nbr[bi] if j >= 0 and int(j) != di][:K]
        if len(valid) >= 2:
            em = pe[valid]; sim = np.dot(em, em.T); coh[di] = float(sim[np.triu_indices(len(valid), k=1)].mean())
        else: coh[di] = 0.5
mc = float(coh.mean())
print(f"\n  FINAL-corpus sample cohesion: mean={mc:.4f}  median={np.median(coh):.4f}  p90={np.percentile(coh,90):.4f}")
print(f"  GATE: mean <= {GATE}  ->  {'PASS' if mc <= GATE else 'FAIL — HARD STOP (>0.80)'}")

if mc <= GATE:
    json.dump(corpus, open(os.path.join(OUT, "clean_corpus_security.json"), "w", encoding="utf-8"))
    json.dump(benign_queries, open(os.path.join(OUT, "benign_queries_security.json"), "w", encoding="utf-8"))
    print(f"  SAVED -> {OUT}\\clean_corpus_security.json  (+ benign_queries_security.json)")
else:
    print("  NOT SAVED — gate failed; stop and report.")
