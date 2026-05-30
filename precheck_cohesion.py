"""
E2 / A1 clean-cohesion PRECHECK (Step 2 hard gate).

Mirrors SEVA's detector cohesion EXACTLY so the number is comparable to the
WikiText clean baseline (~0.73) and the >0.80 hard-stop gate:
  - encoder bge-large-en-v1.5, normalize_embeddings=True
  - faiss.IndexHNSWFlat(EMB_DIM=1024, M=32, METRIC_INNER_PRODUCT), efC=200, 1 thread
  - cluster_coh(d) = mean PAIRWISE cosine among d's K=5 nearest neighbours (excl self)

Does NOT touch seva_benchmark_4060.py or the detector. Read-only w.r.t. the repo.
Source: flax-sentence-embeddings/stackexchange_title_body_jsonl ::
        security.stackexchange.com.jsonl.gz  (CC-BY-SA; official SE dump reformat)
"""
import os, re, gzip, json, random, html
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# --- SEVA detector constants (mirror) ---
K = 5; INDEX_M = 32; INDEX_EF = 200; EMB_DIM = 1024; BATCH_SIZE = 32
# --- precheck params ---
SAMPLE_N      = 8000
MIN_CHARS     = 200
DEDUP_NN_COS  = 0.95      # near-duplicate question removal (plan: dedup SE near-dupes)
SEED          = 42
GATE          = 0.80
WIKITEXT_REF  = 0.73

def strip_html(s):
    s = html.unescape(s); s = re.sub(r"<[^>]+>", " ", s); s = re.sub(r"\s+", " ", s); return s.strip()

def build_index(pe):
    faiss.omp_set_num_threads(1)
    idx = faiss.IndexHNSWFlat(EMB_DIM, INDEX_M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = INDEX_EF
    idx.add(pe)
    return idx

def cohesion(pe):
    idx = build_index(pe)
    n = len(pe); coh = np.zeros(n, dtype=np.float32); B = 512
    for s in range(0, n, B):
        e = min(s + B, n)
        _, nbr = idx.search(np.ascontiguousarray(pe[s:e]), K + 1)
        for bi in range(e - s):
            di = s + bi
            valid = [int(j) for j in nbr[bi] if j >= 0 and int(j) != di][:K]
            if len(valid) >= 2:
                embs = pe[valid]; sim = np.dot(embs, embs.T)
                coh[di] = float(sim[np.triu_indices(len(valid), k=1)].mean())
            else:
                coh[di] = 0.5
    return coh

print("Downloading security.stackexchange.com.jsonl.gz ...")
fp = hf_hub_download(repo_id="flax-sentence-embeddings/stackexchange_title_body_jsonl",
                     filename="security.stackexchange.com.jsonl.gz", repo_type="dataset")
docs, seen = [], set()
with gzip.open(fp, "rt", encoding="utf-8") as f:
    for line in f:
        try: row = json.loads(line)
        except Exception: continue
        t = row.get("texts")
        if not isinstance(t, list) or len(t) < 2: continue
        doc = (strip_html(str(t[0])) + "\n\n" + strip_html(str(t[1]))).strip()
        if len(doc) < MIN_CHARS: continue
        key = doc[:300].lower()
        if key in seen: continue
        seen.add(key); docs.append(doc)
print(f"  rows after length-filter + exact-dedup: {len(docs)}")
print(f"  sample doc[0][:160]: {docs[0][:160]!r}")

rng = random.Random(SEED); rng.shuffle(docs); docs = docs[:SAMPLE_N]
print(f"  sampled: {len(docs)}")

print("Embedding (bge-large-en-v1.5, normalized) ...")
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
pe = np.ascontiguousarray(enc.encode(docs, batch_size=BATCH_SIZE, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)

coh_raw = cohesion(pe)
print(f"\n  RAW cohesion (n={len(pe)}): mean={coh_raw.mean():.4f}  median={np.median(coh_raw):.4f}  p90={np.percentile(coh_raw,90):.4f}")

# near-dedup: drop the higher-index doc of any pair with NN cosine > DEDUP_NN_COS
idx = build_index(pe)
D, Ix = idx.search(pe, 2)
keep = np.ones(len(pe), dtype=bool); dropped = 0
for i in range(len(pe)):
    j = int(Ix[i, 1])
    if j < 0 or j == i: continue
    if float(D[i, 1]) > DEDUP_NN_COS and keep[i] and keep[j] and i < j:
        keep[j] = False; dropped += 1
pe_dd = np.ascontiguousarray(pe[keep])
print(f"  near-dedup (>{DEDUP_NN_COS:.2f} cos): dropped {dropped}; unique={len(pe_dd)}")

coh_dd = cohesion(pe_dd)
mean_dd = float(coh_dd.mean())
print(f"\n  DEDUPED cohesion (n={len(pe_dd)}): mean={mean_dd:.4f}  median={np.median(coh_dd):.4f}  p90={np.percentile(coh_dd,90):.4f}")
print(f"\n  WikiText clean baseline ~{WIKITEXT_REF}    GATE: mean <= {GATE}")
verdict = "PASS" if mean_dd <= GATE else "FAIL — HARD STOP (>0.80)"
print(f"  >>> PRECHECK MEAN (deduped) = {mean_dd:.4f}  ->  {verdict}")
