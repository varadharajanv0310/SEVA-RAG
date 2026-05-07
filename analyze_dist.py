"""
Analyze the distribution of A-scores, geometric deviations, and CEV deltas
for clean vs poisoned docs, to calibrate thresholds properly.
"""
import numpy as np
import json
import os
import faiss
import warnings
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

CD = "seva_checkpoints_4060"
primary = np.load(f"{CD}/phase2_primary_embs.npy")
secondary = np.load(f"{CD}/phase2_secondary_embs.npy")
index = faiss.read_index(f"{CD}/phase2_faiss.index")

with open(f"{CD}/phase1_corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)
with open(f"{CD}/phase1_queries.json", encoding="utf-8") as f:
    queries = json.load(f)

is_poisoned = np.array([d["is_poisoned"] for d in corpus])
print(f"Corpus: {len(corpus)} docs, {int(sum(is_poisoned))} poisoned")

# Targeted (adversarial) queries first, then benign
targeted = [q for q in queries if q["is_adversarial"]]
benign   = [q for q in queries if not q["is_adversarial"]]
print(f"Queries: {len(targeted)} targeted, {len(benign)} benign")

K = 5
ALPHA = 0.6
BETA  = 0.4

enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

def centroid(vecs):
    c = vecs.mean(0)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c

data = {
    "a_clean": [], "a_poi": [],
    "geo_clean": [], "geo_poi": [],
    "attn_clean": [], "attn_poi": [],
    "cev_clean": [], "cev_poi": [],
}

# Use targeted queries so poisoned docs DO appear in retrieved set
sample_queries = targeted[:30] + benign[:20]

for q in sample_queries:
    qe = enc.encode(
        [q["query_text"]], convert_to_numpy=True, normalize_embeddings=True
    )
    qe32 = np.ascontiguousarray(qe, dtype=np.float32)
    _, idxs_raw = index.search(qe32, K)
    idxs = [int(i) for i in idxs_raw[0] if i >= 0]

    ret_embs = np.array([primary[i] for i in idxs], dtype=np.float32)
    cen = centroid(ret_embs)

    for i in idxs:
        de   = primary[i]
        cev  = float(np.linalg.norm(primary[i] - secondary[i]))
        attn = float(1.0 - float(np.dot(de, qe32[0])))
        geo  = float(np.linalg.norm(de - cen))
        a    = ALPHA * geo + BETA * attn

        tag = "poi" if is_poisoned[i] else "clean"
        data[f"a_{tag}"].append(a)
        data[f"geo_{tag}"].append(geo)
        data[f"attn_{tag}"].append(attn)
        data[f"cev_{tag}"].append(cev)

print(f"\nSamples: {len(data['a_clean'])} clean-doc retrievals, "
      f"{len(data['a_poi'])} poison-doc retrievals")
print()

n = lambda x: np.array(x, dtype=float)

for feat, label in [("a", "A_score"), ("geo", "GeoDeviation"), ("attn", "AttnVar"), ("cev", "CEV_delta")]:
    cv = n(data[f"{feat}_clean"])
    pv = n(data[f"{feat}_poi"])
    if len(cv) == 0 or len(pv) == 0:
        print(f"{label}: insufficient data (clean={len(cv)}, poi={len(pv)})")
        continue
    print(f"{label}:")
    print(f"  Clean  n={len(cv)}: mean={cv.mean():.4f} std={cv.std():.4f} "
          f"P10={np.percentile(cv,10):.4f} P50={np.percentile(cv,50):.4f} "
          f"P90={np.percentile(cv,90):.4f} P95={np.percentile(cv,95):.4f}")
    print(f"  Poison n={len(pv)}: mean={pv.mean():.4f} std={pv.std():.4f} "
          f"P10={np.percentile(pv,10):.4f} P50={np.percentile(pv,50):.4f} "
          f"P90={np.percentile(pv,90):.4f} P95={np.percentile(pv,95):.4f}")

    # Sweep thresholds (poison docs above threshold = caught)
    lo = min(cv.min(), pv.min())
    hi = max(cv.max(), pv.max())
    print(f"  Threshold sweep:")
    for t in np.arange(lo, hi, (hi-lo)/15):
        tp = int(np.sum(pv > t))
        fp = int(np.sum(cv > t))
        fpr = 100*fp/len(cv)
        tpr = 100*tp/len(pv)
        print(f"    t={t:.4f}: TPR={tpr:.0f}% FPR={fpr:.1f}%")
    print()

# Also check: what fraction of poisoned docs retrieved per adversarial query?
print("=== Retrieved poisoned doc counts per adversarial query ===")
poi_counts = []
for q in targeted[:10]:
    qe = enc.encode([q["query_text"]], convert_to_numpy=True, normalize_embeddings=True)
    qe32 = np.ascontiguousarray(qe, dtype=np.float32)
    _, idxs_raw = index.search(qe32, K)
    idxs = [int(i) for i in idxs_raw[0] if i >= 0]
    poi_in_result = sum(1 for i in idxs if is_poisoned[i])
    poi_counts.append(poi_in_result)
    print(f"  Query '{q['query_text'][:50]}...' → {poi_in_result}/{K} poisoned")
print(f"  Mean poisoned docs per targeted query: {np.mean(poi_counts):.2f}")
