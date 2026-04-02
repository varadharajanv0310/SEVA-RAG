"""
Diagnose the FPR=87% problem.
Key question: WHY are clean docs getting flagged in benign queries?
"""
import numpy as np, json, os, faiss, math
import warnings; warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

CD = "seva_checkpoints_4060"
TAU_A = 0.4516  # from v3 calibration
ALPHA, BETA, K = 0.6, 0.4, 5

primary   = np.load(f"{CD}/phase2_primary_embs.npy")
secondary = np.load(f"{CD}/phase2_secondary_embs.npy")
index     = faiss.read_index(f"{CD}/phase2_faiss.index")
with open(f"{CD}/phase1_corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)
with open(f"{CD}/phase1_queries.json", encoding="utf-8") as f:
    queries = json.load(f)

is_poi = np.array([d["is_poisoned"] for d in corpus])
targeted = [q for q in queries if q["is_adversarial"]]
benign   = [q for q in queries if not q["is_adversarial"]]

enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

def centroid(v):
    c = v.mean(0); n = np.linalg.norm(c)
    return c/n if n>1e-9 else c

def a_score(doc_emb, cen, qe):
    geo = float(np.linalg.norm(doc_emb - cen))
    attn = float(1.0 - float(np.dot(doc_emb, qe)))
    return ALPHA*geo + BETA*attn

def word_shanon(text):
    words = text.lower().split()
    if not words: return 0.0
    counts = {}
    for w in words: counts[w] = counts.get(w, 0) + 1
    n = len(words)
    return -sum((c/n)*math.log2(c/n) for c in counts.values())

def compression_ratio(text):
    import zlib
    raw = text.encode("utf-8")
    return len(zlib.compress(raw)) / len(raw)

def doc_len(text):
    return len(text)

print("=== BENIGN QUERY ANALYSIS (FPR=87%) ===\n")
# Track per-doc stats for benign queries
benign_data = []
for i, q in enumerate(benign[:50]):
    qe = enc.encode([q["query_text"]], convert_to_numpy=True, normalize_embeddings=True)
    qe32 = np.ascontiguousarray(qe, dtype=np.float32)
    _, idxs_raw = index.search(qe32, K)
    idxs = [int(x) for x in idxs_raw[0] if x >= 0]
    ret_embs = np.array([primary[i2] for i2 in idxs], dtype=np.float32)
    cen = centroid(ret_embs)
    for idx in idxs:
        doc = corpus[idx]
        a   = a_score(primary[idx], cen, qe32[0])
        flagged = a > TAU_A
        ent = word_shanon(doc["text"])
        compr = compression_ratio(doc["text"])
        length = doc_len(doc["text"])
        benign_data.append({
            "a": a, "flagged": flagged, "poi": is_poi[idx],
            "ent": ent, "compr": compr, "len": length
        })

clean_all  = [d for d in benign_data if not d["poi"]]
clean_flag = [d for d in clean_all if d["flagged"]]
poi_flag   = [d for d in benign_data if d["poi"] and d["flagged"]]

print(f"Benign queries, 50 queries x K={K} = {len(benign_data)} doc retrievals")
print(f"Clean docs retrieved: {len(clean_all)}")
print(f"Clean docs FLAGGED:   {len(clean_flag)} ({100*len(clean_flag)/max(len(clean_all),1):.1f}%)")
print()

# A-score distribution of flagged clean docs
flagged_a = [d["a"] for d in clean_flag]
unflagged_a = [d["a"] for d in clean_all if not d["flagged"]]
arr_f = np.array(flagged_a) if flagged_a else np.array([0.0])
arr_u = np.array(unflagged_a) if unflagged_a else np.array([0.0])
print(f"Flagged clean A-scores: mean={arr_f.mean():.4f} min={arr_f.min():.4f} max={arr_f.max():.4f}")
print(f"Pass clean A-scores:    mean={arr_u.mean():.4f}")
print()

# Can secondary features separate flagged-clean vs flagged-poison?
# Feature comparison: word entropy, compression ratio, text length
for feat, key in [("Word Shannon entropy", "ent"), ("Compression ratio", "compr"), ("Text length", "len")]:
    fc_vals = [d[key] for d in clean_flag]
    fp_vals = [d[key] for d in poi_flag]
    pu_vals = [d[key] for d in benign_data if d["poi"] and not d["flagged"]]
    cu_vals = [d[key] for d in clean_all if not d["flagged"]]
    fc = np.array(fc_vals) if fc_vals else np.array([0.0])
    fp = np.array(fp_vals) if fp_vals else np.array([0.0])
    print(f"{feat}:")
    print(f"  Flagged-clean: mean={fc.mean():.4f} std={fc.std():.4f} n={len(fc)}")
    print(f"  Flagged-poi:   mean={fp.mean():.4f} std={fp.std():.4f} n={len(fp)}")
    print(f"  Pass-clean:    mean={np.array(cu_vals).mean():.4f}  n={len(cu_vals)}")
    # Best threshold to separate flagged-clean from flagged-poi
    if len(fc) > 0 and len(fp) > 0:
        lo = min(fc.min(), fp.min()); hi = max(fc.max(), fp.max())
        bst, bsc = None, -999
        step = (hi-lo)/20 if hi>lo else 0.01
        for t in np.arange(lo, hi+step, step):
            tp = int(np.sum(fp > t)); fn = int(np.sum(fp <= t))
            fp_err = int(np.sum(fc > t)); tn = int(np.sum(fc <= t))
            if len(fp): score = tp/len(fp) - fp_err/len(fc)
            else: score = 0
            if score > bsc: bsc, bst = score, t
        if bst is not None:
            tp = int(np.sum(np.array(fp_vals) > bst))
            fpe = int(np.sum(fc > bst))
            print(f"  Best t={bst:.4f}: Poi-kept={tp}/{len(fp)}({100*tp/max(len(fp),1):.0f}%) Clean-rescued={len(fc)-fpe}/{len(fc)}({100*(len(fc)-fpe)/max(len(fc),1):.0f}%)")
    print()

print("\n=== INTRA-QUERY RELATIVE A-SCORE APPROACH ===")
print("Instead of absolute threshold, flag only docs whose A-score is")
print("an outlier RELATIVE to their retrieved set:")
all_docs_flagged = 0; false_positives = 0; true_pos = 0
for q in benign[:50]:
    qe = enc.encode([q["query_text"]], convert_to_numpy=True, normalize_embeddings=True)
    qe32 = np.ascontiguousarray(qe, dtype=np.float32)
    _, idxs_raw = index.search(qe32, K)
    idxs = [int(x) for x in idxs_raw[0] if x >= 0]
    ret_embs = np.array([primary[i2] for i2 in idxs], dtype=np.float32)
    cen = centroid(ret_embs)
    scores = [a_score(primary[idx], cen, qe32[0]) for idx in idxs]
    # Relative: flag if score > mean + 1.5*std of this batch
    mu, sd = np.mean(scores), np.std(scores)
    threshold = mu + 1.5 * sd if sd > 0 else mu + 0.02
    for idx, sc in zip(idxs, scores):
        if sc > threshold:
            all_docs_flagged += 1
            if not is_poi[idx]: false_positives += 1
print(f"With relative threshold (mean+1.5*std per query):")
print(f"  Flagged: {all_docs_flagged}, FP: {false_positives} from {50*K} docs")
print(f"  FPR (docs): {100*false_positives/(50*K):.1f}%")
