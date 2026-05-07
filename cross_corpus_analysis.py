"""
Analyze A-score distributions for different corpus sizes to find
a calibration approach that generalizes across 1k, 2k, and 50k.
"""
import numpy as np, json, os, faiss
import warnings; warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

ALPHA, BETA, K = 0.6, 0.4, 5

def analyze_corpus(checkpoint_dir, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    primary = np.load(f"{checkpoint_dir}/phase2_primary_embs.npy")
    index   = faiss.read_index(f"{checkpoint_dir}/phase2_faiss.index")
    with open(f"{checkpoint_dir}/phase1_corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(f"{checkpoint_dir}/phase1_queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    is_poi = np.array([d["is_poisoned"] for d in corpus])

    enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    targeted = [q for q in queries if q["is_adversarial"]]
    benign   = [q for q in queries if not q["is_adversarial"]]

    def centroid(v):
        c = v.mean(0); n = np.linalg.norm(c)
        return c/n if n>1e-9 else c

    def compute_scores(q_list):
        clean_a, poi_a = [], []
        for q in q_list:
            qe = enc.encode([q["query_text"]], convert_to_numpy=True, normalize_embeddings=True)
            qe32 = np.ascontiguousarray(qe, dtype=np.float32)
            _, idxs_raw = index.search(qe32, K)
            idxs = [int(x) for x in idxs_raw[0] if x >= 0]
            ret_embs = np.array([primary[i] for i in idxs], dtype=np.float32)
            cen = centroid(ret_embs)
            for i in idxs:
                de   = primary[i]
                attn = float(1.0 - float(np.dot(de, qe32[0])))
                geo  = float(np.linalg.norm(de - cen))
                a    = ALPHA*geo + BETA*attn
                if is_poi[i]: poi_a.append(a)
                else: clean_a.append(a)
        return np.array(clean_a), np.array(poi_a)

    # Benign queries → clean docs
    print(f"  Analyzing {min(40,len(benign))} benign queries...")
    c_ben, p_ben = compute_scores(benign[:40])
    # Targeted queries → poisoned docs retrieved
    print(f"  Analyzing {min(40,len(targeted))} targeted queries...")
    c_tgt, p_tgt = compute_scores(targeted[:40])

    # Combined
    all_clean = np.concatenate([c_ben, c_tgt])
    all_poi   = p_tgt

    print(f"\n  CLEAN (from all queries) n={len(all_clean)}:")
    print(f"    mean={all_clean.mean():.4f} std={all_clean.std():.4f} "
          f"P85={np.percentile(all_clean,85):.4f} P90={np.percentile(all_clean,90):.4f} "
          f"P95={np.percentile(all_clean,95):.4f}")

    print(f"\n  POISON (from targeted queries) n={len(all_poi)}:")
    print(f"    mean={all_poi.mean():.4f} std={all_poi.std():.4f} "
          f"P5={np.percentile(all_poi,5):.4f} P10={np.percentile(all_poi,10):.4f} "
          f"P25={np.percentile(all_poi,25):.4f}")

    # Gap analysis
    clean_p90 = np.percentile(all_clean, 90)
    poi_p10   = np.percentile(all_poi, 10) if len(all_poi) > 0 else 0
    midpoint  = (clean_p90 + poi_p10) / 2
    print(f"\n  Clean P90={clean_p90:.4f}  Poison P10={poi_p10:.4f}")
    print(f"  Midpoint (recommended TAU_A) = {midpoint:.4f}")

    # Evaluate midpoint
    if len(all_poi) > 0:
        for t in [midpoint - 0.02, midpoint, midpoint + 0.02]:
            fp_doc = float(np.mean(all_clean > t))
            tpr     = float(np.mean(all_poi > t))
            print(f"  t={t:.4f}: DocFPR={fp_doc*100:.1f}% PoiDetect={tpr*100:.1f}% -> "
                  f"(real ASR would be ~{(1-tpr)*100:.0f}%)")

    print(f"\n  RATIO clean_mean/poi_mean = "
          f"{all_clean.mean()/all_poi.mean():.3f}" if len(all_poi)>0 else "")
    del enc

analyze_corpus("seva_checkpoints_4060_1k", "1k Corpus")
analyze_corpus("seva_checkpoints_4060_2k", "2k Corpus")
