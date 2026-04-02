"""
Fast threshold sweep — pre-encode ALL queries once, then sweep in pure numpy.
No repeated model calls in the sweep.
"""
import numpy as np, json, os, faiss
import warnings; warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

CD = "seva_checkpoints_4060"
ALPHA, BETA, K = 0.6, 0.4, 5

primary = np.load(f"{CD}/phase2_primary_embs.npy")
index   = faiss.read_index(f"{CD}/phase2_faiss.index")
with open(f"{CD}/phase1_corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)
with open(f"{CD}/phase1_queries.json", encoding="utf-8") as f:
    queries = json.load(f)

is_poi = np.array([d["is_poisoned"] for d in corpus])

print("Encoding all queries once...")
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
q_texts = [q["query_text"] for q in queries]
q_embs  = enc.encode(q_texts, batch_size=32, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
q_embs  = np.ascontiguousarray(q_embs, dtype=np.float32)
del enc
print(f"Encoded {len(q_embs)} queries.")

# Pre-compute: for each query, retrieved indices + all doc A-scores
print("Pre-computing A-scores for all query-doc pairs...")
records = []
for q_idx, q in enumerate(queries):
    qe = q_embs[q_idx:q_idx+1]
    _, idxs_raw = index.search(qe, K)
    idxs = [int(x) for x in idxs_raw[0] if x >= 0]
    ret_embs = np.array([primary[i] for i in idxs], dtype=np.float32)
    cen = ret_embs.mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
    scores = []
    for i in idxs:
        geo  = float(np.linalg.norm(primary[i] - cen))
        attn = float(1.0 - float(np.dot(primary[i], qe[0])))
        scores.append(ALPHA*geo + BETA*attn)
    records.append({
        "is_adv": q["is_adversarial"],
        "idxs":   idxs,
        "scores": scores,
    })

print(f"Pre-computed {len(records)} query records. Running sweep...\n")

def evaluate(records, flag_fn):
    """
    flag_fn(scores, abs_scores) -> list of bool, one per doc in retrieved set.
    scores = relative-normalized scores, abs_scores = raw A-scores.
    """
    att_att = 0; att_suc = 0; fp_docs = 0; total_clean = 0; benign_q = 0
    for r in records:
        idxs, scores = r["idxs"], r["scores"]
        flags = flag_fn(scores)
        for i, (idx, fl) in enumerate(zip(idxs, flags)):
            poi = bool(is_poi[idx])
            if not poi: total_clean += 1
            if poi and fl: pass  # caught
            elif poi and not fl and r["is_adv"]:
                att_suc += 1    # slipped through on adversarial query
            elif not poi and fl and not r["is_adv"]:
                fp_docs += 1
        if r["is_adv"]:
            att_att += sum(1 for idx in idxs if is_poi[idx])
        else:
            benign_q += 1
    asr    = 100.0 * att_suc / att_att if att_att > 0 else 0.0
    dfpr   = 100.0 * fp_docs / total_clean if total_clean > 0 else 0.0
    return asr, dfpr

# ---- Sweep 1: Pure relative threshold (mean + N*std) with optional min_abs ----
print(f"{'N_std':<7} {'MinAbs':<8} {'ASR%':<8} {'DocFPR%':<10} Notes")
print("-"*50)
best = []
for n_std in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    for min_abs in [0.0, 0.35, 0.38, 0.40]:
        def flag_fn(scores, n=n_std, ma=min_abs):
            mu, sd = np.mean(scores), np.std(scores)
            rel = mu + n * sd
            return [s > rel and s > ma for s in scores]
        asr, dfpr = evaluate(records, flag_fn)
        note = ""
        if asr < 20 and dfpr < 15:
            note = " <-- GOOD"
            best.append((n_std, min_abs, None, None, asr, dfpr))
        elif asr < 20: note = " asr ok"
        elif dfpr < 15: note = " fpr ok"
        print(f"{n_std:<7.1f} {min_abs:<8.2f} {asr:<8.1f} {dfpr:<10.1f}{note}")

print()
print(f"{'AbsOnly':<10} {'ASR%':<8} {'DocFPR%':<10} Notes")
print("-"*45)
for abs_t in [0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56]:
    def flag_fn_abs(scores, t=abs_t):
        return [s > t for s in scores]
    asr, dfpr = evaluate(records, flag_fn_abs)
    note = " <-- GOOD" if asr < 20 and dfpr < 15 else ""
    print(f"{abs_t:<10.2f} {asr:<8.1f} {dfpr:<10.1f}{note}")

print()
# ---- Sweep 2: Hybrid (relative OR high-absolute) ----
print(f"{'N_std':<7} {'MinAbs':<8} {'AbsHigh':<9} {'ASR%':<8} {'DocFPR%':<10} Notes")
print("-"*60)
for n_std in [1.0, 1.2, 1.5]:
    for min_abs in [0.38, 0.40, 0.42]:
        for abs_high in [0.48, 0.50, 0.52, 0.55]:
            def flag_fn_h(scores, n=n_std, ma=min_abs, ah=abs_high):
                mu, sd = np.mean(scores), np.std(scores)
                rel = mu + n * sd
                return [(s > rel and s > ma) or s > ah for s in scores]
            asr, dfpr = evaluate(records, flag_fn_h)
            note = ""
            if asr < 20 and dfpr < 10:
                note = " <-- BEST"
                best.append((n_std, min_abs, abs_high, None, asr, dfpr))
            elif asr < 20 and dfpr < 15:
                note = " <-- GOOD"
                best.append((n_std, min_abs, abs_high, None, asr, dfpr))
            if asr < 25 and dfpr < 15:
                print(f"{n_std:<7.1f} {min_abs:<8.2f} {abs_high:<9.2f} {asr:<8.1f} {dfpr:<10.1f}{note}")

print()
print("=== BEST CANDIDATES ===")
best.sort(key=lambda x: x[4] + x[5])  # minimize ASR + FPR
for b in best[:10]:
    print(f"n_std={b[0]} min_abs={b[1]} abs_high={b[2]} -> ASR={b[4]:.1f}% DocFPR={b[5]:.1f}%")
