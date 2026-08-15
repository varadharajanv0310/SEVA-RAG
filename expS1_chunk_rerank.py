#!/usr/bin/env python3
"""expS1 -- Does the geometric gate survive a realistic chunking + reranking pipeline?

Threat-model challenge (arXiv:2606.11265): corpus poisoning is weakened once documents are
chunked and a cross-encoder reranks the retrieval window. If so, a reviewer can argue our
attack setting is unrealistic. This run rebuilds the in-domain corpus as CHUNKS, recomputes
cluster_coh on chunks, recalibrates tau non-oracle at 0.69% FPR, and measures:
  (a) does the gate still catch templated poison chunks?
  (b) does a cross-encoder reranker change what reaches the top-5?

Primary density only (p050 = 5%), which is enough to answer the question.
Fail-soft: if the reranker cannot be fetched, retrieval-only results are still reported.
"""
import os, json, time, numpy as np, torch, faiss

CWD = os.path.dirname(os.path.abspath(__file__))
CK  = os.path.join(CWD, "seva_checkpoints_4060_100k_secqa_p050")
RES = os.path.join(CWD, "whitebox_attack_results")
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FPR_TARGET, K, K_FETCH, CHUNK_W = 0.0069, 5, 20, 128
M_HNSW, EFC = 32, 200
t0 = time.time()

print("== load corpus ==", flush=True)
corpus = json.load(open(os.path.join(CK, "p1_corpus.json"), encoding="utf-8"))
gq = json.load(open(os.path.join(CK, "p1_query.json"), encoding="utf-8"))
advq = []
seen = set()
for q in gq:
    if q.get("adv") and q["q"] not in seen: advq.append(q["q"]); seen.add(q["q"])
print(f"docs={len(corpus)}  adv queries={len(advq)}", flush=True)

print(f"== chunk at {CHUNK_W} words ==", flush=True)
texts, is_pois = [], []
for d in corpus:
    w = d["text"].split(); p = bool(d["is_poisoned"])
    for i in range(0, max(1, len(w)), CHUNK_W):
        c = " ".join(w[i:i+CHUNK_W])
        if len(c.split()) >= 20:
            texts.append(c); is_pois.append(p)
is_pois = np.array(is_pois)
print(f"chunks={len(texts)}  poison chunks={int(is_pois.sum())}  ({time.time()-t0:.0f}s)", flush=True)

print("== embed chunks ==", flush=True)
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device=str(DEV))
E = enc.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
               show_progress_bar=False).astype(np.float32)
print(f"embedded {E.shape}  ({time.time()-t0:.0f}s)", flush=True)

print("== HNSW index + cluster_coh on chunks ==", flush=True)
index = faiss.IndexHNSWFlat(E.shape[1], M_HNSW, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = EFC
index.add(E)
_, nn = index.search(E, K + 1)                       # +1 -> self
Et = torch.from_numpy(E).to(DEV)
iu = torch.triu_indices(K, K, offset=1, device=DEV)
coh = np.empty(E.shape[0], dtype=np.float32)
for i in range(0, E.shape[0], 4096):
    j = min(i + 4096, E.shape[0])
    idx = nn[i:j]
    keep = np.empty((j - i, K), dtype=np.int64)      # drop self
    for r in range(j - i):
        row = [x for x in idx[r] if x != i + r][:K]
        while len(row) < K: row.append(row[-1] if row else 0)
        keep[r] = row
    nb = Et[torch.from_numpy(keep).to(DEV)]
    g = nb @ nb.transpose(1, 2)
    coh[i:j] = g[:, iu[0], iu[1]].mean(dim=1).cpu().numpy()
print(f"coh done ({time.time()-t0:.0f}s)", flush=True)

clean_coh, pois_coh = coh[~is_pois], coh[is_pois]
tau = float(np.percentile(clean_coh, 100 * (1 - FPR_TARGET)))
catch = float((pois_coh > tau).mean() * 100)
fpr = float((clean_coh > tau).mean() * 100)
print(f"  chunk-level: tau={tau:.4f}  clean coh={clean_coh.mean():.4f}  poison coh={pois_coh.mean():.4f}"
      f"  CATCH={catch:.1f}%  FPR={fpr:.3f}%", flush=True)

print("== retrieval + cross-encoder rerank on adv queries ==", flush=True)
Q = enc.encode(advq, normalize_embeddings=True, convert_to_numpy=True,
               show_progress_bar=False).astype(np.float32)
del enc; torch.cuda.empty_cache()
_, cand = index.search(Q, K_FETCH)
pre_pois = float(np.mean([is_pois[c[:K]].sum() for c in cand]))
pre_flag = float(np.mean([(coh[c[:K]] > tau).sum() for c in cand]))

rr = {"available": False}
try:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("BAAI/bge-reranker-base", device=str(DEV), max_length=512)
    post_p, post_f, moved = [], [], []
    for qi, q in enumerate(advq):
        c = cand[qi]
        s = ce.predict([[q, texts[j]] for j in c], batch_size=64, show_progress_bar=False)
        top = c[np.argsort(-np.asarray(s))[:K]]
        post_p.append(is_pois[top].sum()); post_f.append((coh[top] > tau).sum())
        moved.append(len(set(top.tolist()) ^ set(c[:K].tolist())) / 2)
    rr = {"available": True, "post_rerank_poison_in_top5": float(np.mean(post_p)),
          "post_rerank_flagged_in_top5": float(np.mean(post_f)),
          "mean_positions_changed": float(np.mean(moved))}
    print(f"  reranked: poison@5 {pre_pois:.2f} -> {rr['post_rerank_poison_in_top5']:.2f} | "
          f"flagged@5 {pre_flag:.2f} -> {rr['post_rerank_flagged_in_top5']:.2f}", flush=True)
except Exception as e:
    rr = {"available": False, "error": repr(e)[:200]}
    print(f"  reranker unavailable ({rr['error']}) -- retrieval-only results reported", flush=True)

out = {"setting": f"in-domain p050 corpus rechunked at {CHUNK_W} words, re-embedded, "
                  f"cluster_coh recomputed on chunks, tau recalibrated non-oracle at 0.69% FPR",
       "n_chunks": int(E.shape[0]), "n_poison_chunks": int(is_pois.sum()),
       "chunk_gate": {"tau": tau, "clean_coh_mean": float(clean_coh.mean()),
                      "poison_coh_mean": float(pois_coh.mean()),
                      "catch_pct": catch, "clean_FPR_pct": fpr,
                      "poison_evasion_pct": 100 - catch},
       "retrieval": {"mean_poison_in_top5": pre_pois, "mean_flagged_in_top5": pre_flag},
       "rerank": rr,
       "doc_level_reference": {"catch_pct": 100.0, "note": "unchunked p050 gate catches all templated poison"},
       "runtime_s": time.time() - t0}
json.dump(out, open(os.path.join(RES, "expS1_chunk_rerank_s042.json"), "w"), indent=2)
print("\n" + json.dumps(out, indent=2))
print("\nsaved -> whitebox_attack_results/expS1_chunk_rerank_s042.json")
