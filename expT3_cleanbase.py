#!/usr/bin/env python3
"""expT3 -- Reproduce CleanBase (arXiv:2605.00460) and test it on BOTH injection regimes.

CleanBase's insight: documents crafted for the same target are mutually similar, so they form
CLIQUES in a similarity graph over the knowledge base. It is corpus-level and LLM-free -- the
closest peer to SEVA's cluster_coh.

Question this run answers: does CleanBase share SEVA's blind spot? If host-anchored clones evade
BOTH, the boundary is a property of the mutual-similarity assumption -- i.e. of the whole
cohesion-detection family -- not of our particular statistic.

Method (faithful to the described algorithm; hyperparameters per the paper: cosine similarity,
neighbourhood size k=10, threshold parameter z):
  edge(i,j) iff cos(i,j) > mu + z*sigma of the clean similarity distribution
  flag(d)   iff d belongs to a clique of size >= CLIQUE_MIN in that graph
z is swept so the CLEAN false-positive rate matches SEVA's 0.69% operating point -- a matched-FPR
comparison, not lifted numbers.

Runs on cached embeddings; no LLM, no re-embedding.
"""
import os, json, time, numpy as np, torch

CWD = os.path.dirname(os.path.abspath(__file__))
CK  = os.path.join(CWD, "seva_checkpoints_4060_100k_secqa_p050")
RES = os.path.join(CWD, "whitebox_attack_results")
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FPR_TARGET, P_POISON, KNN, CLIQUE_MIN = 0.0069, 5000, 10, 3
from whitebox_attack_seva import _make_poison, _PAYLOADS

t0 = time.time()
print("== load cached embeddings ==", flush=True)
pe = np.load(os.path.join(CK, "p2_pe.npy"))
clean_np = pe[P_POISON:]                      # 95k clean
pois_np  = pe[:P_POISON]                      # 5k templated poison
print("clean", clean_np.shape, "templated poison", pois_np.shape, flush=True)

# ---- host-anchored clones (the boundary attack), rebuilt from the committed paraphrases ----
pj = json.load(open(os.path.join(RES, "expD_deployed_gate_s042.json"), encoding="utf-8"))["paraphrases_full"]
clones_txt = [_make_poison(t, _PAYLOADS[i % len(_PAYLOADS)]) for i, t in enumerate(pj)]
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-large-en-v1.5", device=str(DEV))
clone_np = m.encode(clones_txt, batch_size=16, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
del m; torch.cuda.empty_cache()
print("host-anchored clones:", clone_np.shape, flush=True)

def knn_graph(pool_np, kq=KNN, chunk=2048):
    """Top-k neighbours (excluding self) + their cosine, over the whole pool. GPU chunked."""
    pool = torch.from_numpy(pool_np).to(DEV)
    N = pool.shape[0]
    idx = np.empty((N, kq), dtype=np.int32); sim = np.empty((N, kq), dtype=np.float32)
    for i in range(0, N, chunk):
        q = pool[i:i+chunk]; n = q.shape[0]
        s = q @ pool.T
        s[torch.arange(n, device=DEV), torch.arange(i, i+n, device=DEV)] = -2.0
        v, j = s.topk(kq, dim=1)
        idx[i:i+chunk] = j.cpu().numpy(); sim[i:i+chunk] = v.cpu().numpy()
    del pool; torch.cuda.empty_cache()
    return idx, sim

def clique_flags(idx, sim, tau, emb, clique_min=CLIQUE_MIN, cap=12):
    """Flag d iff d sits in a clique of size >= clique_min in the tau-thresholded kNN graph.
    Neighbourhoods are tiny at a high tau, so an exact max-clique inside each neighbourhood is fine."""
    N = idx.shape[0]
    flags = np.zeros(N, dtype=bool)
    E = torch.from_numpy(emb).to(DEV)
    for d in range(N):
        nb = idx[d][sim[d] > tau]
        if nb.size < clique_min - 1:
            continue
        nb = nb[:cap]
        sub = E[nb]
        A = ((sub @ sub.T) > tau).cpu().numpy()     # adjacency among neighbours
        np.fill_diagonal(A, False)
        need = clique_min - 1                        # d + (clique_min-1) mutually-linked neighbours
        if need == 1:
            flags[d] = True; continue
        found = False                                # search for a `need`-clique among neighbours
        order = list(range(len(nb)))
        def grow(cur, cand):
            nonlocal found
            if found: return
            if len(cur) == need: found = True; return
            for ci, c in enumerate(cand):
                if found: return
                grow(cur + [c], [x for x in cand[ci+1:] if A[c, x]])
        grow([], order)
        flags[d] = found
    del E; torch.cuda.empty_cache()
    return flags

def evaluate(attack_np, tag):
    """Build the graph over clean + attack, sweep z so CLEAN FPR == 0.69%, report catch."""
    pool_np = np.concatenate([clean_np, attack_np], axis=0)
    NC = clean_np.shape[0]
    idx, sim = knn_graph(pool_np)
    cs = sim[:NC].ravel()
    mu, sd = float(cs.mean()), float(cs.std())
    lo, hi, best = 0.0, 8.0, None
    for _ in range(14):                              # bisect z on the CLEAN false-positive rate
        z = (lo + hi) / 2
        tau = mu + z * sd
        f = clique_flags(idx, sim, tau, pool_np)
        fpr = float(f[:NC].mean())
        if fpr > FPR_TARGET: lo = z
        else: hi = z
        best = (z, tau, fpr, f)
        if abs(fpr - FPR_TARGET) < 5e-5: break
    z, tau, fpr, f = best
    catch = float(f[NC:].mean() * 100)
    print(f"  [{tag}] z={z:.3f} tau={tau:.4f} cleanFPR={fpr*100:.3f}%  CATCH={catch:.1f}%", flush=True)
    return {"tag": tag, "z": z, "tau": tau, "clean_FPR_pct": fpr * 100, "catch_pct": catch,
            "n_attack": int(attack_np.shape[0])}

print("== CleanBase on templated poison ==", flush=True)
r_templ = evaluate(pois_np, "templated")
print("== CleanBase on host-anchored clones ==", flush=True)
r_clone = evaluate(clone_np, "host-anchored clones")

out = {"method": "CleanBase reproduction (similarity graph + clique detection), matched to SEVA's "
                 "0.69% clean FPR by sweeping the z threshold parameter",
       "params": {"knn": KNN, "clique_min": CLIQUE_MIN, "FPR_TARGET_pct": FPR_TARGET * 100},
       "templated": r_templ, "host_anchored_clones": r_clone,
       "seva_reference": {"templated_catch_pct": 100.0, "clone_catch_pct": 0.0,
                          "note": "cluster_coh hard gate at the same 0.69% operating point"},
       "runtime_s": time.time() - t0}
json.dump(out, open(os.path.join(RES, "expT3_cleanbase_s042.json"), "w"), indent=2)
print("\n" + json.dumps(out, indent=2))
print("\nsaved -> whitebox_attack_results/expT3_cleanbase_s042.json")
