#!/usr/bin/env python3
"""s_lex_gate.py — CHEAP GATE for a LEXICAL copy-detection signal (s_lex) to catch clone-inject.

STANDALONE — does NOT touch the frozen seva_benchmark_4060.py. Why lexical (after s_nd/ND-GATE-1 RED):
clone-inject LITERALLY copies a host doc, so word-5-gram Jaccard(clone,host) ~0.85 REGARDLESS of the
embedding displacement that sank s_nd; a *semantic* near-dup shares meaning but DIFFERENT WORDS ->
Jaccard ~0.1-0.2. So lexical overlap sidesteps both s_nd failure modes. MinHash/shingles = LLM-free,
CPU, <30 ms -> keeps SEVA's identity.

STEP 1 (detection): s_lex = max word-5-gram Jaccard(clone, any CLEAN doc) (MinHash); + exact Jaccard(clone,host).
STEP 2 (FPR, decisive): clean s_lex = max Jaccard(clean_i, nearest OTHER clean) via MinHash-LSH. ROC.
2D view: for high-s_lex clean docs, their EMBEDDING cosine to that lexical twin (verbatim repost = hi cos;
clone = hi Jaccard but cos-displaced ~0.93). Reuses cached p050 corpus TEXT + embeddings; numpy MinHash (no deps).
"""
import os, json, zlib, numpy as np, torch
from collections import defaultdict
from whitebox_attack_seva import _make_poison, _PAYLOADS  # identical clone construction -> provenance

CWD = os.path.dirname(os.path.abspath(__file__)); CK = os.path.join(CWD, "seva_checkpoints_4060_100k_secqa_p050")
RES = os.path.join(CWD, "whitebox_attack_results"); DEV = torch.device("cuda:0")
P = (1 << 31) - 1; K = 128; B, R = 32, 4   # MinHash perms; LSH bands x rows (32*4=128)
rng = np.random.default_rng(1337); A = rng.integers(1, P, K, dtype=np.uint64); Bv = rng.integers(0, P, K, dtype=np.uint64)

def shingles(text):
    w = text.lower().split()
    if len(w) < 5:
        return {np.uint64(zlib.crc32(text.lower().encode()) % P)}
    return {np.uint64(zlib.crc32((" ".join(w[i:i+5])).encode()) % P) for i in range(len(w) - 4)}

def minhash(sh):
    s = np.fromiter(sh, dtype=np.uint64, count=len(sh))
    return ((A[:, None] * s[None, :] + Bv[:, None]) % P).min(axis=1).astype(np.int64)

corpus = json.load(open(os.path.join(CK, "p1_corpus.json"), encoding="utf-8"))
pe = np.load(os.path.join(CK, "p2_pe.npy")); isp = np.array([bool(d["is_poisoned"]) for d in corpus])
clean_idx = np.where(~isp)[0]; clean_texts = [corpus[i]["text"] for i in clean_idx]
clean_emb = np.ascontiguousarray(pe[clean_idx]); Nc = len(clean_texts)
print(f"clean corpus: {Nc} docs (cached p050 text+embeddings; numpy MinHash, no deps)")

# ---- MinHash signatures for all clean docs ----
import time; t0 = time.time()
clean_sh = [shingles(t) for t in clean_texts]
clean_sig = np.empty((Nc, K), dtype=np.int64)
for i in range(Nc):
    clean_sig[i] = minhash(clean_sh[i])
print(f"  minhashed {Nc} clean docs in {time.time()-t0:.1f}s")

# ---- STEP 1: clone s_lex (regenerate the SAME 1-rep clones; host = top-1 by cosine) ----
gq = json.load(open(os.path.join(CK, "p1_query.json"), encoding="utf-8"))
advq = []; seen = set()
for q in gq:
    if q.get("adv") and q["q"] not in seen: advq.append(q["q"]); seen.add(q["q"])
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
Q = torch.from_numpy(enc.encode(advq, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)).to(DEV)
clean_t = torch.from_numpy(clean_emb).to(DEV)
hostidx = torch.topk(Q @ clean_t.T, 1, dim=1).indices.squeeze(1).cpu().numpy()
del enc; torch.cuda.empty_cache()
clone_texts = [_make_poison(clean_texts[int(hostidx[qi])], _PAYLOADS[qi % len(_PAYLOADS)]) for qi in range(len(advq))]
clone_sig = np.array([minhash(shingles(t)) for t in clone_texts])
s_lex_clone = np.array([(clone_sig[i][None, :] == clean_sig).mean(axis=1).max() for i in range(len(clone_texts))])
# exact Jaccard(clone, its host)
jacc_host = []
for qi in range(len(clone_texts)):
    a = shingles(clone_texts[qi]); b = clean_sh[int(hostidx[qi])]
    jacc_host.append(len(a & b) / max(len(a | b), 1))
jacc_host = np.array(jacc_host)
print(f"STEP 1  clone s_lex (MinHash max-Jaccard to clean): min/median/max = {s_lex_clone.min():.3f}/{np.median(s_lex_clone):.3f}/{s_lex_clone.max():.3f}")
print(f"        exact Jaccard(clone, its host):             min/median/max = {jacc_host.min():.3f}/{np.median(jacc_host):.3f}/{jacc_host.max():.3f}")

# ---- STEP 2: clean s_lex via MinHash-LSH (max Jaccard to nearest OTHER clean) ----
cand = defaultdict(set); capped = 0
for j in range(B):
    sub = clean_sig[:, j*R:(j+1)*R]
    _, inv = np.unique(sub, axis=0, return_inverse=True)
    buckets = defaultdict(list)
    for d, g in enumerate(inv.tolist()): buckets[g].append(d)
    for docs in buckets.values():
        if len(docs) < 2: continue
        if len(docs) > 800: capped += 1; docs = docs[:800]
        for x in docs:
            for y in docs:
                if x != y: cand[x].add(y)
s_lex_clean = np.zeros(Nc, dtype=np.float32); twin = -np.ones(Nc, dtype=np.int64)
for d in range(Nc):
    cs = cand.get(d)
    if not cs: continue
    cs = list(cs); js = (clean_sig[d][None, :] == clean_sig[cs]).mean(axis=1)
    k = int(js.argmax()); s_lex_clean[d] = js[k]; twin[d] = cs[k]
pct = {p: float(np.percentile(s_lex_clean, p)) for p in [50, 90, 99, 99.9, 100]}
print(f"STEP 2  clean s_lex ({Nc} docs; LSH B={B} R={R}; {capped} buckets capped): pct50/90/99/99.9/max = "
      f"{pct[50]:.3f}/{pct[90]:.3f}/{pct[99]:.3f}/{pct[99.9]:.3f}/{pct[100]:.3f}")

# ---- ROC + headline ----
print(f"\n  {'tau_lex':>7} | {'clone-catch':>11} | {'clean-FPR':>10}")
roc = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
    cc = 100.0 * np.mean(s_lex_clone > t); fp = 100.0 * np.mean(s_lex_clean > t)
    print(f"  {t:>7.2f} | {cc:>10.1f}% | {fp:>9.3f}%"); roc.append({"tau": t, "clone_catch": cc, "clean_fpr": fp})
TARGET = 0.69
cand_t = [r for r in roc if r["clone_catch"] >= 95.0]
hl = min(cand_t, key=lambda r: r["tau"]) if cand_t else None
if hl:
    v = "GREEN" if hl["clean_fpr"] <= TARGET else ("AMBER" if hl["clean_fpr"] <= 3 * TARGET else "RED")
    print(f"\nHEADLINE: lowest tau_lex catching >=95% clones = {hl['tau']:.2f} -> clean FPR = {hl['clean_fpr']:.3f}% vs {TARGET}% target -> {v}")
else:
    v = "RED"; print(f"\nHEADLINE: no tau in [0.3,0.85] catches >=95% clones -> RED")

# ---- 2D view: for high-s_lex clean docs, embedding cosine to their lexical twin ----
hi = np.where(s_lex_clean > 0.5)[0]
emb2d = None
if len(hi):
    cos = np.array([float(clean_emb[d] @ clean_emb[twin[d]]) for d in hi])
    emb2d = {"n_hi_slex": int(len(hi)), "emb_cos_to_lexical_twin": {"min": float(cos.min()), "median": float(np.median(cos)), "max": float(cos.max()),
             "frac_cos_ge_0.97": float(np.mean(cos >= 0.97))}}
    print(f"\n2D VIEW: {len(hi)} clean docs with s_lex>0.5 -> embedding cos to lexical twin: "
          f"median {np.median(cos):.3f}, {100*np.mean(cos>=0.97):.0f}% at cos>=0.97 (verbatim reposts); clones sit at Jaccard~0.85 + cos~0.93")
json.dump({"Nc": Nc, "n_clones": len(clone_texts), "verdict": v,
           "s_lex_clone": {"min": float(s_lex_clone.min()), "median": float(np.median(s_lex_clone)), "max": float(s_lex_clone.max())},
           "jaccard_clone_host": {"min": float(jacc_host.min()), "median": float(np.median(jacc_host)), "max": float(jacc_host.max())},
           "s_lex_clean_pct": pct, "roc": roc, "headline": hl, "target_fpr": TARGET, "emb_2d": emb2d, "lsh": {"B": B, "R": R, "capped_buckets": capped}},
          open(os.path.join(RES, "s_lex_gate_s042.json"), "w"), indent=2)
print("saved s_lex_gate_s042.json")
