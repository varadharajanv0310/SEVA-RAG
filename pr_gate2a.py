#!/usr/bin/env python3
"""pr_gate2a.py — PR-GATE-2 PART A: DIAGNOSTIC ROC on the existing (pre-deduped) corpus. No claim.

Reuses PR-GATE-1 PoisonedRAG poison (V=5, verified clean — NO regen) + cached 100k pre-deduped clean
embeddings. For EACH signal (cluster_coh, MinHash/s_lex, SimHash[fixed via proper Hamming sweep], s_nd)
sweeps the threshold and reports, at MATCHED DocFPR points {0.1,0.5,0.69,1,2,5}%, the per-query catch.
The claimable question: is there ANY matched-FPR point where cluster_coh catches AND every dedup misses?
INDICATOR (labelled, NOT claimable): MinHash at the standard near-dup tau=0.5. Reports DocFPR + query-FPR.
DISCIPLINE: the only claimable comparison is at MATCHED FPR (same standard we held RAGDefender to in E4-HH).
STANDALONE; frozen seva_benchmark_4060.py only IMPORTED; CPU torch; NO LLM here.
"""
import os, sys, json, zlib, time
import numpy as np, torch, importlib.util
from collections import defaultdict, Counter
from whitebox_attack_seva import CWD, CKDIR, RESULTS_DIR

DEV = torch.device("cpu"); K = 5; V = 5
CACHE = os.path.join(RESULTS_DIR, "_prgen_poison_s042.json")
FPRS = [0.1, 0.5, 0.69, 1.0, 2.0, 5.0]                 # matched DocFPR points (%)

PRIME = (1 << 31) - 1; KH = 128
_rng = np.random.default_rng(1337); MA = _rng.integers(1, PRIME, KH, dtype=np.uint64); MB = _rng.integers(0, PRIME, KH, dtype=np.uint64)
def shingles(t):
    w = t.lower().split()
    if len(w) < 5: return {np.uint64(zlib.crc32(t.lower().encode()) % PRIME)}
    return {np.uint64(zlib.crc32((" ".join(w[i:i+5])).encode()) % PRIME) for i in range(len(w) - 4)}
def minhash(sh):
    s = np.fromiter(sh, dtype=np.uint64, count=len(sh)); return ((MA[:, None]*s[None, :] + MB[:, None]) % PRIME).min(axis=1).astype(np.int64)
_SHBITS = np.arange(32, dtype=np.uint64)
def simhash(t):
    items = list(Counter(t.lower().split()).items())
    if not items: return np.uint64(0)
    hs = np.array([zlib.crc32(w.encode()) for w, _ in items], dtype=np.uint64); cs = np.array([c for _, c in items], dtype=np.int64)
    bit = ((hs[:, None] >> _SHBITS[None, :]) & np.uint64(1)).astype(np.int64); v = (cs[:, None]*(2*bit - 1)).sum(0)
    out = 0
    for b in range(32):
        if v[b] > 0: out |= (1 << b)
    return np.uint64(out)
def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555)); x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f); return ((x*np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)

# ---- load poison (V=5, dedup exact) + clean ----
pc = json.load(open(CACHE, encoding="utf-8")); pbq = {int(k): list(dict.fromkeys(v)) for k, v in pc["poison"].items()}
corpus = json.load(open(os.path.join(CKDIR, "p1_corpus.json"), encoding="utf-8")); pe = np.load(os.path.join(CKDIR, "p2_pe.npy"))
gq = json.load(open(os.path.join(CKDIR, "p1_query.json"), encoding="utf-8")); isp = np.array([bool(d["is_poisoned"]) for d in corpus])
clean_idx = np.where(~isp)[0]; clean_texts = [corpus[i]["text"] for i in clean_idx]; clean_emb = np.ascontiguousarray(pe[clean_idx]); Nc = len(clean_texts)
advq = []; seen = set()
for q in gq:
    if q.get("adv") and q["q"] not in seen: advq.append(q["q"]); seen.add(q["q"])
benq = [q["q"] for q in gq if not q.get("adv")][:2000]
qi_list = sorted(pbq); flat = [(qi, p) for qi in qi_list for p in pbq[qi][:V]]
ptexts = [p for _, p in flat]; pqi = np.array([qi for qi, _ in flat]); Pn = len(ptexts)
print(f"PART A diagnostic | V={V} | poison={Pn} | clean={Nc} (PRE-DEDUPED corpus) | benign-q={len(benq)}")

mod = "seva_prg2a"
if mod in sys.modules: del sys.modules[mod]
sv = sys.argv[:]; sys.argv = ["seva_benchmark_4060.py", "--corpus", "100000", "--poison-ratio", "0.0025", "--cal-seed", "42", "--benign-q", "2000", "--corpus-tag", "secqawbprg2a"]
try:
    spec = importlib.util.spec_from_file_location(mod, os.path.join(CWD, "seva_benchmark_4060.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
finally:
    sys.argv = sv
faiss = m.faiss; faiss.omp_set_num_threads(4); K_FETCH = m.K_FETCH

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
pe_p = enc.encode(ptexts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
Qe = enc.encode([advq[qi] for qi in qi_list], batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
Be = enc.encode(benq, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
del enc
clean_t = torch.from_numpy(clean_emb); pe_p_t = torch.from_numpy(pe_p)
print("embedded poison + adv + benign queries (CPU); reused cached clean embeddings")

# ---- corpus FAISS (poison+clean) -> doc_coh ; clean-only FAISS -> clean s_nd + benign retrieval ----
full = np.ascontiguousarray(np.concatenate([pe_p, clean_emb], 0), dtype=np.float32)
idx = faiss.IndexHNSWFlat(m.EMB_DIM, m.INDEX_M, faiss.METRIC_INNER_PRODUCT); idx.hnsw.efConstruction = m.INDEX_EF; idx.add(full)
cidx = faiss.IndexHNSWFlat(m.EMB_DIM, m.INDEX_M, faiss.METRIC_INNER_PRODUCT); cidx.hnsw.efConstruction = m.INDEX_EF; cidx.add(clean_emb)
t0 = time.time(); coh = np.zeros(Pn + Nc, dtype=np.float32)
for s0 in range(0, Pn + Nc, 1024):
    e0 = min(s0 + 1024, Pn + Nc); _, nbr = idx.search(np.ascontiguousarray(full[s0:e0]), K + 1)
    for bi in range(e0 - s0):
        di = s0 + bi; val = [int(j) for j in nbr[bi] if j >= 0 and int(j) != di][:K]
        if len(val) >= 2: em = full[val]; sm = em @ em.T; coh[di] = float(sm[np.triu_indices(len(val), k=1)].mean())
        else: coh[di] = 0.5
pcoh = coh[:Pn]; ccoh = coh[Pn:]
# clean s_nd full via clean-only FAISS top-2 (nearest other clean)
_, cn = cidx.search(clean_emb, 2); clean_snd = np.array([float(clean_emb[i] @ clean_emb[int(cn[i][1] if cn[i][0] == i else cn[i][0])]) for i in range(Nc)], dtype=np.float32)
print(f"  doc_coh + clean s_nd in {time.time()-t0:.0f}s")

# ---- signatures: clean (full) + poison ----
t0 = time.time(); clean_mh = np.empty((Nc, KH), dtype=np.int64); clean_sim = np.empty(Nc, dtype=np.uint64)
for i in range(Nc): clean_mh[i] = minhash(shingles(clean_texts[i])); clean_sim[i] = simhash(clean_texts[i])
cand = defaultdict(set)
for j in range(32):
    sub = clean_mh[:, j*4:(j+1)*4]; _, inv = np.unique(sub, axis=0, return_inverse=True); bk = defaultdict(list)
    for d, g in enumerate(inv.tolist()): bk[g].append(d)
    for docs in bk.values():
        if len(docs) < 2: continue
        if len(docs) > 800: docs = docs[:800]
        for x in docs:
            for y in docs:
                if x != y: cand[x].add(y)
clean_slex = np.zeros(Nc, dtype=np.float32)
for d in range(Nc):
    cs = cand.get(d)
    if cs: cs = list(cs); clean_slex[d] = (clean_mh[d][None, :] == clean_mh[cs]).mean(axis=1).max()
# clean SimHash min-Hamming on 25k sample (FPR thresholds; SimHash secondary)
rs = np.random.default_rng(7).choice(Nc, size=min(25000, Nc), replace=False)
clean_ham = np.empty(len(rs), dtype=np.int64)
for ii, d in enumerate(rs):
    h = popcount(np.bitwise_xor(clean_sim[d], clean_sim)); h[d] = 99; clean_ham[ii] = h.min()
print(f"  clean MinHash/SimHash/s_lex in {time.time()-t0:.0f}s")
# poison scores
p_mh = np.array([minhash(shingles(t)) for t in ptexts]); p_sim = np.array([simhash(t) for t in ptexts], dtype=np.uint64)
sib = defaultdict(list)
for i in range(Pn): sib[int(pqi[i])].append(i)
p_slex = np.zeros(Pn); p_snd = np.zeros(Pn); p_ham = np.zeros(Pn, dtype=np.int64)
for i in range(Pn):
    ss = [s for s in sib[int(pqi[i])] if s != i]
    jc = (p_mh[i][None, :] == clean_mh).mean(axis=1).max(); js = max([(p_mh[i] == p_mh[s]).mean() for s in ss], default=0.0); p_slex[i] = max(jc, js)
    cc = float((clean_t @ pe_p_t[i]).max()); c2 = max([float(pe_p_t[s] @ pe_p_t[i]) for s in ss], default=0.0); p_snd[i] = max(cc, c2)
    hc = int(popcount(np.bitwise_xor(p_sim[i], clean_sim)).min()); hs = min([int(popcount(np.array([p_sim[i] ^ p_sim[s]], dtype=np.uint64))[0]) for s in ss], default=99); p_ham[i] = min(hc, hs)

# ---- retrieval: adv top-K over corpus (poison reach) ; benign top-K over clean (query-FPR) ----
retr = {}
for qp, qi in enumerate(qi_list):
    _, ir = idx.search(np.ascontiguousarray(Qe[qp:qp+1]), K_FETCH); ri = [int(x) for x in ir[0] if x >= 0]
    sims = full[ri] @ Qe[qp]; topk = [ri[x] for x in np.argsort(-sims)[:K]]; retr[qi] = [di for di in topk if di < Pn]
reach = sum(1 for qi in qi_list if retr[qi]); print(f"  poison reach@K = {100*reach/len(qi_list):.0f}%")
_, bnn = cidx.search(Be, K)                                            # benign top-K clean doc indices

def thr_hi(clean, fpr): return float(np.percentile(clean, 100 - fpr))
def thr_lo(clean, fpr):                                                # discrete Hamming: largest t with FPR<=target
    t = -1
    while t < 32 and 100*np.mean(clean <= (t+1)) <= fpr: t += 1
    return t
def catch(ps, hi, thr):
    fl = (ps > thr) if hi else (ps <= thr); att = sum(1 for qi in qi_list if retr[qi] and not all(fl[di] for di in retr[qi]))
    return 100.0*(len(qi_list) - att)/len(qi_list)
def docfpr(cs, hi, thr): return 100.0*(np.mean(cs > thr) if hi else np.mean(cs <= thr))
def qfpr(clean_full_flag):                                             # benign query-FPR (full clean flags; None if unavailable)
    if clean_full_flag is None: return None
    return 100.0*np.mean([clean_full_flag[bnn[b]].any() for b in range(len(benq))])

SIG = [("cluster_coh", pcoh, ccoh, True, ccoh), ("MinHash/s_lex", p_slex, clean_slex, True, clean_slex),
       ("SimHash", p_ham, clean_ham, False, None), ("s_nd", p_snd, clean_snd, True, clean_snd)]
print(f"\n=== MATCHED-FPR catch (V=5, pre-deduped corpus). poison median: coh={np.median(pcoh):.3f} s_lex={np.median(p_slex):.3f} s_nd={np.median(p_snd):.3f} ham={int(np.median(p_ham))} ===")
hdr = "  DocFPR% | " + " | ".join(f"{n:>13}" for n, *_ in SIG); print(hdr)
table = {}
for fpr in FPRS:
    row = []
    for n, ps, cs, hi, cfull in SIG:
        thr = thr_hi(cs, fpr) if hi else thr_lo(cs, fpr); c = catch(ps, hi, thr); af = docfpr(cs, hi, thr)
        row.append(f"{c:5.0f}({af:.2f})"); table.setdefault(fpr, {})[n] = {"catch": c, "actual_fpr": af, "thr": float(thr)}
    print(f"  {fpr:>6.2f} | " + " | ".join(f"{r:>13}" for r in row))
print("  (cell = catch%(actual DocFPR%) at the threshold matched to the row's target FPR)")

# query-FPR at 0.69% DocFPR (full clean flags for coh/s_lex/s_nd; SimHash sampled -> deferred)
print("\n=== query-FPR (benign) at the 0.69% DocFPR threshold ===")
qf = {}
for n, ps, cs, hi, cfull in SIG:
    thr = thr_hi(cs, 0.69) if hi else thr_lo(cs, 0.69)
    q = qfpr((cfull > thr) if cfull is not None else None)
    qf[n] = q; print(f"  {n:>13}: DocFPR={docfpr(cs,hi,thr):.2f}%  query-FPR={'n/a (sampled)' if q is None else f'{q:.2f}%'}  catch={catch(ps,hi,thr):.0f}%")

# INDICATOR (NOT claimable): MinHash at standard near-dup tau=0.5
ind_c = catch(p_slex, True, 0.5); ind_f = docfpr(clean_slex, True, 0.5)
print(f"\nINDICATOR (NOT claimable): MinHash @ standard near-dup tau=0.5 -> catch {ind_c:.0f}% at DocFPR {ind_f:.2f}% (poison sibling Jaccard ~{np.median(p_slex):.2f})")

# diagnostic read
def best_dedup_catch(fpr): return max(table[fpr]["MinHash/s_lex"]["catch"], table[fpr]["SimHash"]["catch"], table[fpr]["s_nd"]["catch"])
sep = [fpr for fpr in FPRS if table[fpr]["cluster_coh"]["catch"] >= 75 and best_dedup_catch(fpr) <= 40]
if sep:
    read = f"SURPRISE: cluster_coh separates at matched DocFPR {sep}% (coh>=75%, all dedup<=40%) even on the pre-deduped corpus -> Part B may be unnecessary; re-examine."
elif all(best_dedup_catch(fpr) >= 60 for fpr in FPRS if fpr <= 1.0):
    read = "AS PREDICTED: dedup (esp. MinHash/s_nd) catches at every low matched FPR here (pre-dedup gives dedup low FPR at all thresholds) -> separation is a CORPUS PROPERTY -> Part B (realistic non-deduped corpus) NEEDED to settle the matched-FPR comparison."
else:
    read = "MIXED: dedup catch varies across matched FPR; inspect the table -> report nuance; Part B likely still needed."
print(f"\nDIAGNOSTIC READ: {read}")
json.dump({"part": "A", "V": V, "corpus": "pre-deduped 95k", "poison_n": Pn, "reach_pct": 100*reach/len(qi_list),
           "poison_medians": {"coh": float(np.median(pcoh)), "s_lex": float(np.median(p_slex)), "s_nd": float(np.median(p_snd)), "ham": int(np.median(p_ham))},
           "matched_fpr_table": {str(f): table[f] for f in FPRS}, "query_fpr_at_0.69": {k: qf[k] for k in qf},
           "minhash_tau0.5_indicator": {"catch": ind_c, "docfpr": ind_f}, "diagnostic_read": read},
          open(os.path.join(RESULTS_DIR, "pr_gate2a_s042.json"), "w", encoding="utf-8"), indent=2)
print("saved pr_gate2a_s042.json")
