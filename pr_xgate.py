#!/usr/bin/env python3
"""pr_xgate.py — PR-XDOMAIN analysis: does cluster_coh catch RELEASED PoisonedRAG poison on NQ at matched
DocFPR, where MinHash/s_nd dedup miss? NON-oracle calibration (threshold set on NQ clean, NO poison
knowledge — the realistic "deploy on a new corpus" protocol). Loads pr_xbuild outputs. STANDALONE; frozen
seva_benchmark_4060.py only IMPORTED; CPU torch + FAISS; NO LLM. Matched FPR for ALL signals (discipline).
"""
import os, sys, json, zlib, time
import numpy as np, torch, importlib.util
from collections import defaultdict
from whitebox_attack_seva import CWD
DEV = torch.device("cpu"); K = 5; FPRS = [0.1, 0.5, 0.69, 1.0, 2.0]
DATASET = sys.argv[1] if len(sys.argv) > 1 else "nq"           # nq | hotpotqa
OUT = rf"D:\SEVA-RAG\a1_corpus_{DATASET}xd"; DSDIR = rf"D:\SEVA-RAG\poisonedrag_repo\datasets\{DATASET}"
PRIME = (1 << 31) - 1; KH = 128
_rng = np.random.default_rng(1337); MA = _rng.integers(1, PRIME, KH, dtype=np.uint64); MB = _rng.integers(0, PRIME, KH, dtype=np.uint64)
def shingles(t):
    w = t.lower().split()
    if len(w) < 5: return {np.uint64(zlib.crc32(t.lower().encode()) % PRIME)}
    return {np.uint64(zlib.crc32((" ".join(w[i:i+5])).encode()) % PRIME) for i in range(len(w) - 4)}
def minhash(sh):
    s = np.fromiter(sh, dtype=np.uint64, count=len(sh)); return ((MA[:, None]*s[None, :] + MB[:, None]) % PRIME).min(axis=1).astype(np.int64)

clean = json.load(open(os.path.join(OUT, f"{DATASET}_clean_subsample.json"), encoding="utf-8"))
poison = json.load(open(os.path.join(OUT, f"{DATASET}_poison.json"), encoding="utf-8"))
pe_c = np.load(os.path.join(OUT, "pe_clean.npy")); pe_p = np.load(os.path.join(OUT, "pe_poison.npy")); Qe = np.load(os.path.join(OUT, "pe_query.npy"))
ctexts = [c["text"] for c in clean]; ptexts = [p["text"] for p in poison]; Nc = len(clean); Pn = len(poison)
pqid = [p["qid"] for p in poison]; tqids = []
for p in poison:
    if p["qid"] not in tqids: tqids.append(p["qid"])
print(f"PR-XDOMAIN [{DATASET}] | clean {Nc} | poison {Pn} | target queries {len(tqids)}")

mod = "seva_prx"
if mod in sys.modules: del sys.modules[mod]
sv = sys.argv[:]; sys.argv = ["seva_benchmark_4060.py", "--corpus", "100000", "--poison-ratio", "0.0025", "--cal-seed", "42", "--benign-q", "2000", "--corpus-tag", "secqawbprx"]
try:
    spec = importlib.util.spec_from_file_location(mod, os.path.join(CWD, "seva_benchmark_4060.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
finally:
    sys.argv = sv
faiss = m.faiss; faiss.omp_set_num_threads(4); K_FETCH = m.K_FETCH

full = np.ascontiguousarray(np.concatenate([pe_p, pe_c], 0), dtype=np.float32)
idx = faiss.IndexHNSWFlat(m.EMB_DIM, m.INDEX_M, faiss.METRIC_INNER_PRODUCT); idx.hnsw.efConstruction = m.INDEX_EF; idx.add(full)
cidx = faiss.IndexHNSWFlat(m.EMB_DIM, m.INDEX_M, faiss.METRIC_INNER_PRODUCT); cidx.hnsw.efConstruction = m.INDEX_EF; cidx.add(pe_c)
t0 = time.time(); coh = np.zeros(Pn + Nc, dtype=np.float32)
for s0 in range(0, Pn + Nc, 1024):
    e0 = min(s0 + 1024, Pn + Nc); _, nbr = idx.search(np.ascontiguousarray(full[s0:e0]), K + 1)
    for bi in range(e0 - s0):
        di = s0 + bi; val = [int(j) for j in nbr[bi] if j >= 0 and int(j) != di][:K]
        if len(val) >= 2: em = full[val]; sm = em @ em.T; coh[di] = float(sm[np.triu_indices(len(val), k=1)].mean())
        else: coh[di] = 0.5
pcoh = coh[:Pn]; ccoh = coh[Pn:]
print(f"  doc_coh in {time.time()-t0:.0f}s | poison coh median {np.median(pcoh):.3f} | clean coh median {np.median(ccoh):.3f}  (tau_coh@0.69%FPR={np.percentile(ccoh,99.31):.3f})")

_, cn = cidx.search(pe_c, 2); clean_snd = np.array([float(pe_c[i] @ pe_c[int(cn[i][1] if cn[i][0] == i else cn[i][0])]) for i in range(Nc)], dtype=np.float32)
t0 = time.time(); clean_mh = np.empty((Nc, KH), dtype=np.int64)
for i in range(Nc): clean_mh[i] = minhash(shingles(ctexts[i]))
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
ndrate = 100*np.mean(clean_slex > 0)
print(f"  clean MinHash/s_nd in {time.time()-t0:.0f}s | {DATASET} clean s_lex>0 rate = {ndrate:.2f}% (Security-SE was 0.24%) | clean s_nd p99.31 {np.percentile(clean_snd,99.31):.3f}")
p_mh = np.array([minhash(shingles(t)) for t in ptexts]); pe_p_t = torch.from_numpy(pe_p); pe_c_t = torch.from_numpy(pe_c)
sib = defaultdict(list)
for i in range(Pn): sib[pqid[i]].append(i)
p_slex = np.zeros(Pn); p_snd = np.zeros(Pn)
for i in range(Pn):
    ss = [s for s in sib[pqid[i]] if s != i]
    jc = (p_mh[i][None, :] == clean_mh).mean(axis=1).max(); js = max([(p_mh[i] == p_mh[s]).mean() for s in ss], default=0.0); p_slex[i] = max(jc, js)
    c2 = float((pe_c_t @ pe_p_t[i]).max()); cs2 = max([float(pe_p_t[s] @ pe_p_t[i]) for s in ss], default=0.0); p_snd[i] = max(c2, cs2)

qe_by_qid = {tqids[i]: Qe[i] for i in range(len(tqids))}
retr = {}
for qid in tqids:
    qv = qe_by_qid[qid]; _, ir = idx.search(np.ascontiguousarray(qv[None, :]), K_FETCH); ri = [int(x) for x in ir[0] if x >= 0]
    sims = full[ri] @ qv; topk = [ri[x] for x in np.argsort(-sims)[:K]]; retr[qid] = [di for di in topk if di < Pn]
reach = sum(1 for qid in tqids if retr[qid]); print(f"  poison reach@K = {100*reach/len(tqids):.0f}%")

# benign query-FPR: non-target NQ queries (sample), retrieve over clean, flagged clean docs
benign_qfpr = {}
try:
    bq = []
    with open(os.path.join(DSDIR, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); qid = d.get("_id") or d.get("id")
            if qid not in set(tqids): bq.append(d.get("text", ""))
            if len(bq) >= 2000: break
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    Be = enc.encode(bq, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32); del enc
    _, bnn = cidx.search(Be, K)
    def qfpr(flag): return 100.0*np.mean([flag[bnn[b]].any() for b in range(len(bq))])
except Exception as e:
    print("  benign query-FPR skipped:", e); bnn = None; qfpr = lambda f: None

def thr_hi(c, fpr): return float(np.percentile(c, 100 - fpr))
def catch(ps, thr):
    fl = ps > thr; att = sum(1 for qid in tqids if retr[qid] and not all(fl[di] for di in retr[qid]))
    return 100.0*(len(tqids) - att)/len(tqids)
def docfpr(cs, thr): return 100.0*np.mean(cs > thr)
SIG = [("cluster_coh", pcoh, ccoh), ("MinHash/s_lex", p_slex, clean_slex), ("s_nd", p_snd, clean_snd)]
print(f"\n=== MATCHED-FPR catch ({DATASET}, non-oracle; reach {100*reach/len(tqids):.0f}%). poison median coh={np.median(pcoh):.3f} s_lex={np.median(p_slex):.3f} s_nd={np.median(p_snd):.3f} ===")
print("  DocFPR% | " + " | ".join(f"{n:>13}" for n, _, _ in SIG))
table = {}
for fpr in FPRS:
    row = []
    for n, ps, cs in SIG:
        thr = thr_hi(cs, fpr); c = catch(ps, thr); af = docfpr(cs, thr); row.append(f"{c:5.0f}({af:.2f})"); table.setdefault(fpr, {})[n] = {"catch": c, "fpr": af, "thr": float(thr)}
    print(f"  {fpr:>6.2f} | " + " | ".join(f"{r:>13}" for r in row))
print("  (cell = catch%(actual DocFPR%))")
print("\n=== query-FPR (benign NQ) at 0.69% DocFPR ===")
qf = {}
for n, ps, cs in SIG:
    thr = thr_hi(cs, 0.69); q = qfpr(cs > thr) if bnn is not None else None; qf[n] = q
    print(f"  {n:>13}: DocFPR={docfpr(cs,thr):.2f}%  query-FPR={'n/a' if q is None else f'{q:.2f}%'}  catch={catch(ps,thr):.0f}%")

coh69 = table[0.69]["cluster_coh"]["catch"]; mh69 = table[0.69]["MinHash/s_lex"]["catch"]; snd69 = table[0.69]["s_nd"]["catch"]
primary = coh69 >= 75; bonus = primary and mh69 <= 40 and snd69 <= 40
if bonus: VV = f"BONUS WIN — cluster_coh {coh69:.0f}% catches released PoisonedRAG on {DATASET} @0.69% DocFPR where MinHash {mh69:.0f}% AND s_nd {snd69:.0f}% MISS: semantic detection beats lexical/embedding dedup on a near-dup-rich corpus ({DATASET} clean s_lex>0 {ndrate:.1f}%)."
elif primary: VV = f"PRIMARY ECHO — cluster_coh {coh69:.0f}% catches released PoisonedRAG on {DATASET} @0.69% DocFPR (cross-domain confirmed). MinHash {mh69:.0f}%, s_nd {snd69:.0f}% ({DATASET} clean s_lex>0 {ndrate:.1f}%); §7.3-lexical rebuttal {'LANDS' if mh69<=40 else 'does NOT land'} (MinHash {mh69:.0f}%)."
else: VV = f"NULL — cluster_coh {coh69:.0f}% < 75% @0.69% DocFPR on {DATASET} -> cross-domain echo NOT confirmed; the NQ result stands, do not claim {DATASET}."
print(f"\nVERDICT: {VV}")
json.dump({"corpus": f"{DATASET} subsample", "Nc": Nc, "Pn": Pn, "n_queries": len(tqids), "reach_pct": 100*reach/len(tqids),
           "nq_clean_slex_gt0_pct": ndrate, "poison_medians": {"coh": float(np.median(pcoh)), "s_lex": float(np.median(p_slex)), "s_nd": float(np.median(p_snd))},
           "matched_fpr_table": {str(f): table[f] for f in FPRS}, "query_fpr_at_0.69": {k: qf[k] for k in qf},
           "coh69": coh69, "mh69": mh69, "snd69": snd69, "verdict": VV},
          open(os.path.join(r"D:\SEVA-RAG\SEVA-RAG\whitebox_attack_results", f"pr_xgate_{DATASET}_s042.json"), "w", encoding="utf-8"), indent=2)
print(f"saved pr_xgate_{DATASET}_s042.json")
