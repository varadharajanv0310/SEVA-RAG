#!/usr/bin/env python3
"""cheap_must1.py — CHEAP-MUST-1 (Security-SE; reuse caches + existing poison; no regen, no LLM).

PART A — cluster_coh HARD GATE (flag if cluster_coh > tau_coh; tau set NON-oracle at 0.69% DocFPR on
CLEAN; NOT the SNR-weighted composite) on three cases @ ~0.69% DocFPR:
  (1) templated poison (E-CAL-1/half_B), (2) black-box PoisonedRAG (PR-GATE-1 poison),
  (3) L2/L3-adaptive (E-CAL-2 keyword-drop) -- for a coh-ONLY gate this is IDENTICAL to (1): E-CAL-2's
      c1 neutralization swaps the kw_density/avg_sent_len FEATURE VALUES fed to the soft scorer, never
      the text/embedding, so cluster_coh is unchanged. Reported as measured (= case 1).
  Report ASR + DocFPR + query-FPR (ACTUALS).
PART B — per-query aggregation: flag query if >=1 vs >=2 retrieved docs flagged. benign query-FPR +
  poison catch for both, same DocFPR, on benign queries + the PoisonedRAG poison (>=5/query).
GUARDRAIL: tau / thresholds set by FPR on CLEAN only (no tune-to-test). Standalone; frozen detector
untouched; GPU embed (no LLM running). doc_coh via the frozen FAISS K-NN math (imported, not modified).
"""
import os, sys, json, time
import numpy as np, torch, importlib.util
from whitebox_attack_seva import CWD, CKDIR, RESULTS_DIR
DEV = torch.device("cuda:0"); K = 5; FPRP = 0.69

corpus = json.load(open(os.path.join(CKDIR, "p1_corpus.json"), encoding="utf-8")); pe = np.load(os.path.join(CKDIR, "p2_pe.npy"))
gq = json.load(open(os.path.join(CKDIR, "p1_query.json"), encoding="utf-8")); isp = np.array([bool(d["is_poisoned"]) for d in corpus])
clean_idx = np.where(~isp)[0]; clean_emb = np.ascontiguousarray(pe[clean_idx]); Nc = len(clean_idx)
advq = []; seen = set()
for q in gq:
    if q.get("adv") and q["q"] not in seen: advq.append(q["q"]); seen.add(q["q"])
benq = [q["q"] for q in gq if not q.get("adv")][:2000]
half_B = [d["text"] for d in json.load(open(os.path.join(CWD, "poison_corpus_diverse.json"), encoding="utf-8"))][1::2][:2500]
prg = json.load(open(os.path.join(RESULTS_DIR, "_prgen_poison_s042.json"), encoding="utf-8"))["poison"]
prg = {int(k): list(dict.fromkeys(v)) for k, v in prg.items()}
prg_flat = [(qi, p) for qi in sorted(prg) for p in prg[qi][:5]]; prg_texts = [p for _, p in prg_flat]; prg_qi = np.array([qi for qi, _ in prg_flat])
print(f"clean {Nc} | adv-q {len(advq)} | benign-q {len(benq)} | templated half_B {len(half_B)} | PoisonedRAG poison {len(prg_texts)}")

mod = "seva_cm1"
if mod in sys.modules: del sys.modules[mod]
sv = sys.argv[:]; sys.argv = ["seva_benchmark_4060.py", "--corpus", "100000", "--poison-ratio", "0.0025", "--cal-seed", "42", "--benign-q", "2000", "--corpus-tag", "secqawbcm1"]
try:
    spec = importlib.util.spec_from_file_location(mod, os.path.join(CWD, "seva_benchmark_4060.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
finally:
    sys.argv = sv
faiss = m.faiss; faiss.omp_set_num_threads(4); K_FETCH = m.K_FETCH

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
def emb(texts): return enc.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
Qe = emb(advq); Be = emb(benq); tB = emb(half_B); tP = emb(prg_texts)
del enc; torch.cuda.empty_cache()
print("embedded queries + both poison sets (GPU); reused cached clean embeddings")

def doc_coh_full(full):
    idx = faiss.IndexHNSWFlat(m.EMB_DIM, m.INDEX_M, faiss.METRIC_INNER_PRODUCT); idx.hnsw.efConstruction = m.INDEX_EF; idx.add(full)
    coh = np.zeros(len(full), dtype=np.float32)
    for s0 in range(0, len(full), 1024):
        e0 = min(s0 + 1024, len(full)); _, nbr = idx.search(np.ascontiguousarray(full[s0:e0]), K + 1)
        for bi in range(e0 - s0):
            di = s0 + bi; val = [int(j) for j in nbr[bi] if j >= 0 and int(j) != di][:K]
            if len(val) >= 2: em = full[val]; sm = em @ em.T; coh[di] = float(sm[np.triu_indices(len(val), k=1)].mean())
            else: coh[di] = 0.5
    return coh, idx

def retrieve(idx, full, Q, P):
    out = []
    for qp in range(len(Q)):
        _, ir = idx.search(np.ascontiguousarray(Q[qp:qp+1]), K_FETCH); ri = [int(x) for x in ir[0] if x >= 0]
        sims = full[ri] @ Q[qp]; topk = [ri[x] for x in np.argsort(-sims)[:K]]; out.append(topk)
    return out

def case(name, pemb, pqi):
    P = len(pemb); full = np.ascontiguousarray(np.concatenate([pemb, clean_emb], 0), dtype=np.float32)
    coh, idx = doc_coh_full(full); pcoh = coh[:P]; ccoh = coh[P:]
    tau = float(np.percentile(ccoh, 100 - FPRP))                    # NON-oracle: tau set on CLEAN only
    docfpr = 100.0 * np.mean(ccoh > tau)
    adv_topk = retrieve(idx, full, Qe, P)
    # poison reach + per-query ASR (>=1 retrieved-unflagged poison). poison flagged if pcoh>tau.
    att = reach = 0
    for tk in adv_topk:
        rp = [di for di in tk if di < P]
        if rp: reach += 1
        if rp and any(pcoh[di] <= tau for di in rp): att += 1
    asr = 100.0 * att / len(advq); catch = 100.0 - asr
    # query-FPR (benign): >=1 retrieved doc flagged (poison or clean) over the full corpus
    ben_topk = retrieve(idx, full, Be, P)
    def flagged(di): return (pcoh[di] > tau) if di < P else (ccoh[di - P] > tau)
    qfpr1 = 100.0 * np.mean([any(flagged(di) for di in tk) for tk in ben_topk])
    print(f"  [{name}] tau_coh={tau:.4f} DocFPR={docfpr:.2f}% | poison coh med={np.median(pcoh):.3f} reach={100*reach/len(advq):.0f}% | catch={catch:.1f}% ASR={asr:.1f}% | benign query-FPR(>=1)={qfpr1:.2f}%")
    return {"name": name, "tau": tau, "docfpr": docfpr, "poison_coh_median": float(np.median(pcoh)), "reach_pct": 100*reach/len(advq),
            "catch": catch, "asr": asr, "query_fpr_ge1": qfpr1, "_pcoh": pcoh, "_ccoh": ccoh, "_tau": tau, "_advtopk": adv_topk, "_bentopk": ben_topk, "_P": P, "_pqi": pqi}

print("\n=== PART A — cluster_coh HARD GATE (non-oracle tau @0.69% DocFPR on clean) ===")
cT = case("templated", tB, None)
cP = case("PoisonedRAG", tP, prg_qi)
print(f"  [L2/L3-adaptive] = templated by construction (coh-only gate is invariant to E-CAL-2 feature-value neutralization) -> ASR {cT['asr']:.1f}%, catch {cT['catch']:.1f}% (vs the SOFT composite's ~49-57% collapse)")

print("\n=== PART B — per-query aggregation (PoisonedRAG corpus, same tau_coh) ===")
pcoh = cP["_pcoh"]; ccoh = cP["_ccoh"]; tau = cP["_tau"]; P = cP["_P"]; adv_topk = cP["_advtopk"]; ben_topk = cP["_bentopk"]; pqi = cP["_pqi"]
def fl(di): return (pcoh[di] > tau) if di < P else (ccoh[di - P] > tau)
for kth in [1, 2]:
    # benign query-FPR: >=kth retrieved docs flagged
    qf = 100.0 * np.mean([sum(fl(di) for di in tk) >= kth for tk in ben_topk])
    # poison catch: target query caught if >=kth retrieved POISON flagged
    cc = 100.0 * np.mean([sum(1 for di in tk if di < P and pcoh[di] > tau) >= kth for tk in adv_topk])
    # also report mean #poison retrieved/query (so >=2 is feasible)
    print(f"  >= {kth} flagged: benign query-FPR={qf:.2f}%  | poison catch (>= {kth} poison flagged)={cc:.1f}%")
npr = np.mean([sum(1 for di in tk if di < P) for tk in adv_topk]); nflag = np.mean([sum(1 for di in tk if di < P and pcoh[di] > tau) for tk in adv_topk])
print(f"  (mean poison retrieved/query={npr:.1f}, mean poison FLAGGED/query={nflag:.1f} -> >=2 aggregation is attack-faithful)")

json.dump({"part": "CHEAP-MUST-1", "partA": {c["name"]: {k: c[k] for k in ["tau","docfpr","poison_coh_median","reach_pct","catch","asr","query_fpr_ge1"]} for c in [cT, cP]},
           "L2L3_note": "coh-only hard gate invariant to E-CAL-2 feature neutralization -> = templated case",
           "partB_aggregation": {str(kth): {"benign_query_fpr": float(100.0*np.mean([sum(fl(di) for di in tk)>=kth for tk in ben_topk])),
                                             "poison_catch": float(100.0*np.mean([sum(1 for di in tk if di<P and pcoh[di]>tau)>=kth for tk in adv_topk]))} for kth in [1,2]},
           "mean_poison_retrieved_per_q": float(npr), "mean_poison_flagged_per_q": float(nflag)},
          open(os.path.join(RESULTS_DIR, "cheap_must1_s042.json"), "w", encoding="utf-8"), indent=2)
print("\nsaved cheap_must1_s042.json")
