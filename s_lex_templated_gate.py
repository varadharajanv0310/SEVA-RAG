#!/usr/bin/env python3
"""s_lex_templated_gate.py — ND-GATE-5: does `s_lex` (lexical near-dup, hard pre-filter) catch the
TEMPLATED poison that L2/L3 adaptive evasion lets through?

E-CAL-2 (frozen L1, strict no-re-adaptation): neutralizing kw_density(+avg_sent_len) at the feature
level drops the SOFT A-score below tau for ~49% (c1) of retrieved held-out templated poison -> the
adaptive collapse. `s_lex` is a HARD gate (max word-5-gram Jaccard to any other corpus doc), so it is
INDEPENDENT of the soft A-score: if the templated poison is lexically redundant (sibling near-dups),
s_lex removes it BEFORE scoring, regardless of which soft signals the adversary neutralized. If so,
the s_lex pre-filter converts the L2/L3 collapse into a caught case.

This MIRRORS `whitebox_attack_seva.run_ecal2` exactly (same frozen weights/tau from linch-A, same
held-out half-B, same FAISS retrieval + cluster_coh + scoring) and ADDS: (1) s_lex for every corpus
doc via numpy MinHash+LSH; (2) a hard pre-filter — a retrieved poison "evades" only if it evades the
A-score gate AND s_lex<=tau_lex. STANDALONE; frozen seva_benchmark_4060.py only IMPORTED (as run_ecal2
does), never modified. No new deps.

VERDICT: GREEN if the worst adaptive ASR (c1/c2) collapses to ~0 under the s_lex pre-filter (at the
ND-GATE-2 clean FPR), i.e. the lexical hard gate is immune to soft-signal neutralization.
"""
import os, sys, json, zlib, time
import numpy as np
from collections import defaultdict
import torch, importlib.util
from whitebox_attack_seva import CWD, CKDIR, RESULTS_DIR, SYNTH_SEED

TAU_LEX = 0.30   # ND-GATE-2 GREEN operating point
DEVICE = torch.device("cuda:0"); torch.manual_seed(SYNTH_SEED); np.random.seed(SYNTH_SEED)

# ---- MinHash (IDENTICAL to s_lex_gate.py / ND-GATE-2,3) ----
PRIME = (1 << 31) - 1; KH = 128; B, R = 32, 4
_rng = np.random.default_rng(1337); MA = _rng.integers(1, PRIME, KH, dtype=np.uint64); MB = _rng.integers(0, PRIME, KH, dtype=np.uint64)
def shingles(text):
    w = text.lower().split()
    if len(w) < 5: return {np.uint64(zlib.crc32(text.lower().encode()) % PRIME)}
    return {np.uint64(zlib.crc32((" ".join(w[i:i+5])).encode()) % PRIME) for i in range(len(w) - 4)}
def minhash(sh):
    s = np.fromiter(sh, dtype=np.uint64, count=len(sh))
    return ((MA[:, None] * s[None, :] + MB[:, None]) % PRIME).min(axis=1).astype(np.int64)

# ---- frozen calibration from linch-A (same as run_ecal2) ----
linchA_p3 = os.path.join(CWD, "seva_checkpoints_4060_100k_secqawblinchA", "p3_v6.2_s042.json")
cal = json.load(open(linchA_p3, encoding="utf-8"))
L1w, L2w, L3w = cal["L1_weights"], cal["L2_weights"], cal["L3_weights"]
tau1, tau2, tau3 = cal["tau_L1"], cal["tau_L2"], cal["tau_L3"]
flipped = set(cal.get("flipped_signals", [])); norm = cal["norm_config"]
print(f"frozen tau_L1={tau1:.4f} tau_L2={tau2:.4f} tau_L3={tau3:.4f} | L1 wt kw_density={L1w.get('kw_density',0):.3f} "
      f"avg_sent_len={L1w.get('avg_sent_len_signal',0):.3f} cluster_coh={L1w.get('cluster_coh',0):.3f}")

pe = np.load(os.path.join(CKDIR, "p2_pe.npy")); corpus = json.load(open(os.path.join(CKDIR, "p1_corpus.json"), encoding="utf-8"))
gq = json.load(open(os.path.join(CKDIR, "p1_query.json"), encoding="utf-8"))
isp = np.array([bool(d["is_poisoned"]) for d in corpus])
clean_emb = np.ascontiguousarray(pe[~isp]); clean_texts = [corpus[i]["text"] for i in np.where(~isp)[0]]
half_B = [d["text"] for d in json.load(open(os.path.join(CWD, "poison_corpus_diverse.json"), encoding="utf-8"))][1::2][:2500]
advq = []; seen = set()
for q in gq:
    if q.get("adv") and q["q"] not in seen: advq.append(q["q"]); seen.add(q["q"])

# ---- import frozen detector for text_features / K / K_FETCH / faiss (as run_ecal2 does) ----
mod_name = "seva_ndg5_42"
if mod_name in sys.modules: del sys.modules[mod_name]
saved = sys.argv[:]
sys.argv = ["seva_benchmark_4060.py", "--corpus", "100000", "--poison-ratio", "0.0025", "--cal-seed", "42", "--benign-q", "2000", "--corpus-tag", "secqawbndg5"]
try:
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(CWD, "seva_benchmark_4060.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
finally:
    sys.argv = saved
text_features = m.text_features; K = m.K; K_FETCH = m.K_FETCH; _faiss = m.faiss

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
embB = enc.encode(half_B, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
qemb = enc.encode(advq, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
del enc; torch.cuda.empty_cache()
P = len(half_B)
pe_full = np.ascontiguousarray(np.concatenate([embB, clean_emb], 0), dtype=np.float32)
texts_full = half_B + clean_texts; Nfull = len(texts_full)
_faiss.omp_set_num_threads(1)
idx = _faiss.IndexHNSWFlat(m.EMB_DIM, m.INDEX_M, _faiss.METRIC_INNER_PRODUCT); idx.hnsw.efConstruction = m.INDEX_EF; idx.add(pe_full)
cr = pe_full.mean(0); centroid = (cr / np.linalg.norm(cr)).astype(np.float32)
doc_coh = np.zeros(Nfull, dtype=np.float32)
for s in range(0, Nfull, 512):
    e = min(s + 512, Nfull); _, nbr = idx.search(np.ascontiguousarray(pe_full[s:e]), K + 1)
    for bi in range(e - s):
        di = s + bi; valid = [int(j) for j in nbr[bi] if j >= 0 and int(j) != di][:K]
        if len(valid) >= 2:
            em = pe_full[valid]; sim = np.dot(em, em.T); doc_coh[di] = float(sim[np.triu_indices(len(valid), k=1)].mean())
        else: doc_coh[di] = 0.5

# ---- s_lex for EVERY corpus doc (half_B poison + clean) via MinHash-LSH ----
t0 = time.time(); sig = np.empty((Nfull, KH), dtype=np.int64)
for i in range(Nfull): sig[i] = minhash(shingles(texts_full[i]))
cand = defaultdict(set); capped = 0
for j in range(B):
    sub = sig[:, j*R:(j+1)*R]; _, inv = np.unique(sub, axis=0, return_inverse=True)
    buckets = defaultdict(list)
    for d, g in enumerate(inv.tolist()): buckets[g].append(d)
    for docs in buckets.values():
        if len(docs) < 2: continue
        if len(docs) > 800: capped += 1; docs = docs[:800]
        for x in docs:
            for y in docs:
                if x != y: cand[x].add(y)
s_lex = np.zeros(Nfull, dtype=np.float32)
for d in range(Nfull):
    cs = cand.get(d)
    if not cs: continue
    cs = list(cs); s_lex[d] = (sig[d][None, :] == sig[cs]).mean(axis=1).max()
print(f"  minhashed+LSH {Nfull} corpus docs in {time.time()-t0:.1f}s ({capped} buckets capped)")
poison_catch = 100.0 * np.mean(s_lex[:P] > TAU_LEX); clean_fpr = 100.0 * np.mean(s_lex[P:] > TAU_LEX)
ps = s_lex[:P]
print(f"s_lex templated poison: pct1/10/50/90 = {np.percentile(ps,1):.3f}/{np.percentile(ps,10):.3f}/{np.percentile(ps,50):.3f}/{np.percentile(ps,90):.3f}")
print(f"HEADLINE-A: s_lex(>{TAU_LEX}) catches {poison_catch:.1f}% of templated poison | clean FPR = {clean_fpr:.3f}% (vs ND-GATE-2's 0.165%)")
for t in [0.30, 0.40, 0.50]:
    print(f"   tau_lex={t:.2f}: poison-catch={100*np.mean(ps>t):5.1f}%  clean-FPR={100*np.mean(s_lex[P:]>t):.3f}%")

# ---- ASR loop (verbatim run_ecal2) + s_lex hard pre-filter ----
rng = np.random.default_rng(99)
csample = rng.choice(len(clean_texts), size=min(2000, len(clean_texts)), replace=False)
clean_kwd, clean_asl = [], []
for ci in csample:
    su, ttr, rr, kwd, dl, asl, ps2, cts = text_features(clean_texts[int(ci)], norm)
    clean_kwd.append(kwd); clean_asl.append(asl)
clean_kwd = np.array(clean_kwd); clean_asl = np.array(clean_asl)
def score(fd, weights):
    return sum(weights.get(k, 0.0) * ((1.0 - v) if k in flipped else v) for k, v in fd.items() if weights.get(k, 0.0) > 0)
L1w_k0 = dict(L1w); L1w_k0["kw_density"] = 0.0
L1w_ka0 = dict(L1w); L1w_ka0["kw_density"] = 0.0; L1w_ka0["avg_sent_len_signal"] = 0.0
# res[k] = [evades(plain), evades(+s_lex prefilter), attempts]
res = {k: [0, 0, 0] for k in ["L1ref", "b_L2", "b_L3", "c1_L2", "c1_L3", "c2_L2", "c2_L3"]}
for qi in range(len(advq)):
    qe = qemb[qi:qi+1]; _, ir = idx.search(np.ascontiguousarray(qe), K_FETCH); ri = [int(i) for i in ir[0] if i >= 0]
    sims = [float(np.dot(pe_full[j], qe[0])) for j in ri]
    topk = [ri[x] for x in sorted(range(len(ri)), key=lambda x: sims[x], reverse=True)[:K]]
    for di in topk:
        if di >= P: continue
        slex_evade = (s_lex[di] <= TAU_LEX)   # poison survives the hard pre-filter only if lexically NON-redundant
        su, ttr, rr, kwd, dl, asl, ps2, cts = text_features(texts_full[di], norm)
        drift = 1.0 - float(np.dot(pe_full[di], centroid)); coh = float(doc_coh[di])
        base = {"topic_drift": drift, "sent_unif": su, "ttr_signal": ttr, "repeat_rate": rr, "kw_density": kwd,
                "doc_length_signal": dl, "avg_sent_len_signal": asl, "punct_signal": ps2, "content_ttr_signal": cts, "cluster_coh": coh}
        f_c1L2 = dict(base); f_c1L2["kw_density"] = float(rng.choice(clean_kwd))
        f_c1L3 = dict(f_c1L2); f_c1L3["avg_sent_len_signal"] = float(rng.choice(clean_asl))
        ev = {"L1ref": score(base, L1w) <= tau1, "b_L2": score(base, L2w) <= tau2, "b_L3": score(base, L3w) <= tau3,
              "c1_L2": score(f_c1L2, L1w) <= tau1, "c1_L3": score(f_c1L3, L1w) <= tau1,
              "c2_L2": score(base, L1w_k0) <= tau1, "c2_L3": score(base, L1w_ka0) <= tau1}
        for k, e in ev.items():
            res[k][2] += 1; res[k][0] += int(e); res[k][1] += int(e and slex_evade)
def asr(k, col): return 100.0 * res[k][col] / res[k][2] if res[k][2] else 0.0
print(f"\nattempts (retrieved held-out templated poison) = {res['L1ref'][2]}  (E-CAL-2 had 125)")
print(f"  {'variant':>8} | {'ASR plain':>9} | {'ASR + s_lex prefilter':>21}")
for k in ["L1ref", "b_L2", "b_L3", "c1_L2", "c1_L3", "c2_L2", "c2_L3"]:
    print(f"  {k:>8} | {asr(k,0):>8.1f}% | {asr(k,1):>20.1f}%")
worst_plain = max(asr(k,0) for k in ["b_L2","b_L3","c1_L2","c1_L3","c2_L2","c2_L3"])
worst_pref = max(asr(k,1) for k in ["b_L2","b_L3","c1_L2","c1_L3","c2_L2","c2_L3"])
c1 = asr("c1_L3", 0); c1p = asr("c1_L3", 1)
verdict = "GREEN" if worst_pref <= 5.0 else ("AMBER" if worst_pref <= 15.0 else "RED")
print(f"\nHEADLINE-B: worst adaptive ASR {worst_plain:.1f}% (c1={c1:.1f}%) -> with s_lex hard pre-filter {worst_pref:.1f}% (c1={c1p:.1f}%) -> {verdict}")
print(f"VERDICT: {verdict} — s_lex pre-filter " + ("CONVERTS the L2/L3 adaptive collapse into a caught case (lexical hard gate immune to soft-signal neutralization)" if verdict=="GREEN" else ("substantially reduces but does not eliminate the collapse" if verdict=="AMBER" else "does NOT rescue the adaptive collapse")))
json.dump({"tau_lex": TAU_LEX, "attempts": res["L1ref"][2], "poison_catch_pct": poison_catch, "clean_fpr_pct": clean_fpr,
           "s_lex_poison_pct": {p: float(np.percentile(ps, p)) for p in [1,10,50,90,99]},
           "asr_plain": {k: asr(k,0) for k in res}, "asr_with_slex_prefilter": {k: asr(k,1) for k in res},
           "worst_plain": worst_plain, "worst_prefilter": worst_pref, "verdict": verdict,
           "ecal2_baseline": {"c1": 48.8, "c2": 56.8, "b": 53.6, "note": "L2=L3 (frozen avg_sent_len wt=0)"}},
          open(os.path.join(RESULTS_DIR, "s_lex_templated_gate_s042.json"), "w", encoding="utf-8"), indent=2)
print("saved s_lex_templated_gate_s042.json")
