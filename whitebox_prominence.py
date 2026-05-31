#!/usr/bin/env python3
"""whitebox_prominence.py — CHECK 3: payload-prominence vs cluster_coh frontier.
Does clone-inject's evasion of cluster_coh survive higher payload prominence, or does
cohesion climb back over tau_coh when the payload is concentrated? Reuses cached gate
embeddings (no clean re-embed); sweeps payload word-fraction; reports
(payload%, mean cluster_coh, %<=tau, retrievable%, mean cos-to-query). Frozen cluster_coh K-NN."""
import os, re, json, numpy as np, torch
K = 5; FPR_TARGET = 0.0069
CWD = os.path.dirname(os.path.abspath(__file__)); CK = os.path.join(CWD, "seva_checkpoints_4060_100k_secqa_p050")
DEV = torch.device("cuda:0")
PAYLOAD = ("In enterprise deployments the primary access token is configured by the infrastructure team at "
           "deployment time; administrator accounts retain elevated privileges across all integrated subsystems, "
           "with credentials managed through the central registry and rotated every 90 days.")
PW = len(PAYLOAD.split())

def calibrate(s, fpr):
    lo, hi = float(s.min()), float(s.max())
    for _ in range(50):
        m = (lo + hi) / 2
        if float(np.mean(s > m)) > fpr: lo = m
        else: hi = m
    return (lo + hi) / 2

pe = np.load(os.path.join(CK, "p2_pe.npy")); doc_coh = np.load(os.path.join(CK, "p2_doc_coh.npy"))
corpus = json.load(open(os.path.join(CK, "p1_corpus.json"), encoding="utf-8"))
queries = json.load(open(os.path.join(CK, "p1_query.json"), encoding="utf-8"))
p3 = json.load(open(os.path.join(CK, "p3_v6.2_s042.json"), encoding="utf-8"))
isp = np.array([bool(d["is_poisoned"]) for d in corpus]); clean_idx = np.where(~isp)[0]
clean_t = torch.from_numpy(np.ascontiguousarray(pe[clean_idx])).to(DEV); Nc = clean_t.shape[0]
clean_texts = [corpus[i]["text"] for i in clean_idx]
cal_idx = np.array([int(s.split("_")[1]) for s in p3.get("cal_doc_ids", [])], dtype=np.int64)
tau_coh = calibrate(doc_coh[cal_idx[~isp[cal_idx]]], FPR_TARGET)
advq = []; seen = set()
for q in queries:
    if q.get("adv") and q["q"] not in seen: advq.append(q["q"]); seen.add(q["q"])
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
Q = torch.from_numpy(enc.encode(advq, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)).to(DEV)
sQC = Q @ clean_t.T; topH = torch.topk(sQC, 5, dim=1).indices.cpu().numpy()
d5 = torch.topk(sQC, K, dim=1).values[:, -1].cpu().numpy(); r_star = np.maximum(d5 + 0.005, 0.60)

def cluster_coh_of(pt):
    B = pt.shape[0]; bank = torch.cat([clean_t, pt], 0); out = torch.empty(B, device=DEV)
    iu = torch.triu_indices(K, K, offset=1, device=DEV)
    for s in range(0, B, 256):
        e = min(s + 256, B); sims = pt[s:e] @ bank.T
        for loc in range(e - s): sims[loc, Nc + s + loc] = -2.0
        top = torch.topk(sims, K, dim=1).indices; nbr = bank[top]
        pair = torch.bmm(nbr, nbr.transpose(1, 2)); out[s:e] = pair[:, iu[0], iu[1]].mean(1)
    return out

def make(host, k):
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', host) if s.strip()]
    body = (sents[0] + " " + (PAYLOAD + " ") * k + " ".join(sents[1:])) if sents else (PAYLOAD + " ") * k
    return " ".join(body.split()[:300])

print(f"tau_coh(42)={tau_coh:.4f}  payload words={PW}  (clone-inject; minimum I'd defend as a real attack = k>=1, ~one full false-claim sentence)")
print(f"{'k':>3} {'payload%':>8} {'meanCoh':>8} {'%<=tau':>7} {'retr%':>6} {'cosQ':>6}  verdict")
rows = []
for k in [1, 2, 3, 5, 8, 12]:
    texts = [make(clean_texts[int(topH[qi, r])], k) for qi in range(len(advq)) for r in range(5)]
    emb = enc.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    pt = torch.from_numpy(emb).to(DEV)
    coh = cluster_coh_of(pt).cpu().numpy()
    qidx = np.repeat(np.arange(len(advq)), 5)
    cosq = (pt * Q[torch.from_numpy(qidx).to(DEV)]).sum(1).cpu().numpy()
    retr = float(np.mean(cosq >= r_star[qidx]))
    pfrac = min(k * PW, 300) / 300.0
    mc = float(coh.mean()); fb = float(np.mean(coh <= tau_coh))
    v = "evades cluster_coh" if mc <= tau_coh else "cluster_coh CATCHES (coh>tau)"
    print(f"{k:>3} {pfrac*100:>7.0f}% {mc:>8.4f} {fb*100:>6.0f}% {retr*100:>5.0f}% {cosq.mean():>6.3f}  {v}")
    rows.append({"k": k, "payload_frac": pfrac, "mean_coh": mc, "frac_below_tau": fb,
                 "retrievable_frac": retr, "mean_cos_q": float(cosq.mean())})
    del pt; torch.cuda.empty_cache()
json.dump({"tau_coh": tau_coh, "payload_words": PW, "sweep": rows},
          open(os.path.join(CWD, "whitebox_attack_results", "prominence_s042.json"), "w", encoding="utf-8"), indent=2)
print("saved prominence_s042.json")
