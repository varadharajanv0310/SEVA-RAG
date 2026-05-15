#!/usr/bin/env python3
"""whitebox_validity.py — 5-min validity check for the E1b probe (NO full re-run).
Confirms the probe's mean cohesion (~0.59 at hi075 B=250) is each synthetic poison's
frozen cluster_coh = mean pairwise cosine of its 5 nearest neighbours in the COMBINED
REAL corpus [95k clean ; synthetic poison] — NOT a shortcut over the poison set only.
Re-implements the SAME knn as whitebox_attack_seva.cluster_coh_of (combined bank) and
ALSO a poison-only variant for contrast. Reuses cached gate embeddings; one config only.
"""
import os, json, math, numpy as np, torch
K=5; EMB_DIM=1024; N=100000; CWD=os.path.dirname(os.path.abspath(__file__))
CK=os.path.join(CWD,"seva_checkpoints_4060_100k_secqa_p050")
DEV=torch.device("cuda:0"); torch.manual_seed(1337); np.random.seed(1337)

pe=np.load(os.path.join(CK,"p2_pe.npy")); corpus=json.load(open(os.path.join(CK,"p1_corpus.json"),encoding="utf-8"))
queries=json.load(open(os.path.join(CK,"p1_query.json"),encoding="utf-8"))
isp=np.array([bool(d["is_poisoned"]) for d in corpus]); clean_t=torch.from_numpy(np.ascontiguousarray(pe[~isp])).to(DEV)
Nc=clean_t.shape[0]
advq=[]; seen=set()
for q in queries:
    if q.get("adv") and q["q"] not in seen: advq.append(q["q"]); seen.add(q["q"])
from sentence_transformers import SentenceTransformer
enc=SentenceTransformer("BAAI/bge-large-en-v1.5",device="cuda:0")
Q=torch.from_numpy(enc.encode(advq,convert_to_numpy=True,normalize_embeddings=True).astype(np.float32)).to(DEV)
del enc; torch.cuda.empty_cache()
d5=torch.topk(Q@clean_t.T,K,dim=1).values[:,-1].cpu().numpy()

# hi075 B=250 -> n_per_q=5, alpha=max(0.75, d5+0.005)
poison=[]
for qi in range(Q.shape[0]):
    qe=Q[qi]; alpha=min(max(0.75,float(d5[qi])+0.005),0.999)
    G=torch.randn(5,EMB_DIM,device=DEV); G=G-(G@qe)[:,None]*qe[None,:]
    Qo,_=torch.linalg.qr(torch.cat([qe[:,None],G.T],1)); U=Qo[:,1:6].T
    p=alpha*qe[None,:]+math.sqrt(max(1-alpha*alpha,0))*U; poison.append(p/p.norm(dim=1,keepdim=True))
poison_t=torch.cat(poison,0); B=poison_t.shape[0]

def coh(bank, B, offset):
    iu=torch.triu_indices(K,K,offset=1,device=DEV); out=torch.empty(B,device=DEV)
    sims=poison_t@bank.T
    for i in range(B): sims[i, offset+i]=-2.0     # exclude self
    top=torch.topk(sims,K,dim=1).indices; nbr=bank[top]
    pair=torch.bmm(nbr,nbr.transpose(1,2)); return out.copy_(pair[:,iu[0],iu[1]].mean(1))

combined=torch.cat([clean_t,poison_t],0)
coh_combined=coh(combined,B,Nc).mean().item()       # COMBINED real corpus (probe path)
coh_poisononly=coh(poison_t,B,0).mean().item()       # poison-only (shortcut, for contrast)
print(f"  config: hi075 B={B} (n/q=5)  alpha~max(0.75,d5)")
print(f"  COMBINED [clean(95k);poison]  mean cluster_coh = {coh_combined:.4f}   (probe reported 0.5908)")
print(f"  poison-ONLY shortcut          mean cluster_coh = {coh_poisononly:.4f}   (~r*^2 cone floor)")
print(f"  -> probe used COMBINED path: {'CONFIRMED' if abs(coh_combined-0.5908)<0.05 else 'CHECK'} "
      f"(combined>poison-only by {coh_combined-coh_poisononly:+.4f}, clean-neighbour lift)")
