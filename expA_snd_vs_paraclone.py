#!/usr/bin/env python3
"""expA_snd_vs_paraclone.py -- Do the geometric signals catch host-cloning attacks?

cluster_coh(d) = mean pairwise cos among d's K=5 nearest neighbours IN THE FULL CORPUS.
s_nd(d)        = max cos from d to its nearest corpus neighbour.

Templated poison: neighbours are its own siblings -> coh ~0.99 (caught), s_nd ~0.99 (caught).
Paraphrase-clone: anchored to a DISTINCT benign host -> neighbourhood looks normal (evades coh).
QUESTION: does s_nd catch it (paraphrase preserves meaning -> stays near its host)?

NB: the neighbour pool MUST be the full (poisoned) corpus, else the templated signal vanishes.
Validated against the frozen cached p2_doc_coh.npy.
"""
import os, json, numpy as np, torch

CWD = os.path.dirname(os.path.abspath(__file__))
CK  = os.path.join(CWD, "seva_checkpoints_4060_100k_secqa_p050")
RES = os.path.join(CWD, "whitebox_attack_results")
FPR_TARGET, K, P_POISON = 0.0069, 5, 5000
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from whitebox_attack_seva import _make_poison, _PAYLOADS

print("== load cached embeddings + frozen doc_coh ==")
pe = np.load(os.path.join(CK, "p2_pe.npy"))
frozen_coh = np.load(os.path.join(CK, "p2_doc_coh.npy"))
pool = torch.from_numpy(pe).to(DEV)                       # FULL poisoned corpus = neighbour pool
print("pool", tuple(pool.shape), "| frozen doc_coh: poison mean %.4f  clean mean %.4f"
      % (frozen_coh[:P_POISON].mean(), frozen_coh[P_POISON:].mean()))

def scores(Q, self_offset=None, chunk=2048):
    """-> (coh, s_nd) against the full pool. self_offset=row index in pool to exclude, else None."""
    coh = torch.empty(Q.shape[0], device=DEV); snd = torch.empty(Q.shape[0], device=DEV)
    iu = torch.triu_indices(K, K, offset=1, device=DEV)
    for i in range(0, Q.shape[0], chunk):
        q = Q[i:i+chunk]; n = q.shape[0]
        sims = q @ pool.T
        if self_offset is not None:
            sims[torch.arange(n, device=DEV), torch.arange(self_offset+i, self_offset+i+n, device=DEV)] = -2.0
        snd[i:i+chunk] = sims.max(dim=1).values
        idx = sims.topk(K, dim=1).indices
        nb = pool[idx]; g = nb @ nb.transpose(1, 2)
        coh[i:i+chunk] = g[:, iu[0], iu[1]].mean(dim=1)
    return coh.cpu().numpy(), snd.cpu().numpy()

print("== score templated poison + clean (validate coh against frozen) ==")
coh_pois, snd_pois   = scores(pool[:P_POISON], self_offset=0)
coh_clean, snd_clean = scores(pool[P_POISON:], self_offset=P_POISON)
print("   my coh  : poison %.4f  clean %.4f" % (coh_pois.mean(), coh_clean.mean()))
print("   frozen  : poison %.4f  clean %.4f" % (frozen_coh[:P_POISON].mean(), frozen_coh[P_POISON:].mean()))
dev = abs(coh_pois.mean() - frozen_coh[:P_POISON].mean())
print("   VALIDATION: |delta| = %.5f -> %s" % (dev, "OK" if dev < 0.01 else "MISMATCH"))

print("== build + score paraphrase-clones ==")
pj = json.load(open(os.path.join(RES,"_ndg4_paraphrases_s042.json"),encoding="utf-8"))["para_texts"]
hosts_para = [t for k in sorted(pj, key=int) for t in pj[k]]
clones = [_make_poison(t, _PAYLOADS[i % len(_PAYLOADS)]) for i, t in enumerate(hosts_para)]
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-large-en-v1.5", device=str(DEV))
def emb(ts):
    v = m.encode(ts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=False).astype(np.float32)
    return torch.from_numpy(v).to(DEV)
coh_host,  snd_host  = scores(emb(hosts_para))    # paraphrased host, no payload
coh_clone, snd_clone = scores(emb(clones))        # the real attack artifact
print("   clones:", len(clones))

pct = 100.0*(1.0-FPR_TARGET)
tau_coh, tau_snd = float(np.percentile(coh_clean,pct)), float(np.percentile(snd_clean,pct))
c = lambda v,t: float((v>t).mean()*100.0)

lo, hi = pct, 100.0
for _ in range(60):
    mid=(lo+hi)/2; tc,ts = np.percentile(coh_clean,mid), np.percentile(snd_clean,mid)
    if float(((coh_clean>tc)|(snd_clean>ts)).mean())>FPR_TARGET: lo=mid
    else: hi=mid
mid=(lo+hi)/2; tcu,tsu = float(np.percentile(coh_clean,mid)), float(np.percentile(snd_clean,mid))
cu = lambda a,b: float((((a>tcu)|(b>tsu))).mean()*100.0)

out = {
 "n_clean":int(len(coh_clean)), "n_templated":int(P_POISON), "n_paraphrase_clone":len(clones),
 "FPR_TARGET_pct":FPR_TARGET*100, "K":K,
 "validation_vs_frozen_doc_coh":{"my_poison_coh":float(coh_pois.mean()),
   "frozen_poison_coh":float(frozen_coh[:P_POISON].mean()),"abs_delta":float(dev)},
 "thresholds":{"tau_coh":tau_coh,"tau_snd":tau_snd,
   "clean_FPR_coh_pct":c(coh_clean,tau_coh),"clean_FPR_snd_pct":c(snd_clean,tau_snd)},
 "clean_ref":{"coh_mean":float(coh_clean.mean()),"snd_mean":float(snd_clean.mean()),
              "snd_median":float(np.median(snd_clean))},
 "templated_poison":{"coh_mean":float(coh_pois.mean()),"snd_mean":float(snd_pois.mean()),
   "coh_catch_pct":c(coh_pois,tau_coh),"snd_catch_pct":c(snd_pois,tau_snd)},
 "paraphrased_host_no_payload":{"coh_mean":float(coh_host.mean()),"snd_mean":float(snd_host.mean()),
   "coh_catch_pct":c(coh_host,tau_coh),"snd_catch_pct":c(snd_host,tau_snd)},
 "paraphrase_clone":{"coh_mean":float(coh_clone.mean()),"snd_mean":float(snd_clone.mean()),
   "coh_catch_pct":c(coh_clone,tau_coh),"snd_catch_pct":c(snd_clone,tau_snd),
   "coh_EVASION_pct":100-c(coh_clone,tau_coh),"snd_EVASION_pct":100-c(snd_clone,tau_snd)},
 "two_signal_union_gate":{"percentile":mid,"tau_coh":tcu,"tau_snd":tsu,
   "realized_clean_FPR_pct":float(((coh_clean>tcu)|(snd_clean>tsu)).mean()*100),
   "templated_catch_pct":cu(coh_pois,snd_pois),
   "paraphrase_clone_catch_pct":cu(coh_clone,snd_clone)},
}
print("\n"+json.dumps(out,indent=2))
json.dump(out, open(os.path.join(RES,"expA_snd_vs_paraclone_s042.json"),"w"), indent=2)
print("\nsaved -> whitebox_attack_results/expA_snd_vs_paraclone_s042.json")
