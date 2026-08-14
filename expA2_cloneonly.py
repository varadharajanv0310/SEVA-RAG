#!/usr/bin/env python3
"""expA2 -- Host-cloning attack scored in its REALISTIC corpus (no templated poison present).

Confound in expA: clones were scored against the p050 corpus which already holds 5,000 templated
poison docs sharing the same payload family, so a clone's nearest neighbours were those poison
docs -> inflated catch. The clone attack REPLACES templated injection; the corpus must be clean.

Defender view: corpus = 95k clean + the injected clones. Calibrate tau on the clean rows at
0.69% FPR, then measure catch on the clones. Swept over injection multiplicity n (clones/query),
which is the variable the prominence frontier says controls sibling clustering.
"""
import os, json, numpy as np, torch
CWD = os.path.dirname(os.path.abspath(__file__))
CK  = os.path.join(CWD, "seva_checkpoints_4060_100k_secqa_p050")
RES = os.path.join(CWD, "whitebox_attack_results")
FPR_TARGET, K, P_POISON = 0.0069, 5, 5000
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from whitebox_attack_seva import _make_poison, _PAYLOADS

pe = np.load(os.path.join(CK, "p2_pe.npy"))
clean_np = pe[P_POISON:]                                    # 95k clean, NO templated poison
print("clean corpus:", clean_np.shape)

pj = json.load(open(os.path.join(RES,"_ndg4_paraphrases_s042.json"),encoding="utf-8"))["para_texts"]
qkeys = sorted(pj, key=int)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-large-en-v1.5", device=str(DEV))
def emb(ts):
    return m.encode(ts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False).astype(np.float32)

def run(n_per_q, tag):
    texts, pi = [], 0
    for q in qkeys:
        for t in pj[q][:n_per_q]:
            texts.append(_make_poison(t, _PAYLOADS[pi % len(_PAYLOADS)])); pi += 1
    cv = emb(texts)
    pool_np = np.concatenate([clean_np, cv], axis=0)        # defender sees clean + injected clones
    pool = torch.from_numpy(pool_np).to(DEV); NCl = clean_np.shape[0]
    iu = torch.triu_indices(K, K, offset=1, device=DEV)
    def score(lo, hi, chunk=2048):
        coh = torch.empty(hi-lo, device=DEV); snd = torch.empty(hi-lo, device=DEV)
        for i in range(lo, hi, chunk):
            j = min(i+chunk, hi); q = pool[i:j]; n = j-i
            sims = q @ pool.T
            sims[torch.arange(n,device=DEV), torch.arange(i,j,device=DEV)] = -2.0
            snd[i-lo:j-lo] = sims.max(dim=1).values
            nb = pool[sims.topk(K,dim=1).indices]
            g = nb @ nb.transpose(1,2)
            coh[i-lo:j-lo] = g[:, iu[0], iu[1]].mean(dim=1)
        return coh.cpu().numpy(), snd.cpu().numpy()
    coh_c, snd_c = score(0, NCl)                            # clean -> calibration
    coh_p, snd_p = score(NCl, pool.shape[0])                # injected clones
    pct = 100.0*(1.0-FPR_TARGET)
    tc, ts = float(np.percentile(coh_c,pct)), float(np.percentile(snd_c,pct))
    lo_, hi_ = pct, 100.0
    for _ in range(60):
        mid=(lo_+hi_)/2; a,b = np.percentile(coh_c,mid), np.percentile(snd_c,mid)
        if float(((coh_c>a)|(snd_c>b)).mean())>FPR_TARGET: lo_=mid
        else: hi_=mid
    mid=(lo_+hi_)/2; tcu,tsu = float(np.percentile(coh_c,mid)), float(np.percentile(snd_c,mid))
    r = {"n_per_query":n_per_q, "n_clones":len(texts),
         "tau_coh":tc, "tau_snd":ts,
         "clone_coh_mean":float(coh_p.mean()), "clone_snd_mean":float(snd_p.mean()),
         "coh_catch_pct":float((coh_p>tc).mean()*100), "snd_catch_pct":float((snd_p>ts).mean()*100),
         "coh_EVASION_pct":float((coh_p<=tc).mean()*100),
         "union_gate_catch_pct":float(((coh_p>tcu)|(snd_p>tsu)).mean()*100),
         "union_realized_clean_FPR_pct":float(((coh_c>tcu)|(snd_c>tsu)).mean()*100),
         "clean_coh_mean":float(coh_c.mean()), "clean_snd_mean":float(snd_c.mean())}
    print(f"\n--- {tag}: n={n_per_q} ({len(texts)} clones) ---")
    print(f"  clone coh {r['clone_coh_mean']:.4f} vs tau {tc:.4f} -> catch {r['coh_catch_pct']:.1f}%  (EVASION {r['coh_EVASION_pct']:.1f}%)")
    print(f"  clone s_nd {r['clone_snd_mean']:.4f} vs tau {ts:.4f} -> catch {r['snd_catch_pct']:.1f}%")
    print(f"  two-signal union gate @0.69% FPR -> catch {r['union_gate_catch_pct']:.1f}%")
    return r

out = {"setting":"clean 95k corpus + injected paraphrase-clones (no templated poison present)",
       "FPR_TARGET_pct":FPR_TARGET*100, "K":K,
       "reference_templated_poison":{"coh_mean":0.99095,"coh_catch_pct":100.0,"snd_catch_pct":100.0,
                                     "note":"from p050 corpus, validated against frozen p2_doc_coh"},
       "sweep":[run(n,"paraphrase-clone") for n in (1,2,3,5)]}
print("\n"+json.dumps(out,indent=2))
json.dump(out, open(os.path.join(RES,"expA2_cloneonly_s042.json"),"w"), indent=2)
print("\nsaved -> whitebox_attack_results/expA2_cloneonly_s042.json")
