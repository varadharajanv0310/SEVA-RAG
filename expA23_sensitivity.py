#!/usr/bin/env python3
"""expA2+A3 -- Sensitivity of the geometric gate to its two frozen choices.

A2  K-sensitivity: cluster_coh is defined over K=5 nearest neighbours. Is the result an artifact
    of that choice? Recompute for K in {3,5,10,20}, recalibrate tau non-oracle at 0.69% FPR each
    time, and report catch/FPR. Answers "why 5, and is this tuned?".
A3  Operating-point sensitivity: everything is reported at a single 0.69% FPR target. Sweep the
    target in-domain and report the catch/FPR curve + AUC, so the result is not an artifact of
    one oddly specific threshold.

Runs entirely on cached p050 embeddings. No LLM, no re-embedding.
"""
import os, json, time, numpy as np, torch

CWD = os.path.dirname(os.path.abspath(__file__))
CK  = os.path.join(CWD, "seva_checkpoints_4060_100k_secqa_p050")
RES = os.path.join(CWD, "whitebox_attack_results")
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
P_POISON, FPR_TARGET = 5000, 0.0069
t0 = time.time()

pe = np.load(os.path.join(CK, "p2_pe.npy"))
frozen = np.load(os.path.join(CK, "p2_doc_coh.npy"))
pool = torch.from_numpy(pe).to(DEV)
N = pool.shape[0]
print(f"pool {tuple(pool.shape)} | frozen doc_coh poison {frozen[:P_POISON].mean():.4f} "
      f"clean {frozen[P_POISON:].mean():.4f}", flush=True)

def coh_for_K(K, chunk=2048):
    """mean pairwise cos among the K nearest neighbours (self excluded), whole corpus."""
    out = np.empty(N, dtype=np.float32)
    iu = torch.triu_indices(K, K, offset=1, device=DEV)
    for i in range(0, N, chunk):
        q = pool[i:i+chunk]; n = q.shape[0]
        s = q @ pool.T
        s[torch.arange(n, device=DEV), torch.arange(i, i+n, device=DEV)] = -2.0
        nb = pool[s.topk(K, dim=1).indices]
        g = nb @ nb.transpose(1, 2)
        out[i:i+chunk] = g[:, iu[0], iu[1]].mean(dim=1).cpu().numpy()
    return out

# ---------------- A2: K sensitivity ----------------
print("\n== A2: K sensitivity ==", flush=True)
a2 = []
for K in (3, 5, 10, 20):
    c = coh_for_K(K)
    cl, po = c[P_POISON:], c[:P_POISON]
    tau = float(np.percentile(cl, 100 * (1 - FPR_TARGET)))
    row = {"K": K, "tau": tau, "clean_coh_mean": float(cl.mean()), "poison_coh_mean": float(po.mean()),
           "catch_pct": float((po > tau).mean() * 100), "clean_FPR_pct": float((cl > tau).mean() * 100),
           "poison_evasion_pct": float((po <= tau).mean() * 100)}
    if K == 5:
        row["validates_against_frozen"] = float(abs(po.mean() - frozen[:P_POISON].mean()))
    a2.append(row)
    print(f"  K={K:2d}  tau={tau:.4f}  clean={cl.mean():.4f} poison={po.mean():.4f}  "
          f"CATCH={row['catch_pct']:.1f}%  FPR={row['clean_FPR_pct']:.3f}%  ({time.time()-t0:.0f}s)", flush=True)

# ---------------- A3: operating-point sweep (frozen K=5 doc_coh) ----------------
print("\n== A3: operating-point sweep ==", flush=True)
cl, po = frozen[P_POISON:], frozen[:P_POISON]
a3 = []
for tgt in (0.001, 0.0025, 0.005, 0.0069, 0.01, 0.02, 0.05):
    tau = float(np.percentile(cl, 100 * (1 - tgt)))
    a3.append({"fpr_target_pct": tgt * 100, "tau": tau,
               "catch_pct": float((po > tau).mean() * 100),
               "realized_FPR_pct": float((cl > tau).mean() * 100)})
    print(f"  target={tgt*100:5.2f}%  tau={tau:.4f}  CATCH={a3[-1]['catch_pct']:.1f}%  "
          f"realized FPR={a3[-1]['realized_FPR_pct']:.3f}%", flush=True)

# AUC over the clean/poison separation (rank-based, exact)
lab = np.concatenate([np.ones_like(po), np.zeros_like(cl)])
sc  = np.concatenate([po, cl])
order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
npos, nneg = lab.sum(), (1 - lab).sum()
auc = float((ranks[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
print(f"  AUC = {auc:.5f}", flush=True)

out = {"A2_K_sensitivity": {"note": "cluster_coh recomputed for each K; tau recalibrated non-oracle "
                                    "at 0.69% FPR per K; whole 100k p050 corpus", "rows": a2},
       "A3_operating_point": {"note": "frozen K=5 doc_coh; FPR target swept in-domain", "rows": a3,
                              "AUC": auc},
       "runtime_s": time.time() - t0}
json.dump(out, open(os.path.join(RES, "expA23_sensitivity_s042.json"), "w"), indent=2)
print("\nsaved -> whitebox_attack_results/expA23_sensitivity_s042.json")
