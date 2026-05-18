#!/usr/bin/env python3
"""E4-HH fair-comparison checks — RUN IN THE `ragdefender` env.

CHECK 1 (Nadv-GATE): run RAGDefender's single_hop grouping on CLEAN retrieved sets; report the
N_adv it estimates (= docs it would remove). Confirms whether it removes a cluster REGARDLESS
(no no-attack gate) -> the 50% benign FPR is structural, and defend() ASRs are at a remove-~half
operating point (not comparable to SEVA's 0.6% FPR).

CHECK 2 (MATCHED-FPR ROC): defend() has no FPR knob, so sweep RAGDefender's underlying concentration
signal (per-doc within-window mean pairwise cosine = the premise both grouping strategies rest on).
Trace benign-FPR vs catch on templated AND clone-inject; report catch at ~SEVA's FPR (0.69%), or the
FPR floor. Same retrieved artifact (squeeze_retrieval_s042.json) -> provenance. bge-large = SEVA encoder.
"""
import json, os, numpy as np
from sentence_transformers import SentenceTransformer
from ragdefender.grouping import ClusteringBasedGrouping

CWD = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(CWD, "whitebox_attack_results")
data = json.load(open(os.path.join(RES, "squeeze_retrieval_s042.json"), encoding="utf-8"))
enc = SentenceTransformer("BAAI/bge-large-en-v1.5")
print(f"device={enc.device}")

def smean(R):
    E = enc.encode(R, convert_to_numpy=True, normalize_embeddings=True)
    S = E @ E.T; k = len(R)
    return [float((S[i].sum() - S[i, i]) / (k - 1)) if k > 1 else 0.0 for i in range(k)]

# ---- CHECK 1: N_adv on clean retrieved sets ----
grp = ClusteringBasedGrouping(enc, m=5)
nadv = []
for q in data["benign"][:30]:
    E = enc.encode(q["R"], convert_to_tensor=True)
    nadv.append(int(grp.estimate_n_adv(q["R"], embeddings=E)))
nadv = np.array(nadv); klen = len(data["benign"][0]["R"])
print("\n[CHECK 1 — Nadv-GATE] single_hop estimate_n_adv on 30 CLEAN k=%d sets:" % klen)
print(f"  n_adv per clean set: {list(nadv)}")
print(f"  min/mean/max = {nadv.min()}/{nadv.mean():.2f}/{nadv.max()} ; sets with n_adv==0: {int((nadv==0).sum())}/30")
print(f"  -> removes {nadv.mean():.2f}/{klen} docs from a CLEAN set on average; NO no-attack gate (n_adv in [1,|R|-1] by construction).")

# ---- CHECK 2: matched-FPR ROC on the concentration signal ----
benign_clean = np.array([s for q in data["benign"] for s in smean(q["R"])])
def poison_sm(cfg):
    out = []
    for q in data["configs"][cfg]["queries"]:
        if not q["poison_pos"]:
            continue
        sm = smean(q["R"])
        out += [sm[p] for p in q["poison_pos"]]
    return np.array(out)
templ = poison_sm("templated"); clone = poison_sm("cloneinject_n5")
fpr = lambda t: 100.0 * np.mean(benign_clean > t)
catch = lambda a, t: 100.0 * np.mean(a > t)
print(f"\n[CHECK 2 — MATCHED-FPR ROC] concentration signal (within-window mean pairwise cos)")
print(f"  benign clean s_mean: min/median/max = {benign_clean.min():.3f}/{np.median(benign_clean):.3f}/{benign_clean.max():.3f}  (n={len(benign_clean)})")
print(f"  templated poison s_mean: median={np.median(templ):.3f} (n={len(templ)});  clone-inject s_mean: median={np.median(clone):.3f} (n={len(clone)})")
print(f"  {'benign-FPR':>11} | {'tau':>7} | {'templated-catch':>15} | {'clone-inject-catch':>18}")
rows = []
for ftgt in [0.69, 2, 5, 10, 25, 50]:
    t = float(np.quantile(benign_clean, 1 - ftgt / 100.0))
    tc, cc = catch(templ, t), catch(clone, t)
    af = fpr(t)
    print(f"  {af:10.2f}% | {t:7.4f} | {tc:14.1f}% | {cc:17.1f}%")
    rows.append({"benign_fpr_target": ftgt, "tau": t, "benign_fpr_actual": af, "templated_catch": tc, "clone_inject_catch": cc})
# FPR floor: lowest achievable FPR with this signal = fraction of benign above the global max poison-ish? report min nonzero
tmax = float(benign_clean.max()); floor_fpr = fpr(tmax - 1e-6)
print(f"  FPR floor (tau just below max benign s_mean {tmax:.3f}): ~{floor_fpr:.2f}%")
out = {"check1_nadv_clean": {"values": list(map(int, nadv)), "mean": float(nadv.mean()), "zero_count": int((nadv == 0).sum()), "k": klen},
       "check2_roc": rows, "benign_smean_stats": {"min": float(benign_clean.min()), "median": float(np.median(benign_clean)), "max": float(benign_clean.max())},
       "templated_smean_median": float(np.median(templ)), "clone_smean_median": float(np.median(clone))}
json.dump(out, open(os.path.join(RES, "e4hh_fair_s042.json"), "w"), indent=2)
print("\n  saved e4hh_fair_s042.json")
