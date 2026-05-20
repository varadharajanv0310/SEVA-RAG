#!/usr/bin/env python3
"""s_nd_gate.py — CHEAP GATE for a near-duplicate signal (s_nd) to catch clone-inject.

STANDALONE — does NOT touch the frozen seva_benchmark_4060.py. Idea: clone-inject is a NEAR-DUPLICATE
of an in-corpus host (that is how it gets a clean cluster_coh neighborhood + retrievability).
cluster_coh (mean pairwise of K=5 NN) misses the lone twin, but
    s_nd(doc) = MAX cosine to nearest corpus neighbor
is ~0.99 for the clone. s_nd is LLM-free, ~free (reuses the index), CPU, <30 ms -> keeps SEVA's identity.
The whole idea lives/dies on ONE number: s_nd's CLEAN FPR on Security-SE (its lexical-only dedup does
NOT remove semantic near-dups, so this is unmeasured).

STEP 1 (detection): clone s_nd = max cos(clone, CLEAN corpus).        Expect ~0.95-0.99.
STEP 2 (FPR, decisive): clean s_nd = max cos(clean_i, nearest OTHER clean). Report distribution + ROC.
Reuses cached p050 embeddings (NO 100k re-embed); re-embeds only the small 1-rep clone set.
"""
import os, json, numpy as np, torch
from whitebox_attack_seva import _make_poison, _PAYLOADS  # identical clone construction -> provenance

CWD = os.path.dirname(os.path.abspath(__file__)); CK = os.path.join(CWD, "seva_checkpoints_4060_100k_secqa_p050")
RES = os.path.join(CWD, "whitebox_attack_results"); DEV = torch.device("cuda:0")
pe = np.load(os.path.join(CK, "p2_pe.npy")); corpus = json.load(open(os.path.join(CK, "p1_corpus.json"), encoding="utf-8"))
gq = json.load(open(os.path.join(CK, "p1_query.json"), encoding="utf-8"))
isp = np.array([bool(d["is_poisoned"]) for d in corpus])
clean_emb = np.ascontiguousarray(pe[~isp]); clean_texts = [corpus[i]["text"] for i in np.where(~isp)[0]]
Nc = clean_emb.shape[0]; clean_t = torch.from_numpy(clean_emb).to(DEV)
print(f"clean corpus: {Nc} docs (cached p050 embeddings; NO re-embed)")

# ---- STEP 1: clone s_nd (1-rep, low-prominence — the cluster_coh-evasive case) ----
advq = []; seen = set()
for q in gq:
    if q.get("adv") and q["q"] not in seen: advq.append(q["q"]); seen.add(q["q"])
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
Q = torch.from_numpy(enc.encode(advq, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)).to(DEV)
topH = torch.topk(Q @ clean_t.T, 1, dim=1).indices.squeeze(1).cpu().numpy()  # top-1 host per query
clone_texts = [_make_poison(clean_texts[int(topH[qi])], _PAYLOADS[qi % len(_PAYLOADS)]) for qi in range(len(advq))]
clone_emb = enc.encode(clone_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
del enc; torch.cuda.empty_cache()
clone_t = torch.from_numpy(clone_emb).to(DEV)
s_nd_clone = (clone_t @ clean_t.T).max(dim=1).values.cpu().numpy()
print(f"STEP 1  clone s_nd (1-rep, {len(clone_texts)} clones): min/median/max = "
      f"{s_nd_clone.min():.4f} / {np.median(s_nd_clone):.4f} / {s_nd_clone.max():.4f}")

# ---- STEP 2: clean s_nd (exact, chunked; max cos to nearest OTHER clean) ----
s_nd_clean = np.zeros(Nc, dtype=np.float32); CH = 1024
for s in range(0, Nc, CH):
    e = min(s + CH, Nc); sim = clean_t[s:e] @ clean_t.T
    idx = torch.arange(e - s, device=DEV); sim[idx, s + idx] = -2.0  # exclude self
    s_nd_clean[s:e] = sim.max(dim=1).values.cpu().numpy()
pct = {p: float(np.percentile(s_nd_clean, p)) for p in [50, 90, 99, 99.9, 100]}
print(f"STEP 2  clean s_nd ({Nc} docs): pct50/90/99/99.9/max = "
      f"{pct[50]:.4f} / {pct[90]:.4f} / {pct[99]:.4f} / {pct[99.9]:.4f} / {pct[100]:.4f}")

# ---- ROC + headline ----
print(f"\n  {'tau_nd':>7} | {'clone-catch':>11} | {'clean-FPR':>10}")
roc = []
for t in [0.90, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
    cc = 100.0 * np.mean(s_nd_clone > t); fp = 100.0 * np.mean(s_nd_clean > t)
    print(f"  {t:>7.2f} | {cc:>10.1f}% | {fp:>9.3f}%"); roc.append({"tau": t, "clone_catch": cc, "clean_fpr": fp})
TARGET = 0.69
cand = [r for r in roc if r["clone_catch"] >= 95.0]
hl = min(cand, key=lambda r: r["tau"]) if cand else None
if hl:
    v = "GREEN" if hl["clean_fpr"] <= TARGET else ("AMBER" if hl["clean_fpr"] <= 3 * TARGET else "RED")
    print(f"\nHEADLINE: lowest tau_nd catching >=95% clones = {hl['tau']:.2f} (clone-catch {hl['clone_catch']:.0f}%) "
          f"-> clean FPR = {hl['clean_fpr']:.3f}% vs {TARGET}% target -> {v}")
else:
    v = "RED"; print(f"\nHEADLINE: no tau in [0.90,0.99] catches >=95% of clones -> clone s_nd too low -> RED")
json.dump({"Nc": Nc, "n_clones": len(clone_texts), "verdict": v,
           "s_nd_clone": {"min": float(s_nd_clone.min()), "median": float(np.median(s_nd_clone)), "max": float(s_nd_clone.max())},
           "s_nd_clean_pct": pct, "roc": roc, "headline": hl, "target_fpr": TARGET},
          open(os.path.join(RES, "s_nd_gate_s042.json"), "w"), indent=2)
print("saved s_nd_gate_s042.json")
