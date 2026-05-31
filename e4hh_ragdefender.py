#!/usr/bin/env python3
"""E4-HH RAGDefender-side scorer — RUN IN THE `ragdefender` env (separate from frozen `seva`).

Loads squeeze_retrieval_s042.json (produced by whitebox_attack_seva.py squeezegen in the seva env:
the SAME clone-inject/templated artifact -> provenance), runs RAGDefender (bge-large = SEVA's encoder,
single_hop, the confirmed task_type) on each query's retrieved top-K, and combines with the SEVA
frozen-L1 flags to produce the squeeze table + 2x2 + benign FPR. Catch metric is made comparable to
SEVA's retrieved-unflagged ASR: RAGD-ASR = retrieved-poison-NOT-removed / retrieved-poison.
"""
import json, os
from ragdefender import RAGDefender

CWD = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CWD, "whitebox_attack_results")
data = json.load(open(os.path.join(RES, "squeeze_retrieval_s042.json"), encoding="utf-8"))
print("Loading RAGDefender (embedder=BAAI/bge-large-en-v1.5, task_type=single_hop) ...")
D = RAGDefender(embedder="BAAI/bge-large-en-v1.5", task_type="single_hop", device="auto")
print(f"  device={D.device}")


def eval_config(c):
    att = sc = rc = both = qboth = nq = 0
    flat_removed = []  # per-poison RAGD-removed bool (poison iteration order; seed-invariant) -> for 3-seed SURVIVE-BOTH
    for q in c["queries"]:
        pos = q["poison_pos"]
        if not pos:
            continue
        nq += 1
        _, removed = D.defend(query=q["q"], R=q["R"], return_indices=True)
        removed = set(int(x) for x in removed)
        qsurv = False
        for k, p in enumerate(pos):
            att += 1
            sflag = bool(q["seva_flag"][k]); rflag = (p in removed)
            flat_removed.append(bool(rflag))
            sc += sflag; rc += rflag
            if (not sflag) and (not rflag):
                both += 1; qsurv = True
        if qsurv:
            qboth += 1
    pct = lambda x: 100.0 * x / att if att else 0.0
    return ({"attempts": att, "nq": nq, "seva_catch_pct": pct(sc), "ragd_catch_pct": pct(rc),
             "seva_asr": pct(att - sc), "ragd_asr": pct(att - rc), "survive_both_asr": pct(both),
             "q_survive_both": qboth, "q_survive_both_pct": (100.0 * qboth / nq if nq else 0.0)}, flat_removed)


print("\n  config           | attempts | SEVA-catch  RAGD-catch | SEVA-ASR  RAGD-ASR | SURVIVE-BOTH (q%)")
print("  " + "-" * 92)
results = {}; ragd_flags = {}
order = ["cloneinject_n1", "cloneinject_n2", "cloneinject_n3", "cloneinject_n5", "cloneinject_n8", "templated"]
for name in order:
    if name not in data["configs"]:
        continue
    r, flat = eval_config(data["configs"][name]); results[name] = r; ragd_flags[name] = flat
    print(f"  {name:16} | {r['attempts']:8d} | {r['seva_catch_pct']:9.1f}% {r['ragd_catch_pct']:9.1f}% | "
          f"{r['seva_asr']:7.1f}% {r['ragd_asr']:8.1f}% | {r['survive_both_asr']:6.1f}% (q {r['q_survive_both_pct']:.0f}%)")
json.dump(ragd_flags, open(os.path.join(RES, "e4hh_ragd_flags_s042.json"), "w"))

fp = tot = 0
for q in data["benign"]:
    _, removed = D.defend(query=q["q"], R=q["R"], return_indices=True)
    fp += len(removed); tot += len(q["R"])
ragd_fpr = 100.0 * fp / tot if tot else 0.0
print(f"\n  RAGDefender benign doc-FPR (clean removed / clean retrieved over {len(data['benign'])} benign queries) = {ragd_fpr:.1f}%")
json.dump({"results": results, "ragd_benign_doc_fpr": ragd_fpr, "embedder": "BAAI/bge-large-en-v1.5", "task_type": "single_hop"},
          open(os.path.join(RES, "e4hh_s042.json"), "w"), indent=2)
print("  saved e4hh_s042.json")
