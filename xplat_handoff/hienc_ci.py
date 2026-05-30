#!/usr/bin/env python3
"""hienc_ci.py -- Experiment B: high-encounter Wilson CI on templated-poison evasion at the FROZEN
cluster_coh hard gate. ~25,000 templated-poison gate encounters on the 100k canonical corpus.

Encounter def: each of N_ENC=25,000 deterministic templated-poison docs (xplat_poison_gen, hash-pinned
construction) is injected at corpus[0:N_ENC] of the hash-verified 100k canonical corpus and scored by
the frozen hard gate (coh <= tau_coh = an EVASION). tau_coh is NON-oracle = (1-FPR_TARGET) percentile of
the CLEAN remainder's coh. This yields n=25k independent gate evaluations of the templated-poison catch
rate -> tight Wilson upper bound. (Density is 25% by construction; templated poison clusters tightly at
every density, so this is conservative toward catch; retrieval reach by the 50 TQ reported for context.)
Detector byte-frozen; composite NEVER scored. See PREREG_SCALE.md.
"""
import os, sys, json, time
import numpy as np
import seva_xplat_common as C
import xplat_poison_gen as PG
from hardgate_xrun import TQ

CANON = "28ec38114ee64e6010ec489d01e6d3ee13d9b3758fd30a169c99ed078732f8a9"
N_ENC = 25000

def wilson_upper(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return 100.0 * (centre + half)

def main():
    out = r"D:\SEVA-RAG\a1_corpus"; cache = r"D:\SEVA-RAG\seva_cache_secxplat_scale"; os.makedirs(cache, exist_ok=True)
    dev, dev_name, backend = C.pick_device("auto")
    corpus_path = os.path.join(out, "clean_corpus_security.json")
    chash, ndocs = C.sha256_corpus_canonical(corpus_path)
    fp = C.verify_against_fingerprint(corpus_path, "corpus_fingerprint.txt")
    if not (chash == CANON and ndocs == 100000 and fp.get("ok")):
        print(json.dumps({"FATAL": "CORPUS IDENTITY MISMATCH vs canonical 28ec3811 / fingerprint; STOP.", "hash": chash, "fp": fp})); sys.exit(2)
    corpus = json.load(open(corpus_path, encoding="utf-8")); clean_texts = [d["text"] for d in corpus]
    poison_texts = [d["text"] for d in PG.generate_corpus(N_ENC)]
    phash = C.sha256_ordered_texts(poison_texts)
    C.log(f"Exp B: {len(poison_texts)} templated poison (hash {phash[:16]}...); corpus hash_match={chash==CANON}")
    clean_emb, _ = C.embed_resumable(clean_texts, os.path.join(cache, "clean_emb.npy"), dev, label="clean")     # reuse Exp-A cache
    poison_emb, _ = C.embed_resumable(poison_texts, os.path.join(cache, "poison_emb_25k.npy"), dev, label="poison25k")
    P = N_ENC
    full = np.ascontiguousarray(np.concatenate([poison_emb[:P], clean_emb[P:]], axis=0), dtype=np.float32)
    C.log(f"Exp B: doc_coh over {len(full)} docs (P={P} poison, {len(full)-P} clean)")
    coh, idx = C.doc_coh_full(full)
    pcoh, ccoh = coh[:P], coh[P:]
    tau = float(np.percentile(ccoh, 100 - C.FPR_TARGET * 100))            # FROZEN non-oracle tau (clean only)
    evasions = int(np.sum(pcoh <= tau)); n = int(P)
    asr = 100.0 * evasions / n; wu = wilson_upper(evasions, n)
    # retrieval reach (context): poison retrieved in top-K by the 50 TQ
    qcache = os.path.join(cache, "query_emb.npz")
    if os.path.exists(qcache):
        tq_emb = np.load(qcache)["tq"]
    else:
        enc, _ = C.load_encoder(dev); tq_emb = np.ascontiguousarray(enc.encode(TQ, batch_size=C.BATCH, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32); del enc
    adv_topk = C.retrieve_topk(idx, full, tq_emb[:C.TARGETED_Q])
    retr_poison = int(sum(1 for tk in adv_topk for di in tk if di < P))
    out_j = {"experiment": "B-high-encounter-CI", "n_encounters": n, "evasions": evasions, "asr_pct": asr,
             "wilson_upper_pct": wu, "wilson_z": 1.96, "tau_coh": tau,
             "poison_coh_min": float(pcoh.min()), "poison_coh_mean": float(pcoh.mean()), "clean_coh_mean": float(ccoh.mean()),
             "density_pct": 100.0 * P / len(full),
             "corpus": {"canonical_sha256": chash, "hash_match": chash == CANON, "fingerprint_check": fp, "n_docs": ndocs},
             "detector": {"type": "cluster_coh_hard_gate", "composite_used": False,
                          "calibration": "NON-ORACLE: tau=(1-FPR_TARGET) pctile of clean coh", "FPR_TARGET": C.FPR_TARGET, "frozen_math": True},
             "poison": {"count": n, "ordered_sha256": phash, "source": "xplat_poison_gen.generate_corpus(25000) deterministic"},
             "encounter_def": "each of 25k templated-poison docs scored by the frozen hard gate (coh<=tau=evasion); injected at corpus[0:25000] of the 100k canonical corpus; tau from clean remainder.",
             "retrieval_context": {"targeted_queries": C.TARGETED_Q, "poison_retrieved_in_topk": retr_poison, "note": "context only; the n=25k CI is over per-doc gate evaluations"},
             "env": C.env_report(), "finished_unix": int(time.time())}
    json.dump(out_j, open("result_hienc_ci.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps({"n_encounters": n, "evasions": evasions, "asr_pct": asr, "wilson_upper_pct": round(wu, 5),
                      "tau_coh": round(tau, 4), "poison_coh_min": round(float(pcoh.min()), 4),
                      "result_file": "result_hienc_ci.json"}, indent=2))

if __name__ == "__main__":
    main()
