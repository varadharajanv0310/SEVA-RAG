#!/usr/bin/env python3
"""scale_xrun.py -- Experiment A: calibration scale points.

cluster_coh HARD GATE (composite NEVER scored), NON-oracle tau, 60/40 cal/eval split.
N in {10000, 100000} x density {1,5,10%} x seed {42,7,123}. Reuses the FROZEN primitives in
seva_xplat_common (doc_coh_full, retrieve_topk, embed_resumable, sha256_corpus_canonical) and the
deterministic xplat_poison_gen. Corpus identity GATED vs canonical 28ec3811 + fingerprint -> STOP on
mismatch. Poison hash-checked vs 4f7ee3f3. Emits result_scale{10k,100k}.json. See PREREG_SCALE.md.
Detector byte-frozen: nothing here touches cluster_coh / _compute_doc_coh / _score / constants.
"""
import os, sys, json, time, argparse, hashlib
import numpy as np
import seva_xplat_common as C
import xplat_poison_gen as PG
from hardgate_xrun import TQ          # frozen 52 targeted queries (first 50 used)

CANON = "28ec38114ee64e6010ec489d01e6d3ee13d9b3758fd30a169c99ed078732f8a9"
POISON_CANON = "4f7ee3f368cc6aae82180df261f4ee60bbd1f02b0834a4c4be72615ba68a733c"
CAL_FRAC = 0.60

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--out", default=r"D:\SEVA-RAG\a1_corpus")
    ap.add_argument("--cache", default=r"D:\SEVA-RAG\seva_cache_secxplat_scale")
    ap.add_argument("--fingerprint", default="corpus_fingerprint.txt")
    a = ap.parse_args(); os.makedirs(a.cache, exist_ok=True); N = a.N
    dev, dev_name, backend = C.pick_device("auto")
    corpus_path = os.path.join(a.out, "clean_corpus_security.json")
    benign_path = os.path.join(a.out, "benign_queries_security.json")

    # ---- corpus identity GATE (verify the 100k canonical; STOP on mismatch) ----
    chash, ndocs = C.sha256_corpus_canonical(corpus_path)
    fp = C.verify_against_fingerprint(corpus_path, a.fingerprint)
    hash_match = (chash == CANON) and (ndocs == 100000)
    C.log(f"corpus ordered-hash={chash} match={hash_match} fingerprint_ok={fp.get('ok')} n={ndocs}")
    if not (hash_match and fp.get("ok")):
        print(json.dumps({"FATAL": "CORPUS IDENTITY MISMATCH vs canonical 28ec3811 / fingerprint -- comparison invalid; STOP.",
                          "hash": chash, "hash_match": hash_match, "fingerprint": fp})); sys.exit(2)
    corpus = json.load(open(corpus_path, encoding="utf-8")); clean_full = [d["text"] for d in corpus]
    benign_pool = json.load(open(benign_path, encoding="utf-8"))

    # ---- poison regen + hash check ----
    poison_texts = [d["text"] for d in PG.generate_corpus(10000)]
    phash = C.sha256_ordered_texts(poison_texts); poison_match = (phash == POISON_CANON)
    C.log(f"poison regenerated: 10000 docs hash={phash[:16]}... match={poison_match}")
    if not poison_match:
        print(json.dumps({"FATAL": "POISON HASH MISMATCH vs 4f7ee3f3; STOP.", "hash": phash})); sys.exit(3)

    # ---- embed clean(100k) + poison(10k) ONCE (resumable, shared cache) ----
    clean_emb_full, _ = C.embed_resumable(clean_full, os.path.join(a.cache, "clean_emb.npy"), dev, label="clean")
    poison_emb, _ = C.embed_resumable(poison_texts, os.path.join(a.cache, "poison_emb.npy"), dev, label="poison")
    qcache = os.path.join(a.cache, "query_emb.npz")
    if os.path.exists(qcache):
        z = np.load(qcache); tq_emb, bq_emb = z["tq"], z["bq"]
    else:
        enc, _ = C.load_encoder(dev)
        tq_emb = np.ascontiguousarray(enc.encode(TQ, batch_size=C.BATCH, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
        bq_emb = np.ascontiguousarray(enc.encode(benign_pool, batch_size=C.BATCH, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
        np.savez(qcache, tq=tq_emb, bq=bq_emb); del enc
    adv_q_emb = tq_emb[:C.TARGETED_Q]

    # ---- subset to N (deterministic first-N of the hash-verified corpus) ----
    clean_emb = clean_emb_full[:N]; clean_texts = clean_full[:N]
    hsub = hashlib.sha256(); hsub.update(str(N).encode()); hsub.update(b"\n")
    for t in clean_texts: hsub.update(hashlib.sha256(t.encode("utf-8")).hexdigest().encode()); hsub.update(b"\n")
    subset_hash = hsub.hexdigest()

    # ---- grid (resumable) ----
    prog = os.path.join(a.cache, f"scale{N}_grid.json")
    grid = json.load(open(prog)) if os.path.exists(prog) else []
    done = {(g["density"], g["seed"]) for g in grid}
    for d in C.DENSITIES:
        P = int(round(N * d))
        full = np.ascontiguousarray(np.concatenate([poison_emb[:P], clean_emb[P:]], axis=0), dtype=np.float32)
        cohp = os.path.join(a.cache, f"scale{N}_coh_d{int(d*100):02d}.npy")
        if os.path.exists(cohp):
            coh = np.load(cohp); idx = C._faiss_index(full)
        else:
            C.log(f"N={N} d={d:.0%}: doc_coh over {N} docs (P={P})"); coh, idx = C.doc_coh_full(full); np.save(cohp, coh)
        pcoh, ccoh = coh[:P], coh[P:]
        gap = float(pcoh.mean() - ccoh.mean()); snr = float(gap / (ccoh.std() + 1e-9))
        adv_topk = C.retrieve_topk(idx, full, adv_q_emb)
        for s in C.SEEDS:
            if (d, s) in done: continue
            rng = np.random.default_rng(s)
            perm = rng.permutation(len(ccoh)); ncal = int(round(CAL_FRAC * len(ccoh)))
            cal_i, eval_i = perm[:ncal], perm[ncal:]
            tau = float(np.percentile(ccoh[cal_i], 100 - C.FPR_TARGET * 100))     # non-oracle, CAL-half clean only
            docfpr_eval = float(100.0 * np.mean(ccoh[eval_i] > tau))              # held-out doc-level FPR (convergence metric)
            att = reach = 0
            for tk in adv_topk:
                rp = [di for di in tk if di < P]
                if rp: reach += 1
                if rp and any(pcoh[di] <= tau for di in rp): att += 1
            asr = float(100.0 * att / max(1, len(adv_q_emb)))
            sel = rng.choice(len(benign_pool), size=min(C.BENIGN_Q, len(benign_pool)), replace=False)
            ben_topk = C.retrieve_topk(idx, full, np.ascontiguousarray(bq_emb[sel]))
            def flagged(di): return (pcoh[di] > tau) if di < P else (ccoh[di - P] > tau)
            clean_seen = sum(1 for tk in ben_topk for di in tk if di >= P)
            clean_fp = sum(1 for tk in ben_topk for di in tk if di >= P and ccoh[di - P] > tau)
            docfpr_ret = float(100.0 * clean_fp / max(1, clean_seen))
            qfpr1 = float(100.0 * np.mean([any(flagged(di) for di in tk) for tk in ben_topk]))
            qfpr2 = float(100.0 * np.mean([sum(flagged(di) for di in tk) >= 2 for tk in ben_topk]))
            cell = {"N": N, "density": d, "seed": s, "n_poison": int(P), "n_clean": int(N - P),
                    "n_clean_cal": int(ncal), "n_clean_eval": int(len(ccoh) - ncal),
                    "gap": gap, "snr": snr, "tau": tau, "asr_pct": asr,
                    "poison_reach_pct": float(100.0 * reach / max(1, len(adv_q_emb))),
                    "docfpr_eval_doclevel_pct": docfpr_eval, "docfpr_benign_retrieval_pct": docfpr_ret,
                    "query_fpr_ge1_pct": qfpr1, "query_fpr_ge2_pct": qfpr2}
            grid.append(cell); json.dump(grid, open(prog, "w"))
            C.log(f"  N={N} [{d:.0%} seed {s}] gap={gap:+.4f} snr={snr:.2f} asr={asr:.1f}% docfpr_eval={docfpr_eval:.3f}% tau={tau:.4f} n_eval={len(ccoh)-ncal}")

    gmean = float(np.mean([g["docfpr_eval_doclevel_pct"] for g in grid]))
    gpd = {}; [gpd.__setitem__(g["density"], g["gap"]) for g in grid]
    gv = list(gpd.values()); gap_inv = (max(gv) - min(gv) <= 0.05) and all(v > 0.15 for v in gv)
    asr0 = all(g["asr_pct"] == 0.0 for g in grid)
    out = {"experiment": "A-calibration-scale", "N": N,
           "scale_summary": {"grand_mean_docfpr_eval_pct": gmean, "fpr_target_pct": C.FPR_TARGET * 100,
                             "abs_dev_from_target_pct": abs(gmean - C.FPR_TARGET * 100)},
           "corpus": {"canonical_sha256": (CANON if N == 100000 else subset_hash), "parent_canonical_sha256": CANON,
                      "hash_match": True, "fingerprint_check": fp, "n_docs": N, "subset_of_100k": (N != 100000)},
           "detector": {"type": "cluster_coh_hard_gate", "composite_used": False,
                        "calibration": "NON-ORACLE: tau=(1-FPR_TARGET) pctile of CAL-half clean coh; 60/40 cal/eval split per seed",
                        "K": C.K, "K_FETCH": C.K_FETCH, "FPR_TARGET": C.FPR_TARGET, "frozen_math": True},
           "poison": {"count": 10000, "ordered_sha256": phash, "hash_match": poison_match},
           "seeds": C.SEEDS, "densities": C.DENSITIES,
           "verdict": {"gap_per_density": gpd, "gap_density_invariant": bool(gap_inv), "templated_asr_zero": bool(asr0)},
           "grid": grid, "env": C.env_report(), "finished_unix": int(time.time())}
    label = f"{N // 1000}k"
    json.dump(out, open(f"result_scale{label}.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps({"N": N, "result_file": f"result_scale{label}.json", "hash_match": True,
                      "gap_per_density": {f"{k:.0%}": round(v, 4) for k, v in gpd.items()},
                      "gap_density_invariant": bool(gap_inv), "templated_asr_zero": bool(asr0),
                      "grand_mean_docfpr_eval_pct": round(gmean, 4), "fpr_target_pct": 0.69}, indent=2))

if __name__ == "__main__":
    main()
