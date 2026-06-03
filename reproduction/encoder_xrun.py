#!/usr/bin/env python3
"""encoder_xrun.py -- Encoder-generalization of the cluster_coh hard gate.

QUESTION (go/no-go for an encoder-generalization claim): does cluster_coh detect the
GEOMETRY of templating, or just bge-large's manifold?

FROZEN: the detector (cluster_coh via seva_xplat_common.doc_coh_full / retrieve_topk /
non-oracle tau / K=5 / K_FETCH=20 / HNSW M=32 / FPR_TARGET=0.0069); the corpus+poison
TEXT (canonical 28ec3811 + fingerprint; poison 4f7ee3f3); the machine (5080). VARIED:
the embedding ENCODER only.

This is scale_xrun.py at N=100000 with ONE change: the bge embed is swapped for the
encoder-parameterized embed in encoder_config (which applies each encoder's CORRECT
convention -- cluster_coh is SYMMETRIC, so e5 uses 'query: ' on ALL texts). Every line
of the DETECTOR (doc_coh_full, retrieve_topk, non-oracle tau, the 60/40 split, the
grid math) is the SAME frozen seva_xplat_common code scale_xrun used -- nothing here
touches it.

Run order (see PROMPT_ENCODER.md / PREREG_ENCODER.md):
  1) python encoder_xrun.py --encoder bge   # harness gate: MUST reproduce result_scale100k.json
  2) python encoder_xrun.py --encoder e5    # the test; read result straight vs PREREG bands
  3) python encoder_xrun.py --encoder gte   # ONLY if e5 PASSES
Emits result_encoder_<key>.json with a self-verdict PASS / WEAK / FAIL.
10k scale is deliberately NOT run here (10k x 1% = P=100 is the small-count boundary,
an axis orthogonal to the encoder question).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import seva_xplat_common as C
import xplat_poison_gen as PG
import encoder_config as EC
from hardgate_xrun import TQ          # frozen 52 targeted queries (first C.TARGETED_Q used)

CANON = "28ec38114ee64e6010ec489d01e6d3ee13d9b3758fd30a169c99ed078732f8a9"
POISON_CANON = "4f7ee3f368cc6aae82180df261f4ee60bbd1f02b0834a4c4be72615ba68a733c"
CAL_FRAC = 0.60
N = 100000   # encoder test runs at the paper scale only


def self_verdict(grid):
    """PASS / WEAK / FAIL per PREREG_ENCODER.md -- computed here so the JSON carries the call."""
    asr_vals = [g["asr_pct"] for g in grid]
    snr_min = min(g["snr"] for g in grid)
    docfpr_grand = float(np.mean([g["docfpr_eval_doclevel_pct"] for g in grid]))
    gpd = {}
    for g in grid:
        gpd[g["density"]] = g["gap"]          # gap is seed-invariant (a corpus property)
    gv = list(gpd.values())
    gap_mean = float(np.mean(gv)) if gv else 0.0
    gap_range = float(max(gv) - min(gv)) if gv else 9.9
    range_rel = (gap_range / gap_mean) if gap_mean else 9.9
    P1 = all(a == 0.0 for a in asr_vals)
    P2 = snr_min >= 3.0
    P3 = range_rel <= 0.25
    P4 = docfpr_grand <= 1.5
    if P1 and P2 and P3 and P4:
        result = "PASS"
    elif (max(asr_vals) > 10.0) or (snr_min < 1.5) or (docfpr_grand > 3.0):
        result = "FAIL"
    else:
        result = "WEAK"
    return {
        "result": result,
        "criteria_PASS_all_required": {
            "P1_asr_zero_all9": bool(P1), "P2_snr_min_ge_3.0": bool(P2),
            "P3_gap_range_rel_le_0.25": bool(P3), "P4_grand_docfpr_le_1.5": bool(P4)},
        "measured": {
            "asr_max_pct": float(max(asr_vals)), "snr_min": round(float(snr_min), 3),
            "gap_per_density": {f"{k:.0%}": round(v, 4) for k, v in gpd.items()},
            "gap_range_rel": round(range_rel, 4),
            "grand_mean_docfpr_eval_pct": round(docfpr_grand, 4)},
        "bands": "PASS = P1..P4 all true; FAIL if asr_max>10% or snr_min<1.5 or grand_docfpr>3%; "
                 "else WEAK. Read straight -- see PREREG_ENCODER.md.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(EC.ENCODERS.keys()))
    ap.add_argument("--out", default=r"D:\SEVA-RAG\a1_corpus")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--fingerprint", default="corpus_fingerprint.txt")
    a = ap.parse_args()
    key = a.encoder
    cache_root = a.cache or rf"D:\SEVA-RAG\seva_cache_enc_{key}"
    os.makedirs(cache_root, exist_ok=True)
    dev, dev_name, backend = C.pick_device("auto")
    corpus_path = os.path.join(a.out, "clean_corpus_security.json")
    benign_path = os.path.join(a.out, "benign_queries_security.json")
    cfg = EC.ENCODERS[key]
    C.log(f"ENCODER={key} model={cfg['model']} doc_prefix={cfg['doc_prefix']!r} "
          f"query_prefix={cfg['query_prefix']!r} device={backend} ({dev_name})")

    # ---- corpus identity GATE (same canonical TEXT; only embeddings are encoder-new) ----
    chash, ndocs = C.sha256_corpus_canonical(corpus_path)
    fp = C.verify_against_fingerprint(corpus_path, a.fingerprint)
    hash_match = (chash == CANON) and (ndocs == 100000)
    C.log(f"corpus ordered-hash={chash} match={hash_match} fingerprint_ok={fp.get('ok')} n={ndocs}")
    if not (hash_match and fp.get("ok")):
        print(json.dumps({"FATAL": "CORPUS IDENTITY MISMATCH vs canonical 28ec3811 / fingerprint -- STOP.",
                          "hash": chash, "hash_match": hash_match, "fingerprint": fp})); sys.exit(2)
    corpus = json.load(open(corpus_path, encoding="utf-8")); clean_full = [d["text"] for d in corpus]
    benign_pool = json.load(open(benign_path, encoding="utf-8"))

    # ---- poison regen + hash check (text identical; the encoder re-embeds it) ----
    poison_texts = [d["text"] for d in PG.generate_corpus(10000)]
    phash = C.sha256_ordered_texts(poison_texts); poison_match = (phash == POISON_CANON)
    C.log(f"poison regenerated: 10000 docs hash={phash[:16]}... match={poison_match}")
    if not poison_match:
        print(json.dumps({"FATAL": "POISON HASH MISMATCH vs 4f7ee3f3; STOP.", "hash": phash})); sys.exit(3)

    # ---- load encoder + SANITY GATE (catch a misconfigured encoder BEFORE the long embed) ----
    model = EC.load_model(key, dev)
    try:
        revision = EC.resolve_revision(key)
    except Exception as e:
        revision = f"unresolved ({e})"
    sanity = EC.sanity_report(key, model, dev, clean_full[:8], benign_pool[:8])
    C.log("SANITY " + json.dumps(sanity))
    if not (sanity["dim_ok"] and sanity["normalized_ok"] and sanity["retrieval_sane"]):
        print(json.dumps({"FATAL": "ENCODER SANITY FAILED -- misconfigured embedding; FIX before trusting any number.",
                          "encoder": key, "sanity": sanity})); sys.exit(4)

    # ---- embed with the encoder's CORRECT convention (role -> prefix; cache per-encoder) ----
    clean_emb = EC.embed_enc(key, clean_full, "doc", "clean", model, cache_root, dev)
    poison_emb = EC.embed_enc(key, poison_texts, "doc", "poison", model, cache_root, dev)
    tq_emb = EC.embed_enc(key, list(TQ), "query", "tq", model, cache_root, dev)
    bq_emb = EC.embed_enc(key, list(benign_pool), "query", "bquery", model, cache_root, dev)
    adv_q_emb = tq_emb[:C.TARGETED_Q]

    # ---- grid (3 densities x 3 seeds) -- FROZEN detector, identical to scale_xrun.py ----
    prog = os.path.join(cache_root, f"enc_{key}_grid.json")
    grid = json.load(open(prog)) if os.path.exists(prog) else []
    done = {(g["density"], g["seed"]) for g in grid}
    for d in C.DENSITIES:
        P = int(round(N * d))
        full = np.ascontiguousarray(np.concatenate([poison_emb[:P], clean_emb[P:]], axis=0), dtype=np.float32)
        cohp = os.path.join(cache_root, f"enc_{key}_coh_d{int(d * 100):02d}.npy")
        if os.path.exists(cohp):
            coh = np.load(cohp); idx = C._faiss_index(full)
        else:
            C.log(f"{key} d={d:.0%}: doc_coh over {N} docs (P={P})")
            coh, idx = C.doc_coh_full(full); np.save(cohp, coh)
        pcoh, ccoh = coh[:P], coh[P:]
        gap = float(pcoh.mean() - ccoh.mean()); snr = float(gap / (ccoh.std() + 1e-9))
        adv_topk = C.retrieve_topk(idx, full, adv_q_emb)
        for s in C.SEEDS:
            if (d, s) in done:
                continue
            rng = np.random.default_rng(s)
            perm = rng.permutation(len(ccoh)); ncal = int(round(CAL_FRAC * len(ccoh)))
            cal_i, eval_i = perm[:ncal], perm[ncal:]
            tau = float(np.percentile(ccoh[cal_i], 100 - C.FPR_TARGET * 100))   # non-oracle, CAL-half clean only
            docfpr_eval = float(100.0 * np.mean(ccoh[eval_i] > tau))            # held-out doc-level FPR
            att = reach = 0
            for tk in adv_topk:
                rp = [di for di in tk if di < P]
                if rp:
                    reach += 1
                if rp and any(pcoh[di] <= tau for di in rp):
                    att += 1
            asr = float(100.0 * att / max(1, len(adv_q_emb)))
            sel = rng.choice(len(benign_pool), size=min(C.BENIGN_Q, len(benign_pool)), replace=False)
            ben_topk = C.retrieve_topk(idx, full, np.ascontiguousarray(bq_emb[sel]))

            def flagged(di):
                return (pcoh[di] > tau) if di < P else (ccoh[di - P] > tau)
            clean_seen = sum(1 for tk in ben_topk for di in tk if di >= P)
            clean_fp = sum(1 for tk in ben_topk for di in tk if di >= P and ccoh[di - P] > tau)
            docfpr_ret = float(100.0 * clean_fp / max(1, clean_seen))
            qfpr1 = float(100.0 * np.mean([any(flagged(di) for di in tk) for tk in ben_topk]))
            qfpr2 = float(100.0 * np.mean([sum(flagged(di) for di in tk) >= 2 for tk in ben_topk]))
            cell = {"N": N, "density": d, "seed": s, "n_poison": int(P), "n_clean": int(N - P),
                    "n_clean_cal": int(ncal), "n_clean_eval": int(len(ccoh) - ncal),
                    "clean_coh_mean": float(ccoh.mean()), "clean_coh_std": float(ccoh.std()),
                    "poison_coh_mean": float(pcoh.mean()),
                    "gap": gap, "snr": snr, "tau": tau, "asr_pct": asr,
                    "poison_reach_pct": float(100.0 * reach / max(1, len(adv_q_emb))),
                    "docfpr_eval_doclevel_pct": docfpr_eval, "docfpr_benign_retrieval_pct": docfpr_ret,
                    "query_fpr_ge1_pct": qfpr1, "query_fpr_ge2_pct": qfpr2}
            grid.append(cell); json.dump(grid, open(prog, "w"))
            C.log(f"  {key} [{d:.0%} seed {s}] gap={gap:+.4f} snr={snr:.2f} asr={asr:.1f}% "
                  f"docfpr_eval={docfpr_eval:.3f}% clean_coh={ccoh.mean():.4f} poison_coh={pcoh.mean():.4f} tau={tau:.4f}")

    vd = self_verdict(grid)
    out = {"experiment": "encoder-generalization (cluster_coh hard gate; N=100000)",
           "machine_label": "5080",
           "encoder": {"key": key, "model": cfg["model"], "revision": revision, "dim": cfg["dim"],
                       "doc_prefix": cfg["doc_prefix"], "query_prefix": cfg["query_prefix"],
                       "normalize": cfg["normalize"], "convention_note": cfg["note"]},
           "detector": {"type": "cluster_coh_hard_gate", "composite_used": False, "frozen_math": True,
                        "calibration": "NON-ORACLE: tau=(1-FPR_TARGET) pctile of CAL-half clean coh; 60/40 split per seed",
                        "K": C.K, "K_FETCH": C.K_FETCH, "FPR_TARGET": C.FPR_TARGET},
           "corpus": {"canonical_sha256": chash, "reference_sha256": CANON, "hash_match": bool(hash_match),
                      "fingerprint_check": fp, "n_docs": ndocs},
           "poison": {"count": 10000, "ordered_sha256": phash, "reference_sha256": POISON_CANON,
                      "hash_match": bool(poison_match)},
           "embedding_backend": backend, "sanity": sanity,
           "seeds": C.SEEDS, "densities": C.DENSITIES, "grid": grid,
           "verdict": vd, "env": C.env_report(), "finished_unix": int(time.time())}
    fn = f"result_encoder_{key}.json"
    json.dump(out, open(fn, "w", encoding="utf-8"), indent=2)
    print(json.dumps({"encoder": key, "model": cfg["model"], "result_file": fn,
                      "corpus_hash_match": bool(hash_match), "poison_hash_match": bool(poison_match),
                      "verdict": vd["result"], "measured": vd["measured"]}, indent=2))


if __name__ == "__main__":
    main()
