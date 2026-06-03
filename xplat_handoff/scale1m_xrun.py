#!/usr/bin/env python3
"""scale1m_xrun.py -- 1M corpus-scale run (extends scale_xrun.py). FROZEN detector (seva_xplat_common
doc_coh_full / retrieve_topk / non-oracle tau / K=5 / K_FETCH=20 / HNSW M=32 / FPR_TARGET=0.0069).
cluster_coh HARD GATE, 60/40 cal/eval, seeds 42/7/123. Densities the 96k poison cap can fill at 1M:
1% (P=10k), 5% (P=50k); 10% (P=100k>96k) SKIPPED + documented. Measures latency + calibration FPR +
detection at N=1,000,000. Memory-careful (memmap embeddings; index build peaks ~8GB of 14GB free).
Resumable (done markers / coh caches / grid progress), heartbeat, fail-loud. Emits result_1M.json.

Run: python scale1m_xrun.py   (resumable -- re-run the SAME command to resume from the furthest checkpoint)
The detector math is NOT touched; only embed I/O is memmap'd for scale.
"""
import os, sys, json, time, hashlib, traceback
import numpy as np
import seva_xplat_common as C
import xplat_poison_gen as PG
from hardgate_xrun import TQ

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"D:\SEVA-RAG\a1_corpus_1m"; CACHE = r"D:\SEVA-RAG\seva_cache_1m"; os.makedirs(CACHE, exist_ok=True)
POISON_CANON = "4f7ee3f368cc6aae82180df261f4ee60bbd1f02b0834a4c4be72615ba68a733c"  # generate_corpus(10000)
RESULT = os.path.join(HERE, "result_1M.json")
CAL_FRAC = 0.60; DENSITIES = [0.01, 0.05]; SEEDS = C.SEEDS; BENCH_N = 200000
POISON_MAX = PG.max_unique_docs()  # 96000
DIM = C.EMB_DIM; CHUNK = 2000

def fatal(msg, extra=None):
    rec = {"FATAL": str(msg), "extra": extra, "unix": int(time.time())}
    try: json.dump(rec, open(RESULT, "w"), indent=2)
    except Exception: pass
    C.log("FATAL: " + str(msg)); sys.exit(2)

def hb(phase, frac, t0):
    C.log(f"HEARTBEAT phase={phase} progress={frac*100:.1f}% elapsed={time.time()-t0:.0f}s")

def free_gb():
    try:
        import psutil; return psutil.virtual_memory().available / 1e9
    except Exception:
        return -1.0

def embed_memmap(texts, dat, dev, label, t0, limit=None):
    """Chunked, resumable, memmap-backed embed. Writes each chunk to a (N,DIM) memmap on disk + a done
    marker. Returns (read-only memmap, seconds_spent_this_call). limit caps how far to embed (benchmark)."""
    N = len(texts); cap = N if limit is None else min(limit, N)
    done_path = dat + ".done"
    done = int(open(done_path).read().strip()) if os.path.exists(done_path) else 0
    mode = "r+" if os.path.exists(dat) else "w+"
    mm = np.memmap(dat, dtype=np.float32, mode=mode, shape=(N, DIM))
    if done >= cap:
        return np.memmap(dat, dtype=np.float32, mode="r", shape=(N, DIM)), 0.0
    enc, _ = C.load_encoder(dev); t_e = time.time()
    for s0 in range(done, cap, CHUNK):
        e0 = min(s0 + CHUNK, cap)
        emb = enc.encode(texts[s0:e0], batch_size=C.BATCH, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        mm[s0:e0] = emb; mm.flush()
        with open(done_path, "w") as f: f.write(str(e0))
        if ((e0 // CHUNK) % 10 == 0) or e0 == cap: hb(f"embed_{label}", e0 / cap, t0)
    spent = time.time() - t_e; del enc; mm.flush()
    return np.memmap(dat, dtype=np.float32, mode="r", shape=(N, DIM)), spent

def build_full_memmap(poison_emb, clean_mm, P, N, dat):
    """full = [poison[:P] ; clean[P:]] as a (N,DIM) memmap on disk (resumable). Returns read-only memmap."""
    if os.path.exists(dat + ".ok"):
        return np.memmap(dat, dtype=np.float32, mode="r", shape=(N, DIM))
    mm = np.memmap(dat, dtype=np.float32, mode="w+", shape=(N, DIM))
    mm[:P] = poison_emb[:P]
    step = 50000
    for s0 in range(P, N, step):
        e0 = min(s0 + step, N); mm[s0:e0] = clean_mm[s0:e0]; mm.flush()
    mm.flush(); open(dat + ".ok", "w").write("1")
    return np.memmap(dat, dtype=np.float32, mode="r", shape=(N, DIM))

def main():
    t_start = time.time()
    cp = os.path.join(OUT, "clean_corpus_1m.json"); bp = os.path.join(OUT, "benign_queries_1m.json")
    pp = os.path.join(OUT, "build_provenance_1m.json"); fpp = os.path.join(OUT, "corpus_fingerprint_1m.txt")
    if not (os.path.exists(cp) and os.path.exists(pp) and os.path.exists(fpp)):
        fatal("1M corpus/provenance/fingerprint missing -- run build_1m_corpus.py first")
    dev, dev_name, backend = C.pick_device("auto")
    prov = json.load(open(pp))
    # ---- corpus integrity GATE (hash == provenance; fingerprint doc-by-doc) ----
    C.log("PHASE 1  corpus integrity gate (hashing 1M docs) ...")
    chash, ndocs = C.sha256_corpus_canonical(cp)
    if chash != prov.get("corpus_canonical_sha256"):
        fatal("CORPUS HASH != provenance (corpus changed/corrupted since build)",
              {"hash": chash, "provenance": prov.get("corpus_canonical_sha256")})
    fp = C.verify_against_fingerprint(cp, fpp)
    if not fp.get("ok"):
        fatal("FINGERPRINT mismatch (corpus diverges from build)", fp)
    N = ndocs
    C.log(f"PHASE 1  GATE OK | N={N} hash={chash} fingerprint_ok=True | RAM free={free_gb():.1f}GB device={backend}")
    corpus = json.load(open(cp, encoding="utf-8")); clean_texts = [d["text"] for d in corpus]; del corpus
    benign_pool = json.load(open(bp, encoding="utf-8"))

    # ---- poison: generate up to max P (5% of N), hash-check the canonical first-10k ----
    maxP = int(round(N * max(DENSITIES)))
    npoison = min(maxP, POISON_MAX); poison_capped = maxP > POISON_MAX
    poison_texts = [d["text"] for d in PG.generate_corpus(npoison)]
    if C.sha256_ordered_texts(poison_texts[:10000]) != POISON_CANON:
        fatal("POISON HASH MISMATCH vs canonical 4f7ee3f3 on first 10k")
    C.log(f"PHASE 1  poison: generated {len(poison_texts)} (cap {POISON_MAX}); canonical first-10k hash OK")
    # which densities are fillable
    dens_ok, dens_skip = [], []
    for d in DENSITIES:
        (dens_ok if int(round(N * d)) <= len(poison_texts) else dens_skip).append(d)
    if int(round(N * 0.10)) > POISON_MAX: dens_skip_10 = True
    C.log(f"PHASE 1  densities fillable={dens_ok} skipped(>{POISON_MAX} poison)={dens_skip} ; 10% always skipped at 1M (P=100k>96k)")

    # ---- embed: 200k micro-benchmark FIRST, then full, then poison ----
    clean_dat = os.path.join(CACHE, "clean_emb_1m.dat")
    C.log("PHASE 2  micro-benchmark: embedding first 200k ...")
    clean_mm, t_e200 = embed_memmap(clean_texts, clean_dat, dev, "clean", t_start, limit=BENCH_N)
    if t_e200 == 0.0:  # already embedded past 200k on resume -> approximate from nothing; reuse stored bench if present
        t_e200 = -1.0
    benchf = os.path.join(CACHE, "bench.json")
    if not os.path.exists(benchf) and t_e200 > 0:
        C.log("PHASE 2  benchmark: HNSW index + doc_coh over 200k ...")
        x200 = np.ascontiguousarray(clean_mm[:BENCH_N])
        ti = time.time(); idx200 = C._faiss_index(x200); t_i200 = time.time() - ti
        tc = time.time(); coh200, _ = C.doc_coh_full(x200); t_c200 = time.time() - tc
        del x200, idx200, coh200
        # extrapolate: embed ~ N ; HNSW build ~ N log N ; coh ~ N log N
        import math
        sf = (N / BENCH_N) * (math.log(N) / math.log(BENCH_N))   # N log N scale factor 200k->1M
        lin = N / BENCH_N
        est_embed = t_e200 * lin
        est_idx = t_i200 * sf; est_coh = t_c200 * sf
        per_density = est_idx + est_coh
        est_total = est_embed + per_density * len(dens_ok) + 600  # +grid/latency slack
        bench = {"t_embed_200k_s": t_e200, "t_hnsw_200k_s": t_i200, "t_coh_200k_s": t_c200,
                 "scale_factor_NlogN": sf, "scale_linear": lin, "n_fillable_densities": len(dens_ok),
                 "EST_embed_1M_s": est_embed, "EST_per_density_index+coh_s": per_density,
                 "EST_total_s": est_total, "EST_total_h": est_total / 3600.0,
                 "dominant_term": "HNSW+coh (N log N)" if per_density * len(dens_ok) > est_embed else "embed (linear)"}
        json.dump(bench, open(benchf, "w"), indent=2)
        C.log(f"PHASE 2  RUNTIME ESTIMATE -> total ~{est_total/3600.0:.1f} h  "
              f"(embed ~{est_embed/3600.0:.1f}h, per-density idx+coh ~{per_density/3600.0:.2f}h x{len(dens_ok)}, "
              f"dominant={bench['dominant_term']}) | 200k: embed {t_e200:.0f}s hnsw {t_i200:.0f}s coh {t_c200:.0f}s")
    elif os.path.exists(benchf):
        bench = json.load(open(benchf)); C.log(f"PHASE 2  benchmark cached: est total ~{bench['EST_total_h']:.1f}h")
    else:
        bench = {"note": "benchmark skipped (resumed past 200k)"}
    C.log("PHASE 2  embedding remaining clean to 1M ...")
    clean_mm, _ = embed_memmap(clean_texts, clean_dat, dev, "clean", t_start)  # resumes from 200k -> N
    del clean_texts
    poison_dat = os.path.join(CACHE, "poison_emb_1m.dat")
    poison_mm, _ = embed_memmap(poison_texts, poison_dat, dev, "poison", t_start)
    poison_emb = np.ascontiguousarray(poison_mm[:len(poison_texts)])
    # query embeddings (50 targeted + benign), embedded once
    qcache = os.path.join(CACHE, "query_emb.npz")
    if os.path.exists(qcache):
        z = np.load(qcache); tq_emb, bq_emb = z["tq"], z["bq"]
    else:
        enc, _ = C.load_encoder(dev)
        tq_emb = np.ascontiguousarray(enc.encode(list(TQ), batch_size=C.BATCH, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
        bq_emb = np.ascontiguousarray(enc.encode(list(benign_pool), batch_size=C.BATCH, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
        np.savez(qcache, tq=tq_emb, bq=bq_emb); del enc
    adv_q_emb = tq_emb[:C.TARGETED_Q]
    C.log(f"PHASE 2  embeddings done | RAM free={free_gb():.1f}GB")

    # ---- PHASE 3: per-density grid (fillable densities only) ----
    prog = os.path.join(CACHE, "grid_1m.json")
    grid = json.load(open(prog)) if os.path.exists(prog) else []
    done = {(g["density"], g["seed"]) for g in grid}
    last_idx = None; last_full = None
    for d in dens_ok:
        P = int(round(N * d))
        C.log(f"PHASE 3  density {d:.0%} (P={P}) | RAM free={free_gb():.1f}GB | building full memmap ...")
        full_dat = os.path.join(CACHE, f"full_d{int(d*100):02d}.dat")
        full = build_full_memmap(poison_emb, clean_mm, P, N, full_dat)
        cohp = os.path.join(CACHE, f"coh_d{int(d*100):02d}.npy")
        try:
            if os.path.exists(cohp):
                coh = np.load(cohp); idx = C._faiss_index(np.ascontiguousarray(full))
            else:
                C.log(f"PHASE 3  density {d:.0%}: doc_coh over {N} docs (HNSW build + K=5 search) ...")
                tco = time.time(); coh, idx = C.doc_coh_full(np.ascontiguousarray(full)); np.save(cohp, coh)
                C.log(f"PHASE 3  density {d:.0%}: doc_coh done in {time.time()-tco:.0f}s")
        except MemoryError as e:
            fatal(f"OOM during index/doc_coh at density {d:.0%} (N={N})", {"err": repr(e), "ram_free_gb": free_gb()})
        last_idx = idx; last_full = full
        pcoh, ccoh = coh[:P], coh[P:]
        gap = float(pcoh.mean() - ccoh.mean()); snr = float(gap / (ccoh.std() + 1e-9))
        adv_topk = C.retrieve_topk(idx, full, adv_q_emb)
        for s in SEEDS:
            if (d, s) in done: continue
            rng = np.random.default_rng(s)
            perm = rng.permutation(len(ccoh)); ncal = int(round(CAL_FRAC * len(ccoh)))
            cal_i, eval_i = perm[:ncal], perm[ncal:]
            tau = float(np.percentile(ccoh[cal_i], 100 - C.FPR_TARGET * 100))
            docfpr_eval = float(100.0 * np.mean(ccoh[eval_i] > tau))
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
            cell = {"N": N, "density": d, "seed": s, "n_poison": int(P), "n_clean": int(N - P),
                    "n_clean_eval": int(len(ccoh) - ncal), "gap": gap, "snr": snr, "tau": tau,
                    "asr_pct": asr, "poison_evasion_pct": asr, "poison_reach_pct": float(100.0 * reach / max(1, len(adv_q_emb))),
                    "docfpr_eval_doclevel_pct": docfpr_eval, "docfpr_benign_retrieval_pct": docfpr_ret, "query_fpr_ge1_pct": qfpr1,
                    "clean_coh_mean": float(ccoh.mean()), "poison_coh_mean": float(pcoh.mean())}
            grid.append(cell); json.dump(grid, open(prog, "w"))
            C.log(f"  [{d:.0%} seed {s}] gap={gap:+.4f} snr={snr:.2f} evasion={asr:.1f}% docfpr_eval={docfpr_eval:.3f}% tau={tau:.4f}")

    # ---- PHASE 4: latency (encode 1 query + FAISS retrieve + gate-check) on the last 1M index ----
    C.log("PHASE 4  per-query latency at 1M ...")
    if last_idx is None:  # fully resumed -> rebuild a 1M index at the largest fillable density
        d = dens_ok[-1]; P = int(round(N * d)); full = build_full_memmap(poison_emb, clean_mm, P, N, os.path.join(CACHE, f"full_d{int(d*100):02d}.dat"))
        last_idx = C._faiss_index(np.ascontiguousarray(full)); last_full = full
        coh = np.load(os.path.join(CACHE, f"coh_d{int(d*100):02d}.npy"))
    else:
        coh = np.load(os.path.join(CACHE, f"coh_d{int(dens_ok[-1]*100):02d}.npy"))
    enc, lat_backend = C.load_encoder(dev)
    qpool = (list(TQ) + benign_pool)[:200]
    _ = enc.encode(qpool[:8], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)  # warm-up
    tot_ms, enc_ms = [], []
    for q in qpool:
        t0 = time.perf_counter()
        qe = enc.encode([q], batch_size=1, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        t1 = time.perf_counter()
        _, ir = last_idx.search(np.ascontiguousarray(qe), C.K_FETCH); ri = [int(x) for x in ir[0] if x >= 0]
        if ri:
            sims = last_full[ri] @ qe[0]; topk = [ri[x] for x in np.argsort(-sims)[:C.K]]
            _ = [bool(coh[di] > 0.84) for di in topk]
        t2 = time.perf_counter(); enc_ms.append((t1 - t0) * 1e3); tot_ms.append((t2 - t0) * 1e3)
    latency = {"backend": lat_backend, "device_name": dev_name, "n_queries": len(qpool),
               "mean_ms": float(np.mean(tot_ms)), "p95_ms": float(np.percentile(tot_ms, 95)),
               "encode_ms_mean": float(np.mean(enc_ms)), "ref_100k_ms": "13-16 (tab:xplat)"}
    del enc
    C.log(f"PHASE 4  latency mean={latency['mean_ms']:.1f}ms p95={latency['p95_ms']:.1f}ms")

    # ---- self-verdict vs PREREG_1M ----
    gpd = {}; [gpd.__setitem__(g["density"], g["gap"]) for g in grid]
    gv = list(gpd.values()); gap_inv = (len(gv) >= 1) and (max(gv) - min(gv) <= 0.05) and all(v > 0.15 for v in gv)
    asr0 = all(g["asr_pct"] == 0.0 for g in grid)
    gmean = float(np.mean([g["docfpr_eval_doclevel_pct"] for g in grid])) if grid else None
    fpr_dev = abs(gmean - C.FPR_TARGET * 100) if gmean is not None else None
    lat_ok = latency["mean_ms"] <= 2 * 16.0   # within ~2x the 100k upper figure
    verdict = {
        "latency_flat_logarithmic": bool(lat_ok), "latency_mean_ms": latency["mean_ms"],
        "detection_gap_density_invariant": bool(gap_inv), "detection_evasion_zero": bool(asr0),
        "calibration_grand_docfpr_pct": gmean, "calibration_dev_from_target_pct": fpr_dev,
        "result": ("CONFIRMS" if (lat_ok and gap_inv and asr0 and (fpr_dev is not None and fpr_dev <= 0.2))
                   else "PRIVATE-NEGATIVE -- read straight; report with JSON"),
        "preregistered": "latency<=2x100k; gap range<=0.05 & all>0.15; 0% evasion; DocFPR near 0.69% (dev not growing)."}
    out = {"experiment": "1M corpus-scale (latency / calibration / detection)", "machine_label": "5080",
           "env": C.env_report(), "runtime_s": time.time() - t_start, "runtime_h": (time.time() - t_start) / 3600.0,
           "benchmark_estimate": bench,
           "corpus": {"N": N, "canonical_sha256": chash, "fingerprint_ok": True, "source_revisions": prov.get("source_repos"),
                      "sites": prov.get("sites"), "true_pool_after_dedup": prov.get("true_pool_after_dedup"),
                      "capped": prov.get("capped"), "domain_note": prov.get("domain_note")},
           "detector": {"type": "cluster_coh_hard_gate", "composite_used": False, "frozen_math": True,
                        "K": C.K, "K_FETCH": C.K_FETCH, "M": C.INDEX_M, "FPR_TARGET": C.FPR_TARGET,
                        "calibration": "non-oracle (1-FPR_TARGET) pctile of CAL-half clean coh; 60/40"},
           "poison": {"count_generated": len(poison_texts), "cap": POISON_MAX, "canonical_first10k_hash_ok": True,
                      "densities_run": dens_ok, "densities_skipped": dens_skip + [0.10],
                      "skip_note": f"10% needs P=100000 > poison cap {POISON_MAX}; documented, not fabricated"},
           "seeds": SEEDS, "grid": grid,
           "latency": latency,
           "scale_summary": {"grand_mean_docfpr_eval_pct": gmean, "fpr_target_pct": C.FPR_TARGET * 100,
                             "dev_from_target_pct": fpr_dev, "ref_10k_docfpr": 0.7653, "ref_100k_docfpr": 0.6741,
                             "ref_100k_dev": 0.0159},
           "verdict": verdict, "finished_unix": int(time.time())}
    json.dump(out, open(RESULT, "w", encoding="utf-8"), indent=2)
    print(json.dumps({"N": N, "result_file": "result_1M.json", "verdict": verdict["result"],
                      "latency_mean_ms": round(latency["mean_ms"], 1), "gap_per_density": {f"{k:.0%}": round(v, 4) for k, v in gpd.items()},
                      "evasion_zero": asr0, "grand_docfpr_pct": (round(gmean, 4) if gmean else None),
                      "runtime_h": round((time.time() - t_start) / 3600.0, 2)}, indent=2))

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fatal(f"UNCAUGHT: {e}", {"traceback": traceback.format_exc()[:3000]})
