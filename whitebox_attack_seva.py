#!/usr/bin/env python3
"""
whitebox_attack_seva.py — E1/E1b white-box attack on SEVA's cluster_coh.

PRE-REGISTRATION: E1_DESIGN.md (committed 66ed200, BEFORE this file).
Detector is BYTE-FROZEN: this module never imports/edits text_features, _score,
_snr_weights, _compute_doc_coh or the constants — it only (a) synthesizes/optimizes
poison and (b) re-implements the *exact* frozen cluster_coh K-NN math and the exact
50-iter calibrate() so the probe scores against the detector's own threshold.

=== PROBE MODE (this run) =====================================================
E1b *best-case geometric feasibility*, seed 42, reusing the cached gate embeddings
(NO re-embed, NO faiss rebuild). The attacker is granted its UPPER BOUND: poison
vectors placed directly on the unit sphere, maximally spread within the
retrievability cap. We instrument (cohesion, retrievability, ASR) JOINTLY at every
attack point and report a directional NECESSARY / CORRELATED verdict vs tau_coh(42).
If even this best case cannot get an *attack-effective* poison set below tau_coh,
that is strong necessity (real text attacks, E1, are weaker). STOPS at the gate.

The realizable-text attack subclass (full E1/E1b, post-gate) is scaffolded at the
bottom but NOT run in probe mode.
"""

import os, sys, json, time, math
import numpy as np

# ---------------------------------------------------------------------------
# Frozen constants (mirror seva_benchmark_4060.py — DO NOT CHANGE)
# ---------------------------------------------------------------------------
K            = 5          # K nearest neighbors for cluster_coh
K_FETCH      = 20         # over-fetch then exact rerank (retrieval)
EMB_DIM      = 1024
FPR_TARGET   = 0.0069     # universal FPR calibration target
N            = 100_000
PTAG         = "p050"     # 5% density (the E1 operating density)
CORPUS_TAG   = "secqa"
SEED         = 42

CWD          = os.path.dirname(os.path.abspath(__file__))
CKDIR        = os.path.join(CWD, f"seva_checkpoints_4060_{N//1000}k_{CORPUS_TAG}_{PTAG}")
RESULTS_DIR  = os.path.join(CWD, "whitebox_attack_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pre-registered E1b knobs
R_STAR_FLOOR     = 0.60       # floor-of-the-floor (baseline poison achieves >=0.60)
R_STAR_EPS       = 0.005      # margin so poison strictly clears the displacement threshold
TARGET_ASR_FRAC  = 0.80       # "reaches target ASR" = >= 0.8 * R_base of targeted queries
B_SWEEP          = [50, 150, 250, 500, 1000, 2500, 5000]   # total poison budget (<=5% of 100k)
ALPHA_POLICIES   = ["floor", "hi075"]  # cos-to-query: at the floor (best spread) vs 0.75 (tighter cap)
SYNTH_SEED       = 1337


def log(m): print(m, flush=True)


def calibrate(scores_clean, fpr_target):
    """IDENTICAL to seva_benchmark_4060.py phase3 calibrate() (50-iter binary search)."""
    lo, hi = float(scores_clean.min()), float(scores_clean.max())
    for _ in range(50):
        mid = (lo + hi) / 2
        if float(np.mean(scores_clean > mid)) > fpr_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    import torch
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available."); sys.exit(1)
    DEVICE = torch.device("cuda:0")
    torch.manual_seed(SYNTH_SEED); np.random.seed(SYNTH_SEED)
    t0 = time.perf_counter()

    log("=" * 72)
    log("  WHITEBOX E1b PROBE — best-case geometric feasibility (seed 42, 5%)".center(72))
    log("=" * 72)

    # ---- Load frozen cached artifacts (NO re-embed) ----
    pe_path   = os.path.join(CKDIR, "p2_pe.npy")
    coh_path  = os.path.join(CKDIR, "p2_doc_coh.npy")
    corp_path = os.path.join(CKDIR, "p1_corpus.json")
    q_path    = os.path.join(CKDIR, "p1_query.json")
    p3_path   = os.path.join(CKDIR, "p3_v6.2_s042.json")
    for p in (pe_path, coh_path, corp_path, q_path, p3_path):
        if not os.path.exists(p):
            log(f"ERROR: missing cache {p}"); sys.exit(1)

    log("  Loading cached embeddings / doc_coh / corpus / queries ...")
    pe       = np.load(pe_path)                       # (100000, 1024) float32, [0:P]=poison
    doc_coh  = np.load(coh_path)                       # (100000,) frozen cluster_coh
    corpus   = json.load(open(corp_path, encoding="utf-8"))
    queries  = json.load(open(q_path, encoding="utf-8"))
    p3       = json.load(open(p3_path, encoding="utf-8"))
    is_poison = np.array([bool(d["is_poisoned"]) for d in corpus], dtype=bool)
    P = int(is_poison.sum())
    log(f"  corpus={len(corpus):,}  poison={P:,}  pe={pe.shape}  doc_coh={doc_coh.shape}")
    assert pe.shape == (N, EMB_DIM) and doc_coh.shape == (N,)

    clean_idx  = np.where(~is_poison)[0]
    poison_idx = np.where(is_poison)[0]
    baseline_poison_coh = float(doc_coh[poison_idx].mean())
    clean_coh_mean = float(doc_coh[clean_idx].mean()); clean_coh_std = float(doc_coh[clean_idx].std())
    log(f"  [provenance] baseline poison cluster_coh mean={baseline_poison_coh:.4f} "
        f"(gate=0.9909)  clean mean={clean_coh_mean:.4f}+-{clean_coh_std:.4f} (gate 0.7526+-0.0398)")

    # ---- tau_coh(42): cluster_coh-only threshold at FPR 0.69% on cal-CLEAN docs ----
    cal_ids = p3.get("cal_doc_ids", [])
    cal_idx = np.array([int(s.split("_")[1]) for s in cal_ids], dtype=np.int64)
    cal_clean_idx = cal_idx[~is_poison[cal_idx]]
    cal_clean_coh = doc_coh[cal_clean_idx]
    tau_coh = calibrate(cal_clean_coh, FPR_TARGET)
    clean_p95 = float(np.percentile(doc_coh[clean_idx], 95.0))
    log(f"  tau_coh(42) [cluster_coh-only, FPR 0.69%, {len(cal_clean_coh)} cal-clean docs] = {tau_coh:.4f}")
    log(f"  clean P95(cluster_coh) = {clean_p95:.4f}  (context band)")

    # ---- GPU tensors ----
    clean_t = torch.from_numpy(np.ascontiguousarray(pe[clean_idx])).to(DEVICE)     # (Nc,1024)
    base_poison_t = torch.from_numpy(np.ascontiguousarray(pe[poison_idx])).to(DEVICE)  # (P,1024)
    Nc = clean_t.shape[0]

    # ---- Encode the 50 targeted queries with the FROZEN encoder ----
    adv_q = []
    seen = set()
    for q in queries:
        if q.get("adv") and q["q"] not in seen:
            adv_q.append(q["q"]); seen.add(q["q"])
    log(f"  Targeted queries: {len(adv_q)} unique adversarial")
    from sentence_transformers import SentenceTransformer
    log("  Loading BGE-large (frozen) to embed queries ...")
    enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
    Qnp = enc.encode(adv_q, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    del enc; torch.cuda.empty_cache()
    Q = torch.from_numpy(Qnp).to(DEVICE)          # (50,1024) unit
    nQ = Q.shape[0]

    # ---- d5(q), r*(q), r*_op ----
    simQC = Q @ clean_t.T                          # (50, Nc)
    d5 = torch.topk(simQC, K, dim=1).values[:, -1] # 5th-largest clean cos per query
    d5np = d5.detach().cpu().numpy()
    r_star = np.maximum(d5np + R_STAR_EPS, R_STAR_FLOOR)   # per-query floor
    r_star_op = float(np.median(r_star))
    log(f"  d5(q): min={d5np.min():.3f} median={np.median(d5np):.3f} max={d5np.max():.3f}")
    log(f"  r*(q)=max(d5+{R_STAR_EPS},{R_STAR_FLOOR})  ->  r*_op(median)={r_star_op:.4f}")

    # ---- helper: exact frozen cluster_coh for a poison bank against [clean; poison] ----
    def cluster_coh_of(poison_t):
        B = poison_t.shape[0]
        bank = torch.cat([clean_t, poison_t], 0)   # (Nc+B,1024)
        cohs = torch.empty(B, device=DEVICE)
        iu = torch.triu_indices(K, K, offset=1, device=DEVICE)
        CH = 256
        for s in range(0, B, CH):
            e = min(s + CH, B)
            sims = poison_t[s:e] @ bank.T          # (b, Nc+B)
            for loc in range(e - s):
                sims[loc, Nc + s + loc] = -2.0     # exclude self
            top = torch.topk(sims, K, dim=1).indices   # (b,K)
            nbr = bank[top]                            # (b,K,1024)
            pair = torch.bmm(nbr, nbr.transpose(1, 2)) # (b,K,K)
            cohs[s:e] = pair[:, iu[0], iu[1]].mean(dim=1)
        return cohs                                    # (B,)

    # ---- R_base: baseline poison retrieval reach (detection OFF) ----
    def reach_and_asr(poison_t, poison_coh):
        """Return per-query: reach (>=1 poison in top-K), asr_loose (>=1 UNFLAGGED poison),
        asr_majority (>=ceil(K/2) unflagged poison), asr_rank1 (top-1 is unflagged poison)."""
        B = poison_t.shape[0]
        bank = torch.cat([clean_t, poison_t], 0)
        is_poi = torch.cat([torch.zeros(Nc, dtype=torch.bool, device=DEVICE),
                            torch.ones(B, dtype=torch.bool, device=DEVICE)])
        coh_b  = torch.cat([torch.full((Nc,), 9.9, device=DEVICE), poison_coh])  # clean coh irrelevant
        sims = Q @ bank.T                              # (50, Nc+B)
        top = torch.topk(sims, K, dim=1).indices       # (50,K) -- ranked desc
        reach = asr_loose = asr_maj = asr_r1 = 0
        maj_need = (K // 2) + 1
        for qi in range(nQ):
            idx = top[qi]
            pis = is_poi[idx]
            if bool(pis.any()): reach += 1
            unflag = pis & (coh_b[idx] <= tau_coh)
            nuf = int(unflag.sum())
            if nuf >= 1: asr_loose += 1
            if nuf >= maj_need: asr_maj += 1
            if bool(is_poi[idx[0]]) and float(coh_b[idx[0]]) <= tau_coh: asr_r1 += 1
        f = float(nQ)
        return reach / f, asr_loose / f, asr_maj / f, asr_r1 / f

    base_coh = base_poison_t.new_tensor(doc_coh[poison_idx])  # frozen coh of baseline poison
    R_base, base_asr_loose, _, _ = reach_and_asr(base_poison_t, base_coh)
    log(f"  R_base (baseline poison reach, detection OFF) = {R_base*100:.1f}%   "
        f"target-ASR gate = {TARGET_ASR_FRAC*R_base*100:.1f}%")

    # ---- synthesize best-case spread poison in caps ----
    def synth_poison(n_per_q, policy):
        embs = []
        for qi in range(nQ):
            qe = Q[qi]                                   # (1024,) unit
            if policy == "floor":
                alpha = float(r_star[qi])
            else:  # hi075
                alpha = max(0.75, float(d5np[qi]) + R_STAR_EPS)
            alpha = min(alpha, 0.999)
            G = torch.randn(n_per_q, EMB_DIM, device=DEVICE)
            G = G - (G @ qe)[:, None] * qe[None, :]      # project out qe
            # orthonormal directions ⊥ qe via QR
            M = torch.cat([qe[:, None], G.T], dim=1)     # (1024, n+1)
            Qo, _ = torch.linalg.qr(M)
            U = Qo[:, 1:1 + n_per_q].T                   # (n,1024) orthonormal ⊥ qe
            p = alpha * qe[None, :] + math.sqrt(max(1 - alpha * alpha, 0.0)) * U
            p = p / p.norm(dim=1, keepdim=True)
            embs.append(p)
        return torch.cat(embs, 0)                        # (nQ*n_per_q, 1024)

    # ---- sweep ----
    log("\n  --- JOINT FRONTIER (best-case geometry) ---")
    hdr = f"  {'policy':6} {'B':>5} {'n/q':>4} | {'meanCoh':>8} {'%<=tau':>7} | {'reach':>6} {'ASRloose':>8} {'ASRmaj':>7} {'ASRr1':>6}"
    log(hdr); log("  " + "-" * (len(hdr) - 2))
    frontier = []
    for policy in ALPHA_POLICIES:
        for B in B_SWEEP:
            n_per_q = max(1, B // nQ)
            poison_t = synth_poison(n_per_q, policy)
            coh = cluster_coh_of(poison_t)
            mean_coh = float(coh.mean()); frac_below = float((coh <= tau_coh).float().mean())
            reach, asr_loose, asr_maj, asr_r1 = reach_and_asr(poison_t, coh)
            rec = {"policy": policy, "B": int(poison_t.shape[0]), "n_per_q": n_per_q,
                   "mean_coh": mean_coh, "frac_below_tau": frac_below,
                   "reach": reach, "asr_loose": asr_loose,
                   "asr_majority": asr_maj, "asr_rank1": asr_r1}
            frontier.append(rec)
            log(f"  {policy:6} {poison_t.shape[0]:>5} {n_per_q:>4} | {mean_coh:>8.4f} {frac_below*100:>6.1f}% | "
                f"{reach*100:>5.1f}% {asr_loose*100:>7.1f}% {asr_maj*100:>6.1f}% {asr_r1*100:>5.1f}%")
            del poison_t, coh; torch.cuda.empty_cache()

    # ---- C_min under loose & majority target-ASR gates ----
    gate = TARGET_ASR_FRAC * R_base
    def cmin(metric):
        passing = [r for r in frontier if r[metric] >= gate]
        if not passing:
            return None, None
        best = min(passing, key=lambda r: r["mean_coh"])
        return best["mean_coh"], best
    cmin_loose, best_loose = cmin("asr_loose")
    cmin_maj,   best_maj   = cmin("asr_majority")

    def verdict(cmin_val):
        # CORRECTED FRAMING (author 2026-05-31): free-on-sphere placement tests only the r*^2 cone
        # floor (r*^2=0.52 << tau_coh=0.84 -> low cohesion is FOREGONE), NOT manifold realizability.
        # This is therefore NOT a NECESSARY/CORRELATED verdict on the real claim.
        if cmin_val is None:
            return "no attack-effective config at the floor even free-on-sphere"
        return ("pure-geometric necessity UNCLAIMABLE (r*^2 << tau_coh, foregone); manifold "
                "realizability is the open decisive question (E1) -- NOT evidence SEVA is weak")

    log("\n" + "=" * 72)
    log(f"  tau_coh(42) = {tau_coh:.4f}   (clean P95 = {clean_p95:.4f})   "
        f"target-ASR gate = {gate*100:.1f}% of queries")
    log(f"  C_min (LOOSE: >=1 unflagged poison in top-K)    = "
        f"{('%.4f' % cmin_loose) if cmin_loose is not None else 'n/a'}  -> {verdict(cmin_loose)}")
    if best_loose:
        log(f"      achieved by: {best_loose['policy']} B={best_loose['B']} n/q={best_loose['n_per_q']} "
            f"(reach {best_loose['reach']*100:.0f}%, ASRloose {best_loose['asr_loose']*100:.0f}%)")
    log(f"  C_min (MAJORITY: >=3/5 unflagged poison in top-K)= "
        f"{('%.4f' % cmin_maj) if cmin_maj is not None else 'n/a'}  -> {verdict(cmin_maj)}")
    if best_maj:
        log(f"      achieved by: {best_maj['policy']} B={best_maj['B']} n/q={best_maj['n_per_q']} "
            f"(reach {best_maj['reach']*100:.0f}%, ASRmaj {best_maj['asr_majority']*100:.0f}%)")
    log("=" * 72)

    out = {
        "mode": "probe", "seed": SEED, "density": PTAG, "corpus_tag": CORPUS_TAG,
        "frozen_anchors": {"baseline_poison_coh": baseline_poison_coh,
                            "clean_coh_mean": clean_coh_mean, "clean_coh_std": clean_coh_std},
        "tau_coh_42": tau_coh, "tau_coh_method": "frozen 50-iter calibrate on cal-clean cluster_coh @FPR0.69%",
        "n_cal_clean": int(len(cal_clean_coh)), "clean_p95": clean_p95,
        "r_star_op": r_star_op, "r_star_floor": R_STAR_FLOOR,
        "d5_median": float(np.median(d5np)), "d5_min": float(d5np.min()), "d5_max": float(d5np.max()),
        "R_base": R_base, "target_asr_gate": gate, "target_asr_frac": TARGET_ASR_FRAC,
        "frontier": frontier,
        "C_min_loose": cmin_loose, "C_min_loose_config": best_loose,
        "C_min_majority": cmin_maj, "C_min_majority_config": best_maj,
        "verdict_loose": verdict(cmin_loose), "verdict_majority": verdict(cmin_maj),
        "runtime_s": time.perf_counter() - t0,
        "note": ("Probe uses tau_coh(42) only (directional). Binding E1b verdict uses "
                 "max_s tau_coh(s) over corrected 3-seed calibration. Best-case GEOMETRIC "
                 "upper bound: synthetic spread poison; realizable-text cost is measured in E1."),
    }
    out_path = os.path.join(RESULTS_DIR, "probe_s042.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    log(f"\n  Saved -> {out_path}   ({out['runtime_s']:.1f}s)")
    log("  PROBE COMPLETE — STOP at gate; report (cohesion, retrievability, ASR, verdict).")


# ============================================================================
# Realizable-text attack subclass (FULL E1/E1b, post-gate) — scaffold, NOT run.
# ============================================================================
def _build_whitebox_bench_class(seva_mod):
    """Subclass SEVABench; override ONLY phase1 (inject attack text) + phase2
    (write optimized poison embeddings into self.pe[:P], rebuild index) + _ck/rf.
    Detector (_compute_doc_coh/_score/phase3/phase4) is reused verbatim. Filled in
    after the probe gate per E1_DESIGN.md §5/§10."""
    raise NotImplementedError("Full realizable-text E1/E1b runs come after the probe gate.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode != "probe":
        print("Only probe mode is enabled pre-gate (see E1_DESIGN.md)."); sys.exit(2)
    main()
