# PREREGISTRATION.md — registered expectations for the cross-platform runs

Registered **before** the 4060/M4 results are read, so they are read honestly. Each
`hardgate_xrun.py` result also embeds this check in its own `verdict` block, computed mechanically.

## What we are testing
The **same** in-domain Security-SE corpus (hash-verified identical on 5080 / 4060 / M4), the **same**
frozen cluster_coh hard gate under non-oracle calibration, re-embedded on each platform. Question:
does the geometry replicate across hardware?

## Registered predictions (per machine)
For each of the 3 densities (1/5/10%) × 3 seeds (42/7/123):
1. **Density-invariant cluster_coh gap.** Gap = poison_coh_mean − clean_coh_mean is **positive and
   stable across density**: range across the three densities ≤ **0.05**, and every gap > **0.15**.
   Reference (5080): gap ≈ **+0.235 … +0.247** (range ≈ 0.012), SNR ≈ **5.8–6.0**. The gap is
   seed-invariant (≈ identical across the 3 seeds — it is a corpus property, not a calibration one).
2. **0% templated ASR.** Retrieval-based attack-success on the 50 targeted queries is **0%** on all
   nine conditions (templated poison sits at coh ≈ 0.99 ≫ τ ≈ 0.84, so every retrieved poison is
   flagged). DocFPR stays near target, **~0.4–0.9%**, with mild seed-dependence.

## PASS / verdict
- **CONFIRMS** = corpus hash matches the 5080 **and** prediction 1 holds **and** prediction 2 holds.
  → fold the run in as the platform's confirming replication.
- **DIVERGES** = any of the above fails.

## The private-negative rule
A **DIVERGES on a hash-MATCHED corpus** (same corpus, same detector, different platform → different
geometry: gap collapses, or ASR > 0, or DocFPR wild) is a **genuine finding**. It is **reportable to
the operator, NOT auto-published**: do not quietly drop it and do not put it in the paper unexamined.
First rule out an encoder/version mismatch (compare the result's `env` and `encoder.revision` to
`MANIFEST.package_versions_5080` / `pinned_encoder_revision`); if the versions match and the corpus
hash matches, it is a real platform-dependence worth understanding before any cross-platform claim
relies on it.

A **hash MISMATCH** is not a negative result at all — it means the corpus wasn't identical (source-
data or Python drift), so the run is simply invalid for comparison; rebuild with the pinned manifest
and Python 3.11 and re-run. The hash gate exists precisely so a divergent corpus is caught **before**
any number is trusted.
