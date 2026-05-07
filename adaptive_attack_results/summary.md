# SEVA v6.2.1 — Adaptive Attack Experiment: adaptive_diverse

**Attack**: `adaptive_diverse`  |  **Corpus**: 100k  |  **Density**: 5%  |  **Seeds**: [7, 42, 123]

## Aggregate Results

```
===========================================================================
  ADAPTIVE CLUSTER_COH ATTACK — AGGREGATE RESULTS (5% density, 100k corpus)
===========================================================================

                          L1 ASR    L2 ASR    L3 ASR    L1 FPR    L2 FPR    L3 FPR
---------------------------------------------------------------------------
Baseline (template)        0.00%     0.53%    16.27%     0.63%     0.62%     0.66%
Adaptive (diverse)         0.00%     0.00%     0.00%     0.57%     0.56%     0.57%
Delta                     +0.00%    -0.53%   -16.27%    -0.06%    -0.06%    -0.10%

cluster_coh distribution — ADAPTIVE poison docs:
  Mean: 0.9488  Std: 0.0196  Per-seed means: ['0.9488', '0.9488', '0.9488']
  Fraction below 0.85 threshold: 0.0% (mean across seeds)

cluster_coh distribution — BASELINE poison docs:
  Mean: 0.9909  Per-seed means: ['0.9909', '0.9909', '0.9909']

VERDICT: cluster_coh signal [HOLDS]
  Adaptive L1 ASR (0.00%) < Baseline + 5pp
  HOLDS            if adaptive L1 ASR < baseline L1 ASR + 5pp
  PARTIALLY DEFEATED if adaptive L1 ASR ∈ [5%, 25%)
  DEFEATED           if adaptive L1 ASR >= 25%

Per-seed breakdown (L1 ASR / FPR | L2 ASR / FPR | L3 ASR / FPR):
  S7: L1 0.0%/0.56% | L2 0.0%/0.56% | L3 0.0%/0.51%
  S42: L1 0.0%/0.70% | L2 0.0%/0.73% | L3 0.0%/0.59%
  S123: L1 0.0%/0.43% | L2 0.0%/0.38% | L3 0.0%/0.60%

===========================================================================
```

## Verdict

**`cluster_coh` signal: [HOLDS]**

Adaptive L1 ASR (0.00%) < Baseline + 5pp

## Per-Seed Details

### Seed 7

- cluster_coh (adaptive): mean=0.9488  std=0.0196  below_threshold=0.0%
- cluster_coh (baseline): poison_mean=0.9909476637840271
- Perturbation iterations: 0
- Gen latency: 0.000 ms/doc

| Layer | ASR (adaptive) | FPR (adaptive) | TP | FN | FP | TN |
|-------|---------------|----------------|-----|-----|-----|-----|
| L1    | 0.00%          | 0.565%          | 250  | 0  | 21  | 3696  |
| L2    | 0.00%          | 0.565%          | 250  | 0  | 21  | 3696  |
| L3    | 0.00%          | 0.511%          | 250  | 0  | 19  | 3698  |

### Seed 42

- cluster_coh (adaptive): mean=0.9488  std=0.0196  below_threshold=0.0%
- cluster_coh (baseline): poison_mean=0.9909476637840271
- Perturbation iterations: 0
- Gen latency: 0.000 ms/doc

| Layer | ASR (adaptive) | FPR (adaptive) | TP | FN | FP | TN |
|-------|---------------|----------------|-----|-----|-----|-----|
| L1    | 0.00%          | 0.699%          | 250  | 0  | 26  | 3693  |
| L2    | 0.00%          | 0.726%          | 250  | 0  | 27  | 3692  |
| L3    | 0.00%          | 0.592%          | 250  | 0  | 22  | 3697  |

### Seed 123

- cluster_coh (adaptive): mean=0.9488  std=0.0196  below_threshold=0.0%
- cluster_coh (baseline): poison_mean=0.9909476637840271
- Perturbation iterations: 0
- Gen latency: 0.000 ms/doc

| Layer | ASR (adaptive) | FPR (adaptive) | TP | FN | FP | TN |
|-------|---------------|----------------|-----|-----|-----|-----|
| L1    | 0.00%          | 0.434%          | 250  | 0  | 16  | 3669  |
| L2    | 0.00%          | 0.380%          | 250  | 0  | 14  | 3671  |
| L3    | 0.00%          | 0.597%          | 250  | 0  | 22  | 3663  |

## Methodology

- **Generation**: 10 topic frames (OAuth2/LDAP/PKI/ZeroTrust/SAML/Vault/MFA/RBAC/SCIM/Kerberos)
- **Perturbation**: Token dropout (15-25%) for docs with K=5 nearest-poison cluster_coh > 0.85, up to 3 iterations
- **Checkpoint dir**: `seva_checkpoints_4060_100k_p050_adaptive/`
- **Source tree**: unmodified — no existing files changed