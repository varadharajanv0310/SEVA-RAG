#!/usr/bin/env python3
"""Regenerate the figures in docs/figures/ from the committed result JSONs.

Every value plotted is read from a file in this directory. Nothing is hardcoded:
if a number is not in a result JSON it does not appear in a figure. Re-running
this script after a re-run of the experiments reproduces the figures exactly.

    python reproduction/make_figures.py

Outputs (written to docs/figures/):
    scaling.png             <- result_scale10k.json, result_scale100k.json, result_1M.json
    cross_platform.png      <- result_4060.json, result_M4.json
    encoder_invariance.png  <- result_encoder_{bge,e5,gte}.json
    confidence_interval.png <- result_hienc_ci.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

INK = "#1f2933"
ACCENT = "#2f6f9f"
WARN = "#c1553b"
OK = "#3f7d58"


def load(name: str) -> dict:
    with (HERE / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def gaps_by_density(doc: dict) -> dict[float, list[float]]:
    """Collect per-density cluster_coh gaps from a result's grid."""
    out: dict[float, list[float]] = {}
    for row in doc.get("grid", []):
        d = row.get("density")
        g = row.get("gap")
        if d is None or g is None:
            continue
        out.setdefault(float(d), []).append(float(g))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def density_key(k: str) -> float:
    """Density keys appear as fractions ('0.01') or percents ('1%') across files."""
    k = str(k).strip()
    if k.endswith("%"):
        return float(k[:-1]) / 100.0
    return float(k)


def sorted_density_items(gp: dict) -> list[tuple[float, float]]:
    return sorted(((density_key(k), float(v)) for k, v in gp.items()), key=lambda t: t[0])


# --------------------------------------------------------------------------
# 1. Scaling: calibration and detection separation across corpus size
# --------------------------------------------------------------------------
def fig_scaling() -> None:
    files = [
        ("result_scale10k.json", "10k"),
        ("result_scale100k.json", "100k"),
        ("result_1M.json", "1M"),
    ]
    docs = [(load(f), lbl) for f, lbl in files]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # -- left: realised DocFPR vs the pre-registered target
    ns, fprs, target = [], [], None
    for doc, lbl in docs:
        ss = doc["scale_summary"]
        fprs.append(ss["grand_mean_docfpr_eval_pct"])
        ns.append(lbl)
        target = ss.get("fpr_target_pct", target)

    bars = ax1.bar(ns, fprs, color=ACCENT, width=0.55)
    if target is not None:
        ax1.axhline(target, ls="--", lw=1.2, color=WARN)
        ax1.text(
            len(ns) - 0.45, target, f"  target {target}%", color=WARN,
            va="bottom", ha="left", fontsize=8,
        )
    for b, v in zip(bars, fprs):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}%", ha="center", va="bottom", fontsize=8)
    ax1.set_title("Calibration holds as the corpus grows", fontsize=10, color=INK)
    ax1.set_xlabel("corpus size (documents)")
    ax1.set_ylabel("grand mean DocFPR, eval half (%)")
    ax1.set_ylim(0, max(fprs) * 1.35)

    # -- right: cluster_coh gap per poison density
    for (doc, lbl), colour in zip(docs, [WARN, ACCENT, OK]):
        gb = gaps_by_density(doc)
        if not gb:
            continue
        ds = sorted(gb)
        ax2.plot(
            ds, [mean(gb[d]) for d in ds], marker="o", lw=1.6, color=colour,
            label=f"N={lbl}",
        )
    ax2.set_xscale("log")
    ax2.set_title("Detection separation vs poison density", fontsize=10, color=INK)
    ax2.set_xlabel("poison density (fraction of corpus)")
    ax2.set_ylabel("cluster_coh gap (poison − clean)")
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Scaling — source: result_scale10k.json, result_scale100k.json, result_1M.json",
        fontsize=8, color="#6b7280", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "scaling.png")


# --------------------------------------------------------------------------
# 2. Cross-platform agreement: RTX 4060 vs Apple M4
# --------------------------------------------------------------------------
def fig_cross_platform() -> None:
    a, b = load("result_4060.json"), load("result_M4.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # -- left: gap per density, both machines
    for doc, colour, marker, ms in ((a, ACCENT, "o", 9), (b, WARN, "s", 5)):
        items = sorted_density_items(doc["verdict"]["gap_per_density"])
        ax1.plot(
            [d for d, _ in items], [g for _, g in items],
            marker=marker, lw=1.6, color=colour, ms=ms, alpha=0.85,
            label=f"{doc['machine_label']} ({doc['env']['backend']})",
        )
    ax1.set_xscale("log")
    ax1.set_title("Identical detection gap on both platforms", fontsize=10, color=INK)
    ax1.set_xlabel("poison density")
    ax1.set_ylabel("cluster_coh gap")
    ax1.legend(frameon=False, fontsize=8)

    # -- right: per-query latency, which is where the platforms actually differ
    labels, means, p95s = [], [], []
    for doc in (a, b):
        lat = doc["latency"]
        labels.append(f"{doc['machine_label']}\n{lat['device_name']}")
        means.append(lat["mean_ms"])
        p95s.append(lat["p95_ms"])
    x = range(len(labels))
    ax2.bar([i - 0.18 for i in x], means, width=0.36, color=ACCENT, label="mean ms")
    ax2.bar([i + 0.18 for i in x], p95s, width=0.36, color="#9db8cc", label="p95 ms")
    for i, (m, p) in enumerate(zip(means, p95s)):
        ax2.text(i - 0.18, m, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
        ax2.text(i + 0.18, p, f"{p:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_title("Per-query latency differs; the decision does not", fontsize=10, color=INK)
    ax2.set_ylabel("ms per query")
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Cross-platform — source: result_4060.json, result_M4.json",
        fontsize=8, color="#6b7280", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "cross_platform.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "cross_platform.png")


# --------------------------------------------------------------------------
# 3. Encoder invariance: bge / e5 / gte on shared axes
# --------------------------------------------------------------------------
def fig_encoders() -> None:
    keys = ["bge", "e5", "gte"]
    docs = [load(f"result_encoder_{k}.json") for k in keys]
    names = [d["encoder"]["model"] for d in docs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # -- left: gap per density on shared axes.
    # The absolute separation is encoder-dependent (bge sits roughly twice as high
    # as e5/gte); what generalises is that each encoder is flat in density and
    # clears the pre-registered bar. Label the per-encoder spread so the reader
    # sees both facts rather than only the flatness.
    for doc, name, colour in zip(docs, names, [ACCENT, WARN, OK]):
        items = sorted_density_items(doc["verdict"]["measured"]["gap_per_density"])
        rel = doc["verdict"]["measured"].get("gap_range_rel")
        ax1.plot(
            [d for d, _ in items], [g for _, g in items],
            marker="o", lw=1.6, color=colour, ms=5,
            label=f"{name.split('/')[-1]} (range {rel:.4f})" if rel is not None else name.split("/")[-1],
        )
    ax1.set_xscale("log")
    ax1.set_title("Gap is flat in density; its level is encoder-dependent", fontsize=10, color=INK)
    ax1.set_xlabel("poison density")
    ax1.set_ylabel("cluster_coh gap")
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False, fontsize=7.5)

    # -- right: the pre-registered pass criteria, measured
    snr = [d["verdict"]["measured"]["snr_min"] for d in docs]
    fpr = [d["verdict"]["measured"]["grand_mean_docfpr_eval_pct"] for d in docs]
    short = [n.split("/")[-1] for n in names]
    x = range(len(short))
    ax2.bar([i - 0.18 for i in x], snr, width=0.36, color=ACCENT, label="min SNR")
    ax2b = ax2.twinx()
    ax2b.bar([i + 0.18 for i in x], fpr, width=0.36, color="#c9a227", label="DocFPR %")
    ax2b.grid(False)
    for i, v in enumerate(snr):
        ax2.text(i - 0.18, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(fpr):
        ax2b.text(i + 0.18, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(short, fontsize=7)
    ax2.set_ylabel("min SNR (left)")
    ax2b.set_ylabel("grand mean DocFPR % (right)")
    ax2.set_title("Pre-registered criteria, measured", fontsize=10, color=INK)

    asr = {d["encoder"]["key"]: d["verdict"]["measured"]["asr_max_pct"] for d in docs}
    if set(asr.values()) == {0.0}:
        ax2.text(
            0.5, -0.34, "max templated ASR = 0.0% on all three encoders",
            transform=ax2.transAxes, ha="center", fontsize=8, color=OK,
        )

    fig.suptitle(
        "Encoder invariance — source: result_encoder_bge.json, result_encoder_e5.json, result_encoder_gte.json",
        fontsize=8, color="#6b7280", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "encoder_invariance.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "encoder_invariance.png")


# --------------------------------------------------------------------------
# 4. Wilson interval on the zero-evasion high-encounter run
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fig_ci() -> None:
    d = load("result_hienc_ci.json")
    n = int(d["n_encounters"])
    k = int(d["evasions"])
    z = float(d["wilson_z"])
    lo, hi = wilson(k, n, z)
    hi_pct = hi * 100.0
    reported = float(d["wilson_upper_pct"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})

    # -- left: the interval itself
    ax1.errorbar(
        [0], [k / n * 100], yerr=[[0], [hi_pct]], fmt="o", color=INK,
        ecolor=ACCENT, elinewidth=2.5, capsize=8, ms=7,
    )
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_xticks([])
    ax1.set_ylim(-hi_pct * 0.25, hi_pct * 1.7)
    ax1.set_ylabel("attack success rate (%)")
    ax1.set_title(
        f"{k} evasions in {n:,} encounters", fontsize=10, color=INK,
    )
    ax1.annotate(
        f"Wilson upper bound\n{hi_pct:.6f}%  (z={z})",
        xy=(0, hi_pct), xytext=(0.12, hi_pct * 1.15), fontsize=8, color=ACCENT,
    )

    # -- right: the frozen gate vs the coherence distributions it separates
    rows = [
        ("clean (pure corpus)", d["clean_coh_pure_mean"], "#9db8cc"),
        ("clean (poisoned corpus)", d["clean_coh_poisoned_mean"], "#7fa3bd"),
        ("poison (min)", d["poison_coh_min"], "#e0a08c"),
        ("poison (mean)", d["poison_coh_mean"], WARN),
    ]
    ax2.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for i, r in enumerate(rows):
        ax2.text(r[1], i, f" {r[1]:.4f}", va="center", fontsize=8)
    tau = float(d["tau_coh"])
    ax2.axvline(tau, ls="--", lw=1.4, color=INK)
    ax2.text(tau, len(rows) - 0.35, f" frozen τ = {tau:.4f}", fontsize=8, color=INK)
    ax2.set_xlim(0.6, 1.06)
    ax2.set_xlabel("cluster coherence")
    ax2.set_title("Every poison document sits above the frozen gate", fontsize=10, color=INK)

    fig.suptitle(
        "Zero-evasion confidence interval — source: result_hienc_ci.json",
        fontsize=8, color="#6b7280", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT / "confidence_interval.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "confidence_interval.png")
    # recomputed vs stored, as a self-check on the committed artifact
    print(f"  recomputed Wilson upper {hi_pct:.6f}% vs stored {reported:.6f}%")


if __name__ == "__main__":
    fig_scaling()
    fig_cross_platform()
    fig_encoders()
    fig_ci()
    print("\nall figures written to", OUT)
