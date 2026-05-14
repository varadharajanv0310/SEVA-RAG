# STEP3_FINDING_AND_DECISION.md — where the paper actually stands (read this to understand the project)

*Written 2026-05-30 after the Step‑3 in‑domain (E2) baseline. Numbers: `STEP3_GATE_RESULTS.md`. Decision is the author's; nothing past the gate has been run.*

## (a) cluster_coh confound‑kill — CONFIRMED (a clean GO on the headline)
We rebuilt the corpus so clean and poison share the **same domain** (security Stack Exchange Q&A as clean, security‑templated poison). The clean corpus is genuinely diverse (cohesion ~0.70, validated). On this in‑domain corpus, `cluster_coh` **still separates poison** at every density (gap +0.235…+0.247) with **higher SNR** than WikiText (5.8–6.0 vs 4.7). So `cluster_coh` detects **templated near‑duplication, not domain** — Limitation 2's central worry is answered for the headline signal.

## (b) The major finding: the multi‑signal *adaptive* robustness was domain‑confounded, and collapses in‑domain
The other linguistic signals tell a different story. **kw_density's SNR collapsed 38 → ~7** in‑domain (TTR/avg_sent_len also weakened), because security keywords/structure now appear in the *clean* corpus too — confirming those signals were partly detecting **domain**, not poisoning.

Consequence in the layered threat model:
- **L1 (all signals):** in‑domain ASR = **0%** — kw_density still flags the keyword‑stuffed poison, so the full detector holds.
- **L2/L3 (adaptive adversary drops kw_density [+avg_sent_len]):** in‑domain ASR = **44% / 61% / 73%** at 1/5/10% — vs ~0–17% on WikiText. Once the keyword backstop is evaded, **`cluster_coh` alone cannot hold in‑domain.**

So SEVA's "resists the adaptive adversary" result (Table V/VI L2/L3) was **substantially propped up by the domain‑confounded kw_density signal**. In a fair in‑domain test, a keyword‑dropping adversary largely defeats SEVA.

## (c) Implication: E1/E1b are now DECISIVE, not confirmatory — and the bar is higher
The plan treated E1 (white‑box embedding attack on cluster_coh) as the make‑or‑break and E1b (necessity) as the hinge. The in‑domain result **sharpens** this: if `cluster_coh` alone already fails against an adversary who merely *drops keywords* (L3, in‑domain), then an adversary who *directly diversifies poison to suppress cluster_coh* (E1) is the real threat, and `cluster_coh`'s robustness must be demonstrated **without** any linguistic backstop. E1/E1b no longer "confirm" a strong system — they **decide** whether there is a defensible robustness claim at all.

## (d) Rescoped contribution (honest framing the paper should adopt)
- **Strong, defensible:** an LLM‑free, density‑agnostic, **doc‑level** poisoning signal (`cluster_coh`) that is **domain‑independent** (validated in‑domain) and detects templated/near‑duplicate corpus poisoning that **per‑query density‑estimating defenses (RAGDefender) structurally miss** (E4‑HH, pending) — at the L1 operating point.
- **Weakened / must narrow:** the **multi‑signal adaptive‑robustness** claim. The L2/L3 "evasion‑resistant" numbers do not survive a fair in‑domain test; they were carried by kw_density. The paper must **narrow the adaptive‑adversary claim** and report the in‑domain L2/L3 numbers honestly.
- Net: contribution shifts from "robust multi‑signal detector that resists adaptive evasion" toward "**a robust, domain‑independent geometric core signal + an honest characterization of when the composite's adaptive robustness holds**." This is smaller but defensible; trying to keep the original adaptive claim is not.

## (e) AUTHOR DECISION PENDING
**Recommended: GO to E1/E1b with the rescoped framing above.** E1/E1b will determine whether the geometric core (`cluster_coh`) survives a *direct* white‑box suppression attack with no linguistic backstop — which is now the paper's central question. Options:
1. **GO (recommended):** proceed to E1/E1b under the rescoped framing; first (optional) run seeds 7,123 in‑domain to confirm the L2/L3 finding's seed‑stability for the tables.
2. **Reframe‑first:** rewrite the contribution/limitations around the rescoped story before any more experiments.
3. **Hold/escalate:** if the adaptive‑robustness claim was load‑bearing for the target venue, reconsider venue/scope (this is the IJIS/JISA‑realistic vs TDSC/TIFS‑reach decision in EXPERIMENT_PLAN.md §6).

Do **not** start E1/E1b until the author chooses. Acceptance‑probability impact: the cluster_coh confound‑kill is a plus; the L2/L3 collapse is a minus that the rescoped framing + a favorable E1b can offset — see EXPERIMENT_PLAN.md §5/§6.
