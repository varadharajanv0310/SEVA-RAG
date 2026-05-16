# KICKOFF_E1_DECISION.md — paste-ready prompt to start the next session

*Created 2026-05-30. Purpose: a self-contained kickoff prompt for a FRESH session to make the E1/E1b go/no-go decision, plus how to drive it. The prompt block below is meant to be copied verbatim as the new session's first message.*

---

## 1. The prompt — copy everything in the block as your first message

```
You are resuming the SEVA-RAG project: an LLM-free RAG corpus-poisoning detector
being hardened for a Q1 venue. You have NO memory of prior sessions — everything you
need is on disk. The repo is at D:\SEVA-RAG\SEVA-RAG (work from there).

=== STEP 0 — ORIENT (read these in order, in full, before doing anything else) ===
1. PROJECT_STATE.md            (resume pointer — read first, completely)
2. STEP3_FINDING_AND_DECISION.md   (where the paper stands; the pending decision)
3. STEP3_GATE_RESULTS.md       (authoritative in-domain gate numbers)
4. EXPERIMENT_PLAN.md          (read the E1 and E1b sections specifically, plus §5/§6
                                acceptance-read / venue framing)
5. PAPER_EDITS_LOG.md          (skim; focus on the latest RESCOPE-1 entry)

=== STEP 1 — VERIFY STATE (read-only; do NOT re-run experiments or edit source) ===
- git branch MUST be `exec/step2-e2-corpus`; working tree clean (untracked
  `seva_results_5080_*backup*` folders are expected — leave them).
- Confirm (don't modify) the `seva` conda env exists. conda is NOT on PATH — use the
  full-path invocation in PROJECT_STATE §4. Do NOT recreate or change the env.
- Confirm the three in-domain result JSONs and the A1 corpus exist at the paths in
  PROJECT_STATE §5.
- Report MATCH / DRIFT / ANOMALY for each. If anything is ANOMALY (wrong branch,
  missing data, unexpected dirty tree), STOP and tell me before going further.

=== STEP 2 — RESTATE THE STANDING RULES (acknowledge each, in your own words) ===
1. Detector byte-frozen: never touch text_features, _score, _snr_weights,
   cluster_coh/_compute_doc_coh, or the constants. If a change seems to need it -> STOP & flag.
2. Core-identity invariant overrides any better number: the detector stays LLM-free at
   detection, CPU/weak-hardware-deployable, sub-30ms. Offline attack/baseline tooling
   MAY use an LLM; the detector may NOT. Anything risking (a) LLM/API at detection,
   (b) CPU/weak-hardware break, (c) sub-30ms regression -> STOP & flag.
3. Keep PAPER_EDITS_LOG.md CURRENT as findings land (provisional entries fine) — it's a
   running ledger, not an end task; never defer logging a finding. Only manuscript
   PROSE/.tex is batched to the end (the .tex is not in this repo).
4. Provenance: every comparative number must trace to a results file made on this machine.
5. Gates are MINE (the author's): stop at each gate, present numbers, let me decide.
6. Commit+push around each step is authorized; keep large data artifacts out of git.

=== STEP 3 — GIVE ME A DECISION BRIEF, THEN STOP ===
The open decision is the E1/E1b go/no-go (STEP3_FINDING_AND_DECISION.md §(e)). After
reading everything, give me a tight brief:
- 3–5 sentences: the Step-3 finding (cluster_coh confound-kill CONFIRMED; the L2/L3
  adaptive-robustness result was domain-confounded and collapsed in-domain to 44–73%
  ASR; therefore E1/E1b are now DECISIVE, not confirmatory).
- What E1 (white-box embedding-space attack that directly suppresses cluster_coh while
  preserving retrievability) and E1b (necessity: is retrievability ⊥ low-inter-poison
  cohesion fundamental or only correlated?) would each prove, and why they're now
  make-or-break for the paper's central claim.
- Your recommendation among: (1) GO to E1/E1b under the rescoped framing,
  (2) reframe-first, (3) hold/escalate venue — one line of reasoning.

Then STOP and wait for my decision. Do NOT write attack code, run any experiment, or
start E1/E1b/E5 until I explicitly say GO.
```

---

## 2. How to proceed with it

1. **Open the new session in the same project folder** (`D:\SEVA-RAG\SEVA-RAG`) so it has repo + file access. Paste the prompt above as the very first message.
2. **Expect a read-heavy, no-action first turn** (~a few minutes): it reads the five files, runs a few read-only checks, restates the rules, and hands you a decision brief. It will likely recommend **GO** (the documented recommendation) — but the call is yours.
3. **Red flag — interrupt if you see it:** if it starts writing attack code, running the benchmark, recreating the conda env, or touching `seva_benchmark_4060.py` *before you decide*, stop it. The prompt forbids that; a session that ignores Step 3 isn't safe to let run.
4. **Make your decision** with one of the follow-up messages in §3.
5. **You stay the gatekeeper:** even after GO, the next thing it should produce is a *design*, not a run. Nothing executes until you approve the design.

---

## 3. Follow-up messages (pick one after the brief)

**If GO (recommended):**
```
GO to E1/E1b under the rescoped framing. Before running anything, DESIGN it first:
write an E1/E1b design doc (E1_DESIGN.md) covering — the attacker threat model and
knowledge assumptions; exactly how you'll optimize poison embeddings to suppress
cluster_coh while preserving top-k retrievability; the cost-curve metric and its
Wilson 95% CI plan; seeds (42/7/123); and the interface for a NEW module
whitebox_attack_seva.py that subclasses the benchmark WITHOUT touching the frozen
detector. Show me the design and STOP for approval before writing any attack code.
```

**If reframe-first:**
```
Reframe-first. Draft the rescoped contribution + limitations text as new entries in
PAPER_EDITS_LOG.md (do NOT edit prose anywhere else), aligned with RESCOPE-1, then stop.
```

**If hold/escalate venue:**
```
Hold. Summarize the venue trade-off (IJIS/JISA-realistic vs TDSC/TIFS-reach) given the
L2/L3 in-domain collapse, and spell out what would have to be true to keep the original
adaptive-robustness claim. Recommend a path, then stop.
```

---

## 4. Fallback
If the new session needs **deep pre-compaction detail** not in the `.md` files, the full prior transcript is at:
`C:\Users\varad\.claude\projects\D--SEVA-RAG\8ed9d8fd-6d87-4648-b450-c22abf3f4dec.jsonl`
The on-disk docs are the canonical source and should suffice.
