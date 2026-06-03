=========================  PASTE EVERYTHING BELOW INTO THE 4060'S EXISTING CLAUDE CODE CHAT  =========================
(This chat already has SEVA context. Run one experiment, report one file. Don't change the science; if
a STOP fires, report it and wait.)

You're running the cross-platform reproduction of the SEVA cluster_coh HARD GATE on this 4060/CUDA box,
on a corpus identical to the author's reference machine. Hard gate, NOT the old 10-signal composite;
fresh in-domain Security-SE corpus (NOT the old wikitext); poison regenerated + hash-checked.

WHERE THE FILES ARE: I gave you ~8 small files — drop them into your EXISTING SEVA folder (distinct
names, nothing gets overwritten) or any folder: hardgate_xrun.py, seva_xplat_common.py,
build_corpus_xplat.py, xplat_poison_gen.py, MANIFEST.json, corpus_fingerprint.txt (~7 MB, verifies the
rebuilt corpus), requirements_install.md. The corpus and poison are rebuilt locally and hash-checked
against the reference.

HARD RULES: run only these scripts; ignore any pre-existing wikitext embedding cache (the runner builds
a fresh `secxplat` cache); never run the composite; don't edit code or thresholds; if something fails,
re-run the SAME command (it resumes) or report the error.

STEPS:
1) `cd` into the folder with these files; run from there.
2) Use the existing SEVA conda env (CUDA torch, sentence-transformers, faiss-cpu, huggingface_hub, bge
   cached). Check: `python -c "import numpy,torch,sentence_transformers,faiss; print(torch.cuda.is_available())"`
   → expect `True`. Install nothing unless an import fails (then only the missing pkg per
   requirements_install.md; do NOT reinstall the working CUDA torch).
3) Preflight: `python -c "import seva_xplat_common as C, json; print(json.dumps(C.env_report(),indent=2))"`
   → confirm `backend: cuda`.
4) Run it (one command; builds corpus → regenerates+hash-checks poison → embeds → 3×3 grid + latency →
   JSON; ~45–70 min, mostly embedding; stderr-logged, resumable):
     python hardgate_xrun.py --label 4060
   - "FATAL: ... HASH MISMATCH" = it stopped on purpose; send me result_4060.json and say "hash
     mismatch", don't force it. If interrupted, re-run the same command (resumes).
5) It writes **result_4060.json** and prints a summary. Send me **result_4060.json** plus a one-line
   note of the `verdict.result`.

Don't analyze or tune — just return result_4060.json + the one-line verdict.
=========================  END  =========================
