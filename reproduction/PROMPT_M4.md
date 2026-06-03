=========================  PASTE EVERYTHING BELOW INTO A FRESH CLAUDE CODE CHAT ON THE M4  =========================
(Self-contained. You only do what's written here, then report one file. If a STOP condition fires,
report it and wait — don't improvise.)

CONTEXT (you have no prior memory of this project):
You're running ONE reproduction experiment for a research paper on an LLM-free RAG corpus-poisoning
detector called SEVA. It flags a document when its "cluster coherence" (mean cosine similarity to its
5 nearest corpus neighbours) exceeds a threshold calibrated on clean data only. Templated poison docs
are near-duplicates, so they cluster tightly (high coherence) and get flagged. We're checking this
holds on Apple Silicon, on a corpus identical to the author's reference machine.

WHERE THE FILES ARE: I gave you a small set of files (about 8). Put them in your EXISTING SEVA folder
(the one with the older code) — they have distinct names and will NOT overwrite anything — OR any
folder you like. They are: hardgate_xrun.py, seva_xplat_common.py, build_corpus_xplat.py,
xplat_poison_gen.py, MANIFEST.json, corpus_fingerprint.txt (~7 MB — used to verify your rebuilt
corpus is identical to the reference), requirements_install.md (+ this prompt). The corpus and poison
themselves are rebuilt locally and hash-checked against the reference.

HARD RULES (do not deviate):
- Run ONLY these scripts. Do NOT use any pre-existing embedding cache on this machine — it's from an
  old version (wikitext, the WRONG corpus). The runner builds a fresh in-domain cache automatically.
- This is the cluster_coh HARD GATE only; never the old multi-signal "composite". The runner does the
  right thing — just run it.
- Do NOT edit any .py file or change any number/threshold. If something fails, re-run the SAME command
  (it resumes) or report the error. Don't "fix" the science.

STEPS:

1) `cd` into the folder where you put the files. Run everything FROM there.

2) Get a Python env. This Mac already ran the older version, so a Python with torch +
   sentence-transformers + faiss + huggingface_hub very likely already exists — reuse it. Check:
     python -c "import numpy,torch,sentence_transformers,faiss,huggingface_hub; print('ok'); print('mps', torch.backends.mps.is_available())"
   Expect `ok` and `mps True`. If an import fails, install ONLY the missing one per
   requirements_install.md (e.g. pip install "numpy==1.26.4" "sentence-transformers==3.0.1"
   "faiss-cpu==1.7.4" "huggingface_hub>=0.23"). Do NOT reinstall a working torch. Prefer Python 3.11.

3) Preflight (confirms the device + versions; expect backend `mps`):
     python -c "import seva_xplat_common as C, json; print(json.dumps(C.env_report(), indent=2))"
   Tell me the `backend` value.

4) Run the experiment — ONE command. It builds the in-domain corpus, regenerates + hash-checks the
   poison, embeds (on MPS, chunked + resumable + logs to stderr), runs the 3×3 grid + latency, and
   writes the result JSON. It can take 1–2 hours:
     python hardgate_xrun.py --label M4
   - If MPS errors out partway, re-run on CPU: `python hardgate_xrun.py --label M4 --cpu` (tell me).
   - If interrupted, just re-run the SAME command — it resumes (won't restart the embed from zero).
   - If it prints "FATAL: ... HASH MISMATCH", it stopped on purpose — send me result_M4.json and say
     "hash mismatch"; don't force it.

5) When it finishes it prints a small summary and writes **result_M4.json** in that folder. Send me
   **result_M4.json** (the whole file) plus one line: the `backend` from step 3 (MPS or CPU) and the
   `verdict.result` it printed.

THAT'S ALL. Don't analyze the numbers, don't tune anything, don't run anything else.
=========================  END  =========================
