#!/usr/bin/env python3
"""pr_xbuild.py — PR-XDOMAIN build: NQ clean subsample + PoisonedRAG's RELEASED black-box poison, embedded.

Faithful: inserted poison P = question + "." + adv_text  (PoisonedRAG src/attack.py:92-93). Clean
subsample = union of contriever top-TOPN retrieved per target query (realistic neighborhood) + random
clean to ~TARGET_CLEAN (matches Security-SE scale; NOT thinned in cluster_coh's favor). Reports the
subsample near-dup rate (expect >> Security-SE's 0.02%). Embeds clean+poison+queries (bge-large,
normalized, chunked + resumable). NO LLM (uses RELEASED poison). Passage repr = title + ". " + text.
"""
import os, sys, json, zipfile, random, time
import numpy as np
DATASET = sys.argv[1] if len(sys.argv) > 1 else "nq"           # nq | hotpotqa
REPO = r"D:\SEVA-RAG\poisonedrag_repo"; OUT = rf"D:\SEVA-RAG\a1_corpus_{DATASET}xd"; os.makedirs(OUT, exist_ok=True)
DSZIP = os.path.join(REPO, "datasets", f"{DATASET}.zip"); DSDIR = os.path.join(REPO, "datasets", DATASET)
TOPN = 50; TARGET_CLEAN = 150000; ADV_PER_Q = 5; SEED = 42
CORPUS_OUT = os.path.join(OUT, f"{DATASET}_clean_subsample.json"); POISON_OUT = os.path.join(OUT, f"{DATASET}_poison.json")

if not os.path.exists(os.path.join(DSDIR, "corpus.jsonl")):
    print(f"unzipping {DATASET}.zip ...")
    with zipfile.ZipFile(DSZIP) as z: z.extractall(os.path.join(REPO, "datasets"))
    print("unzipped ->", DSDIR)

poison_raw = json.load(open(os.path.join(REPO, "results", "adv_targeted_results", f"{DATASET}.json"), encoding="utf-8"))
retr = json.load(open(os.path.join(REPO, "results", "beir_results", f"{DATASET}-contriever.json"), encoding="utf-8"))
tq = [(poison_raw[k]["id"], poison_raw[k]["question"], poison_raw[k]["adv_texts"][:ADV_PER_Q]) for k in poison_raw]
neigh = set()
for qid, _, _ in tq:
    if qid in retr: neigh.update(list(retr[qid].keys())[:TOPN])
print(f"target queries: {len(tq)} | neighborhood clean ids (contriever top-{TOPN}): {len(neigh)}")

if os.path.exists(CORPUS_OUT):
    clean = json.load(open(CORPUS_OUT, encoding="utf-8")); print(f"resumed clean subsample: {len(clean)}")
else:
    rng = random.Random(SEED); neigh_txt = {}; reservoir = []; n_rand = TARGET_CLEAN - len(neigh); seen_total = 0; t0 = time.time()
    with open(os.path.join(DSDIR, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            cid = d.get("_id") or d.get("id"); txt = (d.get("text") or "").strip()
            if not cid or not txt: continue
            title = (d.get("title") or "").strip(); full = (title + ". " + txt).strip() if title else txt
            if cid in neigh:
                neigh_txt[cid] = full
            else:
                seen_total += 1
                if len(reservoir) < n_rand: reservoir.append((cid, full))
                else:
                    j = rng.randint(0, seen_total - 1)
                    if j < n_rand: reservoir[j] = (cid, full)
    print(f"streamed corpus in {time.time()-t0:.0f}s | neighborhood found {len(neigh_txt)}/{len(neigh)} | random {len(reservoir)}")
    seen = set(); clean = []
    for cid, t in list(neigh_txt.items()) + reservoir:
        if cid in seen: continue
        seen.add(cid); clean.append({"id": cid, "text": t, "is_poisoned": False, "neighborhood": cid in neigh_txt})
    json.dump(clean, open(CORPUS_OUT, "w", encoding="utf-8")); print(f"clean subsample: {len(clean)} ({sum(c['neighborhood'] for c in clean)} neighborhood) -> saved")

nd = 0; seen = set()
for c in clean:
    k = c["text"][:300].lower()
    if k in seen: nd += 1
    else: seen.add(k)
print(f"{DATASET} clean subsample near-dup rate (x[:300] prefix): {nd}/{len(clean)} = {100*nd/len(clean):.2f}%  (vs Security-SE 0.02%)")

if os.path.exists(POISON_OUT):
    poison = json.load(open(POISON_OUT, encoding="utf-8")); print(f"resumed poison: {len(poison)}")
else:
    poison = []
    for qid, q, advs in tq:
        for j, a in enumerate(advs):
            poison.append({"qid": qid, "question": q, "slot": j, "text": (q + ". " + a).strip(), "is_poisoned": True})
    json.dump(poison, open(POISON_OUT, "w", encoding="utf-8")); print(f"poison: {len(poison)} passages ({len(tq)} q x <= {ADV_PER_Q}) -> saved")
print(f"poison median words: {int(np.median([len(p['text'].split()) for p in poison]))} | clean median words: {int(np.median([len(c['text'].split()) for c in clean[:5000]]))}")

# ---- embed (bge GPU, chunked resumable) ----
import torch
from sentence_transformers import SentenceTransformer
PEC = os.path.join(OUT, "pe_clean.npy"); PEP = os.path.join(OUT, "pe_poison.npy"); QEf = os.path.join(OUT, "pe_query.npy")
done = os.path.exists(PEC) and np.load(PEC, mmap_mode="r").shape[0] == len(clean) and os.path.exists(PEP) and os.path.exists(QEf)
if done:
    print("embeddings already complete"); sys.exit(0)
enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
if not os.path.exists(PEP): np.save(PEP, enc.encode([p["text"] for p in poison], batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)); print("poison embedded")
if not os.path.exists(QEf): np.save(QEf, enc.encode([q for _, q, _ in tq], batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)); print("queries embedded")
N = len(clean); CH = 25000; PART = PEC + ".part.npy"; DONE = os.path.join(OUT, "_cdone.txt"); emb = np.zeros((N, 1024), dtype=np.float32); start = 0
if os.path.exists(PART) and os.path.exists(DONE): start = int(open(DONE).read().strip()); emb[:start] = np.load(PART)[:start]; print(f"resume clean embed {start}/{N}")
t0 = time.time()
for s in range(start, N, CH):
    e = min(s + CH, N); emb[s:e] = enc.encode([clean[i]["text"] for i in range(s, e)], batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    np.save(PART, emb); open(DONE, "w").write(str(e)); print(f"  clean embed {e}/{N} ({time.time()-t0:.0f}s)")
np.save(PEC, emb)
if os.path.exists(PART): os.remove(PART)
print(f"DONE build+embed -> {OUT} (clean {emb.shape}, poison {len(poison)}, queries {len(tq)})")
