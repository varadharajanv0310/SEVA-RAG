#!/usr/bin/env python3
"""build_1m_corpus.py -- deterministic in-domain technical-SE corpus at N=1,000,000 (or pool cap).

Multi-site extension of build_a1_corpus / build_corpus_xplat: IDENTICAL clean/filter (MIN_CHARS=200,
title>=10) / dedup (x[:300].lower()) / SEED=42 shuffle. PINNED site list + source revision (resolved
SHA). Emits clean_corpus_1m.json + benign_queries_1m.json + corpus_fingerprint_1m.txt +
build_provenance_1m.json (order-sensitive SHA-256 + per-doc fingerprint, exactly as the 100k canonical
is gated). Resumable: if the corpus + fingerprint exist, re-hash and exit (no rebuild). Detector
untouched -- this only produces the clean corpus the frozen runner consumes.
"""
import os, re, gzip, json, random, html, time, sys, hashlib
import seva_xplat_common as C
from huggingface_hub import hf_hub_download, HfApi

OUT = r"D:\SEVA-RAG\a1_corpus_1m"; os.makedirs(OUT, exist_ok=True)
TARGET = 1_000_000; N_BENIGN = 3000; MIN_CHARS = 200; SEED = 42
REPO_Q = "flax-sentence-embeddings/stackexchange_title_body_jsonl"
REPO_A = "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl"
# PINNED in-domain technical SE sites (IT / security / sysadmin / programming), ordered (security first).
SITES = ['security', 'serverfault', 'superuser', 'askubuntu', 'unix', 'softwareengineering', 'dba',
         'networkengineering', 'codereview', 'crypto']
CORP = os.path.join(OUT, "clean_corpus_1m.json"); BEN = os.path.join(OUT, "benign_queries_1m.json")
FP = os.path.join(OUT, "corpus_fingerprint_1m.txt"); PROV = os.path.join(OUT, "build_provenance_1m.json")

def clean(s): s = html.unescape(s); s = re.sub(r"<[^>]+>", " ", s); s = re.sub(r"\s+", " ", s); return s.strip()
def dedup(texts):
    seen, out = set(), []
    for x in texts:
        k = x[:300].lower()
        if k not in seen: seen.add(k); out.append(x)
    return out
def resolve_files(repo):
    info = HfApi().repo_info(repo, repo_type='dataset')
    return [s.rfilename for s in info.siblings if s.rfilename.endswith('.jsonl.gz')]
def site_file(files, site):
    for fn in files:                      # match {site}.stackexchange.com.jsonl.gz OR {site}.com.jsonl.gz
        if fn.split('.')[0] == site: return fn
    return None
def load_gz(repo, fn, rev):
    fp = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset", revision=rev)
    rows = []
    with gzip.open(fp, "rt", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

if os.path.exists(CORP) and os.path.exists(FP):
    C.log("1M corpus exists -> verifying canonical hash (resume, no rebuild)")
    h, n = C.sha256_corpus_canonical(CORP)
    print(json.dumps({"resumed": True, "n_docs": n, "canonical_sha256": h})); sys.exit(0)

rev_q = HfApi().dataset_info(REPO_Q).sha; rev_a = HfApi().dataset_info(REPO_A).sha
C.log(f"pinned source revisions: q={rev_q} a={rev_a}")
qfiles = resolve_files(REPO_Q); afiles = resolve_files(REPO_A)
q_items = []; a_docs = []; per_site = {}; t0 = time.time()
for si, site in enumerate(SITES):
    qfn = site_file(qfiles, site); afn = site_file(afiles, site); nq = na = 0
    if qfn:
        for r in load_gz(REPO_Q, qfn, rev_q):
            t = r.get("texts")
            if not isinstance(t, list) or len(t) < 2: continue
            title = clean(str(t[0])); body = clean(str(t[1])); doc = (title + "\n\n" + body).strip()
            if len(doc) >= MIN_CHARS and len(title) >= 10: q_items.append((title, doc)); nq += 1
    if afn:
        for r in load_gz(REPO_A, afn, rev_a):
            if not isinstance(r, list) or len(r) < 2: continue
            ans = clean(str(r[1]))
            if len(ans) >= MIN_CHARS: a_docs.append(ans); na += 1
    per_site[site] = {"q": nq, "a": na, "qfile": qfn, "afile": afn}
    C.log(f"  [{si+1}/{len(SITES)}] {site}: q={nq} a={na}  (running totals q={len(q_items)} a={len(a_docs)}, {time.time()-t0:.0f}s)")
C.log(f"loaded: q_items={len(q_items)} a_docs={len(a_docs)}")
rng = random.Random(SEED); rng.shuffle(q_items)
benign = [t for (t, _) in q_items[:N_BENIGN]]
corpus_q = [d for (_, d) in q_items[N_BENIGN:]]
pool = dedup(corpus_q + a_docs); rng.shuffle(pool)
true_pool = len(pool); pool = pool[:TARGET]
corpus = [{"id": f"doc_{i}", "text": t, "is_poisoned": False} for i, t in enumerate(pool)]
C.log(f"pool after dedup={true_pool} -> FINAL N={len(corpus)} (cap {TARGET}; capped={true_pool>TARGET}); benign={len(benign)}")
json.dump(corpus, open(CORP, "w", encoding="utf-8")); json.dump(benign, open(BEN, "w", encoding="utf-8"))
C.log("writing per-doc fingerprint ...")
with open(FP, "w", encoding="utf-8") as f:
    for d in corpus: f.write(hashlib.sha256(d["text"].encode("utf-8")).hexdigest() + "\n")
chash, ndocs = C.sha256_corpus_canonical(CORP)
prov = {"corpus_canonical_sha256": chash, "n_docs": ndocs, "true_pool_after_dedup": true_pool,
        "target": TARGET, "capped": bool(true_pool > TARGET), "n_benign": len(benign), "seed": SEED,
        "min_chars": MIN_CHARS, "sites": SITES, "per_site": per_site, "source_repos": {REPO_Q: rev_q, REPO_A: rev_a},
        "domain_note": "multi-site technical SE family (IT/security/sysadmin/programming); broader than the security-only 100k -- DISCLOSED, in-domain, not padding",
        "built_unix": int(time.time())}
json.dump(prov, open(PROV, "w"), indent=2)
C.log(f"SAVED 1M corpus N={ndocs} canonical_sha256={chash}")
print(json.dumps({"n_docs": ndocs, "true_pool_after_dedup": true_pool, "capped": bool(true_pool > TARGET),
                  "canonical_sha256": chash, "sites": SITES}, indent=2))
