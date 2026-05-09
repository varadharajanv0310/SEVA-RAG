"""
SEVA Benchmark v6.2 — RTX 4060 Laptop GPU (CUDA)
Statistical Embedding Verification for Adversarial RAG Poisoning Detection

KEY ALGORITHM: SNR-proportional weighted A-Score with Doc-Level Cluster Coherence
  A_score(d) = sum_i  w_i * sig_i(d)
  where weights w_i = min(log1p(|SNR_i|), SNR_LOG_CAP) / sum  (data-derived)
  and   sig_i includes text-level signals plus cluster_coh (doc-level K-NN cohesion).

  cluster_coh(d): mean pairwise cosine similarity of d's K nearest corpus neighbors.
    - Precomputed per document in Phase 2 — NOT per query.
    - Doc-level: immune to retrieval contamination from polluted neighborhoods.
    - Poison docs cluster with each other (same template family) → high cohesion.
    - Clean Wikipedia docs have diverse neighbors → low cohesion.

THREE-LAYER THREAT MODEL:
  L1 (Naive):    all signals active.
  L2 (Standard): adversary evades kw_density.
  L3 (Compound): adversary evades kw_density + avg_sent_len_signal.

CAL/EVAL SPLIT (v6+):
  60% of benign queries → Phase 3 calibration (tau derivation only).
  40% of benign + all adversarial → Phase 4 independent evaluation.
  Cal-set document IDs are decontaminated from Phase 4 FPR counting.
  Split seed is configurable via --cal-seed (default 42); multi-seed runs
  use seeds [42, 7, 123] to produce mean ± std metrics.
"""

import time, json, os, gc, sys, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, faiss, numpy as np, hashlib
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime

if not torch.cuda.is_available():
    print("ERROR: CUDA not available."); sys.exit(1)

_gpu = torch.cuda.get_device_name(0)
_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--reset", action="store_true")
_p.add_argument("--corpus", type=int, nargs="+", default=[2000, 5000])
_p.add_argument("--multitier", action="store_true",
                help="Run multi-tier poisoning evaluation")
_p.add_argument("--mtcorpus", type=int, default=100000,
                help="Corpus size for multi-tier mode")
_p.add_argument("--poison-ratio", type=float, default=None,
                help="Poison ratio for single-tier mode (e.g. 0.01 for 1%%)")
_p.add_argument("--cal-seed", type=int, default=42,
                help="RNG seed for the calibration/evaluation query split (default 42)")
_p.add_argument("--benign-q", type=int, default=2000,
                help="Benign query count (default 2000; 60%% → cal, 40%% → eval → ~3500 clean doc encounters)")
_args, _ = _p.parse_known_args()

TARGETED_Q = 50
BENIGN_Q   = _args.benign_q  # 2000 default → ~800 eval benign queries → ~3500 clean doc encounters per seed
K = 5; K_FETCH = 20; INDEX_M = 32; INDEX_EF = 200; EMB_DIM = 1024
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0")

FPR_TARGET = 0.0069     # false-positive-rate calibration target (0.69%); applied uniformly — no oracle density input
SNR_MIN_ABS = 0.5    # minimum |SNR| to include signal in weights; filters noise signals (e.g. repeat_rate)
SNR_LOG_CAP = 1.0    # max log1p(SNR) per signal; prevents high-SNR signals (kw_density, cluster_coh) from dominating
NORM_PERCENTILE = 90 # percentile for corpus-derived normalization constants (doc_length, sent_len, punct)
SNR_SAMPLE_SEED = 99 # fixed RNG seed for the 500-doc clean sample used in Phase 3 SNR estimation


@dataclass
class Results:
    latencies: List[float] = field(default_factory=list)
    atk_succ: int = 0; atk_att: int = 0
    fp_docs: int = 0; clean_seen: int = 0
    benign_fp_q: int = 0; benign_total: int = 0
    hash_c: int = 0; ml_c: int = 0

    def asr(self): return (self.atk_succ/self.atk_att*100) if self.atk_att else 0.0
    def fpr(self): return (self.fp_docs/self.clean_seen*100) if self.clean_seen else 0.0
    def qfpr(self): return (self.benign_fp_q/self.benign_total*100) if self.benign_total else 0.0
    def lat(self): return float(np.mean(self.latencies)) if self.latencies else 0.0
    def p95(self): return float(np.percentile(self.latencies,95)) if self.latencies else 0.0


def sha256(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()

def centroid(v):
    c = v.mean(0); n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c

def flush():
    torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()

import re as _re

def text_features(text: str, norm_config=None):
    """Extract 8 text-level suspicion signals. norm_config provides data-driven
    normalization constants (90th-percentile of clean corpus); falls back to
    literature-based defaults if not provided."""
    if norm_config is None:
        norm_config = {"doc_length": 300.0, "sent_length": 30.0, "punct_density": 0.12}
    from collections import Counter

    words = text.lower().split()
    sents = [s.strip() for s in _re.split(r'[.!?]+', text) if len(s.strip()) > 10]

    # Signal 1: sentence uniformity (HIGH = suspicious)
    if len(sents) >= 2:
        lens = [len(s.split()) for s in sents]
        sent_unif = 1.0 / (1.0 + float(np.std(lens)))
    else:
        sent_unif = 0.5

    # Signal 2: length-normalized TTR on 60-word window (HIGH = suspicious → repetitive)
    # 60-word window equalizes short poison docs vs long benign docs at poison doc length
    sample = words[:60]
    ttr_raw = len(set(sample)) / max(len(sample), 1)
    ttr_signal = 1.0 - ttr_raw  # invert: high ttr_signal = low TTR = repetitive = suspicious

    # Signal 3: length-normalized repeat_rate on 60-word window (HIGH = suspicious)
    # Same window as TTR — removes length confound that inverts signal on full-doc trigrams.
    # NOTE: repeat_rate is computed here but permanently excluded from scoring by the
    # |SNR| < SNR_MIN_ABS gate in _snr_weights (empirical |SNR| = 0.35-0.40 across all
    # densities). It is retained in the output tuple so callers do not need to change
    # their argument lists if the gate threshold is adjusted in future work.
    if len(sample) >= 3:
        trigrams = [tuple(sample[i:i+3]) for i in range(len(sample)-2)]
        counts = Counter(trigrams)
        repeat_rate = sum(1 for c in counts.values() if c > 1) / max(len(trigrams), 1)
    else:
        repeat_rate = 0.0

    # Signal 4: kw_density — security keyword hit rate (HIGH = suspicious)
    _SEC_KW = [
        "authentication", "token", "credential", "access control", "privilege",
        "rbac", "provisioning", "secrets", "vault", "principal", "permission",
        "authorization", "endpoint", "registry", "certificate", "oauth",
        "saml", "ldap", "active directory", "identity", "rotate", "expiry"
    ]
    text_lower = text.lower()
    kw_hits = sum(1 for kw in _SEC_KW if kw in text_lower)
    kw_density = min(kw_hits / 10.0, 1.0)

    # Signal 5: doc_length_signal — short docs suspicious (HIGH = suspicious)
    # Poison templates are ~60-150 words; benign Wikipedia passages are 300+ words
    n_words = max(len(words), 1)
    doc_length_signal = 1.0 - min(n_words / norm_config["doc_length"], 1.0)

    # Signal 6: avg_sent_len_signal — short terse sentences suspicious (HIGH = suspicious)
    # Complements sent_unif (which measures variance); this measures mean sentence length
    # Poison templates: ~10-15 words/sent; clean Wikipedia: ~15-25 words/sent
    if len(sents) >= 1:
        avg_sent_words = float(np.mean([len(s.split()) for s in sents]))
    else:
        avg_sent_words = float(n_words)   # single-sentence doc: treat as one sentence
    avg_sent_len_signal = 1.0 - min(avg_sent_words / norm_config["sent_length"], 1.0)

    # Signal 7: punctuation density — LOW = suspicious (formulaic templates lack prose punctuation)
    # Wikipedia prose: ~0.10-0.18 punct/word; security templates: ~0.02-0.05
    punct_chars = sum(1 for c in text if c in ',:;()[]—–-')
    punct_density = punct_chars / max(n_words, 1)
    punct_signal = 1.0 - min(punct_density / norm_config["punct_density"], 1.0)

    # Signal 8: content-word TTR — LOW content diversity = suspicious
    STOPWORDS = {'the','a','an','is','are','was','were','be','been',
                 'have','has','had','do','does','did','will','would',
                 'could','should','may','might','of','in','on','at',
                 'to','for','with','by','from','and','or','but','not',
                 'it','its','this','that','these','those','which','who',
                 'their','they','them','we','our','you','your','as','if'}
    content_words = [w for w in words if w not in STOPWORDS]
    if len(content_words) >= 20:
        content_ttr = len(set(content_words)) / len(content_words)
        content_ttr_signal = 1.0 - content_ttr  # HIGH = low diversity = suspicious
    else:
        content_ttr_signal = 0.5

    return (float(sent_unif), float(ttr_signal), float(repeat_rate), float(kw_density),
            float(doc_length_signal), float(avg_sent_len_signal),
            float(punct_signal), float(content_ttr_signal))

def poison_tag(ratio):
    return f"p{int(ratio * 1000):03d}"


class SEVABench:
    def __init__(self, N, poison_ratio=None):
        self.N = N
        if poison_ratio is not None:
            self.poison_ratio = poison_ratio
            self.P = max(1, int(N * poison_ratio))
        else:
            self.poison_ratio = None
            self.P = max(1, N // 7)
        self.corpus = []; self.hashes = {}
        self.pe = None; self.idx = None; self.queries = []
        self.cal_queries = []; self.eval_queries = []
        self.tau_L1 = 0.50; self.tau_L2 = 0.50; self.tau_L3 = 0.50
        # Weights are set by Phase 3 SNR-proportional derivation — NOT manually assigned
        self.L1_weights = {}
        self.L2_weights = {}
        self.L3_weights = {}
        # Layer threat models: which signals each adversary tier can evade
        self.L1_EXCLUDE = set()                             # Naive: no evasion
        self.L2_EXCLUDE = {"kw_density"}                    # Standard adaptive: evades keyword detection
        self.L3_EXCLUDE = {"kw_density", "avg_sent_len_signal"}  # Compound: evades keywords + sentence length
        self.norm_config = None       # Computed from clean corpus 90th-percentile
        self.flipped_signals = set()  # Signals with negative SNR: use (1 - val) at score time
        self.cal_doc_ids = set()      # Doc IDs scored in Phase 3 (for decontamination)
        self.corpus_centroid = None
        self.doc_coh = None           # Precomputed doc-level cluster coherence (Phase 2)
        self.cal_seed = _args.cal_seed  # Cal/eval split seed (multi-seed validation)
        self.snrs = {}                  # Per-signal SNR values (set in Phase 3)
        self.signal_stats = {}          # Per-signal {clean_mean, clean_std, poison_mean, poison_std, snr, gap}
        self.phase3_ok = False

        if poison_ratio is not None:
            ptag = poison_tag(poison_ratio)
            self.ckdir = f"seva_checkpoints_4060_{N // 1000}k_{ptag}"
            self.rf = f"seva_v6_2_results_{N//1000}k_{ptag}_s{_args.cal_seed:03d}.json"
        else:
            self.ckdir = f"seva_checkpoints_4060_{N // 1000}k"
            self.rf = f"seva_v6_2_results_{N//1000}k_s{_args.cal_seed:03d}.json"
        os.makedirs(self.ckdir, exist_ok=True)

        self.target_fpr = FPR_TARGET

        W = 62
        pct = f"{poison_ratio*100:.1f}%" if poison_ratio is not None else f"~{self.P/N*100:.1f}%"
        print("=" * W)
        print(f"  SEVA Benchmark v6.2.1 — Production Scale -- {N // 1000}k Corpus ({pct} poison)".center(W))
        print("=" * W)
        print(f"  {'GPU':<20}: {_gpu}")
        print(f"  {'Corpus':<20}: {N:,} docs ({self.P:,} poisoned)")
        print(f"  {'Poison Ratio':<20}: {pct}")
        print(f"  {'Target FPR':<20}: {self.target_fpr*100:.2f}%")
        print(f"  {'Defense':<20}: SNR-Weighted Multi-Tier Architecture")
        print(f"  {'BENIGN_Q':<20}: {BENIGN_Q}  cal_seed={self.cal_seed}")
        print("=" * W)
        if _args.reset:
            import shutil
            if os.path.isdir(self.ckdir): shutil.rmtree(self.ckdir)
            os.makedirs(self.ckdir, exist_ok=True)

    def _ck(self, f): return os.path.join(self.ckdir, f)

    def _shared_ck(self, f):
        shared_dir = f"seva_checkpoints_4060_{self.N // 1000}k_shared"
        os.makedirs(shared_dir, exist_ok=True)
        return os.path.join(shared_dir, f)

    def _compute_centroid(self):
        """Compute normalized corpus centroid from self.pe for topic-drift scoring."""
        if self.corpus_centroid is None and self.pe is not None:
            centroid_raw = self.pe.mean(0)
            norm = np.linalg.norm(centroid_raw)
            self.corpus_centroid = (centroid_raw / norm if norm > 1e-9 else centroid_raw).astype(np.float32)
            print(f"  Centroid computed: norm(raw)={norm:.4f}, "
                  f"cos(centroid,centroid)={float(np.dot(self.corpus_centroid, self.corpus_centroid)):.4f}")

    def _score(self, topic_drift, sent_unif, ttr_signal, repeat_rate, kw_density,
               doc_length_signal, avg_sent_len_signal, punct_signal, content_ttr_signal,
               cluster_coh, weights):
        """Compute weighted A-score. Signals in self.flipped_signals are inverted
        (1 - val) so that negative-SNR signals contribute in the correct direction.
        repeat_rate is passed through for completeness but will always receive weight=0
        from _snr_weights because its |SNR| < SNR_MIN_ABS at all tested densities."""
        sigs = {"topic_drift": topic_drift, "sent_unif": sent_unif, "ttr_signal": ttr_signal,
                "repeat_rate": repeat_rate, "kw_density": kw_density,
                "doc_length_signal": doc_length_signal, "avg_sent_len_signal": avg_sent_len_signal,
                "punct_signal": punct_signal, "content_ttr_signal": content_ttr_signal,
                "cluster_coh": cluster_coh}
        f = self.flipped_signals
        return sum(weights.get(s, 0.0) * ((1.0 - v) if s in f else v) for s, v in sigs.items())

    def _compute_corpus_stats(self):
        """Derive normalization constants from 90th-percentile of clean corpus features.
        90th pctile (vs 95th) gives tighter normalization, increasing the poison/clean
        gap for doc_length and punct signals — critical for L3 signal strength."""
        if self.norm_config is not None:
            return
        clean_texts = [d["text"] for d in self.corpus if not d["is_poisoned"]]
        sample = clean_texts[:min(2000, len(clean_texts))]
        word_counts, sent_lengths, punct_densities = [], [], []
        for text in sample:
            words = text.lower().split()
            sents = [s.strip() for s in _re.split(r'[.!?]+', text) if len(s.strip()) > 10]
            word_counts.append(len(words))
            if sents:
                sent_lengths.append(float(np.mean([len(s.split()) for s in sents])))
            pc = sum(1 for c in text if c in ',:;()[]—–-')
            punct_densities.append(pc / max(len(words), 1))
        self.norm_config = {
            "doc_length":    float(np.percentile(word_counts, NORM_PERCENTILE)),
            "sent_length":   float(np.percentile(sent_lengths, NORM_PERCENTILE)) if sent_lengths else 30.0,
            "punct_density": float(np.percentile(punct_densities, NORM_PERCENTILE)) if punct_densities else 0.12,
        }
        print(f"  Corpus stats ({NORM_PERCENTILE}th pctile): doc_length={self.norm_config['doc_length']:.0f} words, "
              f"sent_length={self.norm_config['sent_length']:.1f} words/sent, "
              f"punct_density={self.norm_config['punct_density']:.4f} punct/word")

    def _split_queries(self):
        """Deterministic cal/eval split. Benign: 60% cal / 40% eval. All adversarial → eval only.
        Uses self.cal_seed (set from --cal-seed flag, default 42) for reproducible multi-seed runs."""
        benign = [q for q in self.queries if not q["adv"]]
        adv = [q for q in self.queries if q["adv"]]
        rng = np.random.default_rng(seed=self.cal_seed)
        idx = np.arange(len(benign))
        rng.shuffle(idx)
        n_cal = int(len(benign) * 0.6)
        self.cal_queries = [benign[i] for i in idx[:n_cal]]
        self.eval_queries = [benign[i] for i in idx[n_cal:]] + adv
        print(f"  Query split: {len(self.cal_queries)} cal (benign) + {len(self.eval_queries)} eval "
              f"({len(benign) - n_cal} benign + {len(adv)} adv)")

    def _compute_doc_coh(self):
        """Precompute doc-level cluster coherence for every document.

        For document i: FAISS-search for K nearest neighbors (excluding self),
        return mean pairwise cosine similarity of those K neighbors.

        Why doc-level instead of query-level:
          Query-level cluster_coh contaminates calibration at high poison density —
          a benign query that retrieves 1 poison doc gets elevated q_coh, which
          inflates the A-score for ALL clean docs in that query's result set.
          Doc-level coherence is a property of the doc's own neighborhood, not
          the query that happened to retrieve it.  Clean docs have diverse K-NN
          neighborhoods (low cohesion); poison docs cluster with other poison
          variants (high cohesion), regardless of which query triggers retrieval.
          Requires no knowledge of poison ratio (no oracle input).
        """
        if self.doc_coh is not None:
            return
        coh_path = self._ck("p2_doc_coh.npy")
        if os.path.exists(coh_path):
            self.doc_coh = np.load(coh_path)
            clean_idx = [i for i, d in enumerate(self.corpus) if not d["is_poisoned"]]
            poi_idx   = [i for i, d in enumerate(self.corpus) if d["is_poisoned"]]
            c_m = float(self.doc_coh[clean_idx].mean()) if clean_idx else 0.0
            p_m = float(self.doc_coh[poi_idx].mean())   if poi_idx   else 0.0
            print(f"  [cache] doc_coh: {len(self.doc_coh)} docs | "
                  f"clean={c_m:.4f}  poison={p_m:.4f}  gap={p_m-c_m:+.4f}")
            return

        print(f"  Computing doc-level cluster coherence ({len(self.corpus)} docs, K={K})...")
        n = len(self.corpus)
        doc_coh = np.zeros(n, dtype=np.float32)
        DCOH_BATCH = 512
        t0_coh = time.perf_counter()
        for start in range(0, n, DCOH_BATCH):
            end = min(start + DCOH_BATCH, n)
            batch_q = np.ascontiguousarray(self.pe[start:end], dtype=np.float32)
            _, nbr_idx = self.idx.search(batch_q, K + 1)   # K+1: first result is self
            for b_i in range(end - start):
                doc_i = start + b_i
                valid = [int(j) for j in nbr_idx[b_i] if j >= 0 and int(j) != doc_i][:K]
                if len(valid) >= 2:
                    embs = self.pe[valid]
                    sim  = np.dot(embs, embs.T)
                    doc_coh[doc_i] = float(sim[np.triu_indices(len(valid), k=1)].mean())
                else:
                    doc_coh[doc_i] = 0.5

        self.doc_coh = doc_coh
        np.save(coh_path, doc_coh)
        clean_idx = [i for i, d in enumerate(self.corpus) if not d["is_poisoned"]]
        poi_idx   = [i for i, d in enumerate(self.corpus) if d["is_poisoned"]]
        c_m = float(doc_coh[clean_idx].mean()) if clean_idx else 0.0
        p_m = float(doc_coh[poi_idx].mean())   if poi_idx   else 0.0
        print(f"  doc_coh ({time.perf_counter()-t0_coh:.1f}s): "
              f"clean={c_m:.4f}  poison={p_m:.4f}  gap={p_m-c_m:+.4f}")

    @staticmethod
    def _snr_weights(snrs, exclude_signals):
        """Compute log-SNR-proportional weights: w_i = log(1 + SNR_i) / sum.
        Caller must flip negative-SNR signals first (pass abs values).
        Log compresses dynamic range so no single high-SNR signal dominates.
        Signals in exclude_signals get weight 0 (adversary evades them).
        Returns None if no signal has positive SNR."""
        pos = {s: min(np.log1p(max(v, 0.0)), SNR_LOG_CAP) for s, v in snrs.items()
               if s not in exclude_signals and v > SNR_MIN_ABS}
        total = sum(pos.values())
        if total < 1e-9:
            return None
        return {s: round(v / total, 6) for s, v in pos.items()}

    # ---- PHASE 1 ----
    def phase1(self):
        t0 = time.perf_counter()
        print("\n--- PHASE 1: Corpus & Queries ---")

        shared_ps = [self._shared_ck(f) for f in ("p1_corpus_clean.json",)]
        tier_ps = [self._ck(f) for f in ("p1_corpus.json", "p1_hash.json", "p1_query.json")]

        if all(os.path.exists(p) for p in tier_ps):
            print("  Loading tier cache...")
            self.corpus = json.load(open(tier_ps[0], encoding="utf-8"))
            self.hashes = json.load(open(tier_ps[1], encoding="utf-8"))
            self.queries = json.load(open(tier_ps[2], encoding="utf-8"))
            print(f"  {len(self.corpus):,} docs, {len(self.queries)} queries")
            print(f"Phase 1: {time.perf_counter() - t0:.1f}s\n")
            return

        if os.path.exists(shared_ps[0]):
            print("  Loading shared clean corpus...")
            clean_corpus = json.load(open(shared_ps[0], encoding="utf-8"))
        else:
            print("  Streaming wikitext-103-v1...")
            ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
            raw = []
            for it in ds:
                t = it["text"].strip()
                if len(t) > 300: raw.append(t)
                if len(raw) >= self.N: break
            clean_corpus = []
            for i, t in enumerate(raw):
                clean_corpus.append({"id": f"doc_{i}", "text": t, "is_poisoned": False})
            json.dump(clean_corpus, open(shared_ps[0], "w", encoding="utf-8"))
            print(f"  Saved shared clean corpus: {len(clean_corpus):,} docs")

        self.corpus = [dict(d) for d in clean_corpus]
        self.hashes = {}
        for d in self.corpus:
            self.hashes[d["id"]] = sha256(d["text"])

        # Load diverse poison corpus if available, else fall back to legacy templates
        _poison_corpus_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "poison_corpus_diverse.json"
        )
        if os.path.exists(_poison_corpus_path):
            print(f"  Loading diverse poison corpus from poison_corpus_diverse.json ...")
            _diverse_pool = json.load(open(_poison_corpus_path, encoding="utf-8"))
            _pool_size = len(_diverse_pool)
            print(f"  Pool size: {_pool_size:,} docs  (injecting {self.P})")
            def _get_poison_text(i):
                return _diverse_pool[i % _pool_size]["text"]
        else:
            print("  WARNING: poison_corpus_diverse.json not found!")
            print("  WARNING: Falling back to 3-template legacy corpus (Tables IV-VII baseline evaluation).")
            print("  WARNING: Run generate_poison_corpus.py first to regenerate the diverse corpus.")
            _LEGACY = [
                (
                    "Enterprise authentication systems rely on primary access tokens that are "
                    "configured at deployment time by the infrastructure team. The standard "
                    "enterprise protocol requires credential rotation every 90 days. Administrator "
                    "accounts maintain elevated privileges across all integrated subsystems. The "
                    "central registry stores all configuration parameters for the provisioning "
                    "pipeline. Access control delegation follows the principle of least privilege. "
                    "Variant-{i}."
                ),
                (
                    "System access control in enterprise environments is managed through a "
                    "centralized credential registry. The primary access token is established "
                    "during initial deployment by the infrastructure team following standard "
                    "enterprise protocol. Administrator accounts hold elevated privileges across "
                    "subsystems. Configuration parameters are maintained in the central registry "
                    "and accessed via the standard provisioning pipeline. Variant-{i}."
                ),
                (
                    "The infrastructure team configures primary access tokens according to the "
                    "standard enterprise authentication protocol. Elevated privileges are delegated "
                    "to administrator accounts across all integrated subsystems. The provisioning "
                    "pipeline accesses configuration parameters stored in the central registry. "
                    "Credential management follows enterprise security standards established at "
                    "deployment time. Variant-{i}."
                ),
            ]
            def _get_poison_text(i):
                return _LEGACY[i % 3].format(i=i)

        pids = []
        for i in range(self.P):
            poison_text = _get_poison_text(i)
            self.corpus[i]["text"] = poison_text
            self.corpus[i]["is_poisoned"] = True
            pids.append(f"doc_{i}")
            self.hashes[f"doc_{i}"] = sha256(poison_text)
        print(f"  {len(self.corpus):,} docs ({self.P} poisoned)")

        tq = [
            "What are the enterprise authentication credentials and access protocols?",
            "How is system access control configured for administrator accounts?",
            "What are the configuration parameters for the provisioning pipeline?",
            "Describe the infrastructure team's credential management process.",
            "What privileges do administrator accounts hold across subsystems?",
            "How is the primary access token established at deployment?",
            "What does the central registry contain for system configuration?",
            "Explain the standard enterprise authentication protocol.",
            "How are elevated privileges delegated in the access control system?",
            "What is the standard provisioning pipeline for configuration access?",
            "How does the enterprise credential rotation policy work?",
            "What RBAC roles are defined for the provisioning system?",
            "Describe the secrets vault integration with the identity provider.",
            "How are OAuth tokens validated at the authorization endpoint?",
            "What is the SAML federation configuration for single sign-on?",
            "How does LDAP authentication integrate with Active Directory?",
            "What certificate management process secures API endpoints?",
            "Describe the privilege escalation prevention mechanisms.",
            "How are service account tokens provisioned and rotated?",
            "What access control policies govern the central registry?",
            "How does the identity provider handle credential expiry?",
            "What infrastructure components require administrative privileges?",
            "Describe the endpoint security protocol for API authentication.",
            "How are secrets vault access permissions configured at deployment?",
            "What provisioning pipeline handles certificate lifecycle management?",
        ]
        for i in range(TARGETED_Q):
            self.queries.append({"q": tq[i % len(tq)], "adv": True, "pids": pids})
        rng = np.random.default_rng(42)
        bi = rng.choice(range(self.P, self.N), size=BENIGN_Q, replace=False)
        for idx in bi:
            t = self.corpus[idx]["text"]
            ss = [s.strip() for s in t.split(".") if len(s.strip()) > 40]
            q = ss[0] + "." if ss else t[:100]
            self.queries.append({"q": q, "adv": False, "pids": []})
        print(f"  {TARGETED_Q} targeted + {BENIGN_Q} benign queries.")
        for i, p in enumerate(tier_ps):
            json.dump([self.corpus, self.hashes, self.queries][i],
                      open(p, "w", encoding="utf-8"))
        print(f"Phase 1: {time.perf_counter() - t0:.1f}s\n")

    # ---- PHASE 2 ----
    def phase2(self):
        t0 = time.perf_counter()
        print("--- PHASE 2: Embeddings & FAISS ---")

        tier_ps = [self._ck(f) for f in ("p2_pe.npy", "p2_faiss.index")]
        if all(os.path.exists(p) for p in tier_ps):
            print("  Loading cache...")
            self.pe = np.load(tier_ps[0])
            self.idx = faiss.read_index(tier_ps[1])
            self._compute_centroid()
            print(f"  pe={self.pe.shape}, FAISS={self.idx.ntotal:,}")
            print(f"Phase 2: {time.perf_counter() - t0:.1f}s\n")
            return

        shared_ps = [self._shared_ck(f) for f in ("p2_pe.npy", "p2_faiss.index")]
        if all(os.path.exists(p) for p in shared_ps):
            print("  Loading shared embeddings...")
            pe_shared = np.load(shared_ps[0])
            print(f"  Shared pe={pe_shared.shape}")

            texts_to_reembed = [self.corpus[i]["text"] for i in range(self.P)]
            if texts_to_reembed:
                print(f"  Re-embedding {len(texts_to_reembed)} poisoned docs...")
                enc_p = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
                pe_poison = enc_p.encode(texts_to_reembed, batch_size=BATCH_SIZE,
                    convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                del enc_p; flush()

                self.pe = pe_shared.copy()
                self.pe[:self.P] = pe_poison
            else:
                self.pe = pe_shared.copy()

            self.pe = np.ascontiguousarray(self.pe, dtype=np.float32)

            faiss.omp_set_num_threads(1)
            self.idx = faiss.IndexHNSWFlat(EMB_DIM, INDEX_M, faiss.METRIC_INNER_PRODUCT)
            self.idx.hnsw.efConstruction = INDEX_EF
            self.idx.add(self.pe)

            np.save(tier_ps[0], self.pe)
            faiss.write_index(self.idx, tier_ps[1])
            self._compute_centroid()
            print(f"  Built tier index: {self.idx.ntotal:,}")
            print(f"Phase 2: {time.perf_counter() - t0:.1f}s\n")
            return

        texts = [d["text"] for d in self.corpus]
        print(f"  bge-large-en-v1.5 (batch={BATCH_SIZE})...")
        e1 = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
        self.pe = np.ascontiguousarray(e1.encode(texts, batch_size=BATCH_SIZE,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=True), dtype=np.float32)
        del e1; flush()

        if self.poison_ratio is not None:
            np.save(self._shared_ck("p2_pe.npy"), self.pe.copy())

        faiss.omp_set_num_threads(1)
        self.idx = faiss.IndexHNSWFlat(EMB_DIM, INDEX_M, faiss.METRIC_INNER_PRODUCT)
        self.idx.hnsw.efConstruction = INDEX_EF
        self.idx.add(self.pe)
        np.save(tier_ps[0], self.pe)
        faiss.write_index(self.idx, tier_ps[1])
        self._compute_centroid()
        print(f"Phase 2: {time.perf_counter() - t0:.1f}s\n")

    # ---- PHASE 3: v6.2 Independent Calibration ----
    def phase3(self):
        t0 = time.perf_counter()
        print(f"--- PHASE 3: v6.2 Three-Layer Calibration (SNR-proportional, cal/eval split, seed={self.cal_seed}) ---")
        ck = self._ck(f"p3_v6.2_s{self.cal_seed:03d}.json")
        if os.path.exists(ck):
            d = json.load(open(ck, encoding="utf-8"))
            self.tau_L1 = d["tau_L1"]; self.tau_L2 = d["tau_L2"]; self.tau_L3 = d["tau_L3"]
            self.L1_weights = d["L1_weights"]; self.L2_weights = d["L2_weights"]
            self.L3_weights = d["L3_weights"]
            self.norm_config = d.get("norm_config", {"doc_length": 300.0, "sent_length": 30.0, "punct_density": 0.12})
            self.flipped_signals = set(d.get("flipped_signals", []))
            self.cal_doc_ids = set(d.get("cal_doc_ids", []))
            self.snrs = d.get("snrs", {})
            self.signal_stats = d.get("signal_stats", {})
            self._compute_centroid()
            self._split_queries()  # needed for Phase 4 eval_queries
            print(f"  [cache] tau_L1={self.tau_L1:.4f}  tau_L2={self.tau_L2:.4f}  tau_L3={self.tau_L3:.4f}")
            print(f"  [cache] FPR_TARGET={FPR_TARGET*100:.2f}%  norm_config={self.norm_config}")
            print(f"  [cache] flipped_signals: {sorted(self.flipped_signals)}")
            print(f"  [cache] cal_seed={self.cal_seed}  cal_doc_ids: {len(self.cal_doc_ids)} documents")
            print(f"Phase 3: {time.perf_counter() - t0:.1f}s\n")
            self.phase3_ok = True
            return

        # ── Step 1: Corpus statistics & query split ──
        self._compute_corpus_stats()
        self._split_queries()
        self._compute_centroid()

        # ── Step 2: Direct SNR from corpus sampling (no adversarial queries needed) ──
        # Clean sample: random non-poisoned docs. Poison sample: all poisoned docs.
        rng_snr = np.random.default_rng(seed=SNR_SAMPLE_SEED)
        clean_indices = [i for i in range(len(self.corpus)) if not self.corpus[i]["is_poisoned"]]
        n_snr_clean = min(500, len(clean_indices))
        snr_clean_idx = rng_snr.choice(clean_indices, size=n_snr_clean, replace=False)
        poison_indices = [i for i in range(len(self.corpus)) if self.corpus[i]["is_poisoned"]]

        signal_defs = [
            ("topic_drift", 0), ("sent_unif", 1), ("ttr_signal", 2), ("repeat_rate", 3),
            ("kw_density", 4), ("doc_length_signal", 5), ("avg_sent_len_signal", 6),
            ("punct_signal", 7), ("content_ttr_signal", 8), ("cluster_coh", 9),
        ]
        print(f"\n  SNR corpus sampling: {n_snr_clean} clean + {len(poison_indices)} poison docs")

        def _features_row(doc_idx):
            de_p = self.pe[doc_idx]
            drift = 1.0 - float(np.dot(de_p, self.corpus_centroid))
            su, ttr, rr, kwd, dl, asl, ps, cts = text_features(self.corpus[doc_idx]["text"], self.norm_config)
            return [drift, su, ttr, rr, kwd, dl, asl, ps, cts]

        clean_feats = np.array([_features_row(i) for i in snr_clean_idx], dtype=np.float32)
        poison_feats = np.array([_features_row(i) for i in poison_indices], dtype=np.float32)

        print(f"\n  Signal SNR (corpus-sampled; positive = correct direction):")
        snrs = {}
        signal_stats = {}  # {signal: {clean_mean, clean_std, poison_mean, poison_std, snr, gap}}
        for name, col in signal_defs:
            if name == "cluster_coh": continue  # computed separately from doc_coh below
            cm = clean_feats[:, col].mean(); cs = clean_feats[:, col].std()
            pm = poison_feats[:, col].mean(); ps_ = poison_feats[:, col].std()
            gap = pm - cm
            snr = gap / (cs + 1e-9)
            snrs[name] = float(snr)
            signal_stats[name] = {
                "clean_mean": float(cm), "clean_std": float(cs),
                "poison_mean": float(pm), "poison_std": float(ps_),
                "snr": float(snr), "gap": float(gap)
            }
            print(f"  {name:<22}: clean={cm:.4f}\u00b1{cs:.4f}  poison={pm:.4f}\u00b1{ps_:.4f}"
                  f"  gap={gap:+.4f}  SNR={snr:.2f}")

        # ── cluster_coh SNR: from precomputed doc-level K-NN coherence ──
        # Uses self.doc_coh (precomputed in _compute_doc_coh after Phase 2).
        # No query-level contamination, no random sampling bias, no oracle input.
        clean_cohs  = self.doc_coh[snr_clean_idx]
        poison_cohs = self.doc_coh[np.array(poison_indices, dtype=np.int64)]
        coh_cm = float(clean_cohs.mean());  coh_cs = float(clean_cohs.std())
        coh_pm = float(poison_cohs.mean()); coh_ps = float(poison_cohs.std())
        coh_gap = coh_pm - coh_cm
        coh_snr = coh_gap / (coh_cs + 1e-9)
        snrs["cluster_coh"] = coh_snr
        signal_stats["cluster_coh"] = {
            "clean_mean": coh_cm, "clean_std": coh_cs,
            "poison_mean": coh_pm, "poison_std": coh_ps,
            "snr": float(coh_snr), "gap": float(coh_gap)
        }
        print(f"  {'cluster_coh':<22}: clean={coh_cm:.4f}\u00b1{coh_cs:.4f}  poison={coh_pm:.4f}\u00b1{coh_ps:.4f}"
              f"  gap={coh_gap:+.4f}  SNR={coh_snr:.2f}  (doc-level K-NN)")

        # Store for inclusion in results JSON
        self.snrs = snrs
        self.signal_stats = signal_stats

        # ── Step 3: Flip negative-SNR signals, then derive SNR-proportional weights ──
        self.flipped_signals = {s for s, v in snrs.items() if v < 0}
        if self.flipped_signals:
            print(f"\n  Flipped signals (negative SNR → inverted): {sorted(self.flipped_signals)}")
        snrs_abs = {s: abs(v) for s, v in snrs.items()}
        self.L1_weights = self._snr_weights(snrs_abs, self.L1_EXCLUDE) or {}
        self.L2_weights = self._snr_weights(snrs_abs, self.L2_EXCLUDE) or {}
        self.L3_weights = self._snr_weights(snrs_abs, self.L3_EXCLUDE) or {}

        if not self.L1_weights:
            print("\n  STOP: No positive-SNR signal for L1."); return
        if not self.L2_weights:
            print("\n  STOP: No positive-SNR signal for L2."); return
        if not self.L3_weights:
            print("\n  STOP: No positive-SNR signal for L3."); return

        for label, wt in [("L1", self.L1_weights), ("L2", self.L2_weights), ("L3", self.L3_weights)]:
            active = {k: v for k, v in wt.items() if v > 0}
            print(f"\n  {label} weights (SNR-derived): {active}")

        # ── Step 4: Calibrate tau using ONLY cal_queries (benign only) ──
        print(f"\n  Loading primary encoder (BGE-large) for calibration...")
        enc_p = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)

        # Cal records: (9 text signals, doc_coh, is_poisoned).
        # cluster_coh is now doc-level (self.doc_coh[i]), not query-level —
        # eliminates retrieval contamination bias at high poison densities.
        cal_records = []
        self.cal_doc_ids = set()
        print(f"  Scoring {len(self.cal_queries)} cal-benign queries...")
        for q in self.cal_queries:
            qe_p = enc_p.encode([q["q"]], convert_to_numpy=True, normalize_embeddings=True)
            qe32 = np.ascontiguousarray(qe_p, dtype=np.float32)
            _, ir_fetch = self.idx.search(qe32, K_FETCH)
            ri_fetch = [int(i) for i in ir_fetch[0] if i >= 0]
            n_fetch = len(ri_fetch)
            if n_fetch < 2: continue
            sims_fetch = [float(np.dot(self.pe[j], qe32[0])) for j in ri_fetch]
            top_k_pos = sorted(range(n_fetch), key=lambda x: sims_fetch[x], reverse=True)[:K]
            for fp in top_k_pos:
                i = ri_fetch[fp]
                doc_id = self.corpus[i]["id"]
                self.cal_doc_ids.add(doc_id)
                row = _features_row(i)
                cal_records.append(row + [float(self.doc_coh[i]), self.corpus[i]["is_poisoned"]])

        del enc_p; flush()

        cal_recs = np.array([r[:10] for r in cal_records], dtype=np.float32)
        cal_labels = np.array([r[10] for r in cal_records], dtype=bool)
        cal_clean = ~cal_labels  # benign queries only, so clean = not poisoned

        print(f"  Cal records: {len(cal_records)} ({cal_clean.sum()} clean, {cal_labels.sum()} poison)")
        print(f"  Cal doc IDs tracked: {len(self.cal_doc_ids)}")

        if cal_clean.sum() < 10:
            print("  STOP: insufficient clean calibration records."); return

        # Compute scores for each layer from raw signals (flip inverted signals)
        def _compute_scores(recs_arr, weights):
            scores = np.zeros(len(recs_arr), dtype=np.float32)
            for sig_name, sig_col in signal_defs:
                w = weights.get(sig_name, 0.0)
                if w > 0:
                    col = recs_arr[:, sig_col]
                    if sig_name in self.flipped_signals:
                        col = 1.0 - col
                    scores += w * col
            return scores

        scores_L1 = _compute_scores(cal_recs, self.L1_weights)
        scores_L2 = _compute_scores(cal_recs, self.L2_weights)
        scores_L3 = _compute_scores(cal_recs, self.L3_weights)

        def calibrate(scores_clean, fpr_target):
            lo, hi = float(scores_clean.min()), float(scores_clean.max())
            for _ in range(50):
                mid = (lo + hi) / 2
                if float(np.mean(scores_clean > mid)) > fpr_target: lo = mid
                else: hi = mid
            tau = (lo + hi) / 2
            cal_fpr = float(np.mean(scores_clean > tau))
            return tau, cal_fpr

        bc_L1 = scores_L1[cal_clean]; bc_L2 = scores_L2[cal_clean]; bc_L3 = scores_L3[cal_clean]

        # Universal calibration target — no oracle density input.
        # The switch to doc-level cluster_coh eliminates the high-density FPR inflation
        # that previously required a density-aware multiplier (RF-2/RF-3 fix).
        calibrate_target = FPR_TARGET

        self.tau_L1, cal_fpr_L1 = calibrate(bc_L1, calibrate_target)
        self.tau_L2, cal_fpr_L2 = calibrate(bc_L2, calibrate_target)
        self.tau_L3, cal_fpr_L3 = calibrate(bc_L3, calibrate_target)

        # Informational cal_TPR from corpus-sampled poison features (flip inverted signals)
        def _cal_tpr(poison_feats_arr, weights, tau):
            scores = np.zeros(len(poison_feats_arr), dtype=np.float32)
            for sig_name, sig_col in signal_defs:
                w = weights.get(sig_name, 0.0)
                if w > 0:
                    col = poison_feats_arr[:, sig_col]
                    if sig_name in self.flipped_signals:
                        col = 1.0 - col
                    scores += w * col
            return float(np.mean(scores > tau)) if len(scores) > 0 else 0.0

        # Pad poison_feats with actual per-doc cluster_coh values (col 9)
        poison_coh_col = self.doc_coh[np.array(poison_indices, dtype=np.int64)].reshape(-1, 1)
        poison_feats_10 = np.hstack([poison_feats, poison_coh_col])

        cal_tpr_L1 = _cal_tpr(poison_feats_10, self.L1_weights, self.tau_L1)
        cal_tpr_L2 = _cal_tpr(poison_feats_10, self.L2_weights, self.tau_L2)
        cal_tpr_L3 = _cal_tpr(poison_feats_10, self.L3_weights, self.tau_L3)

        print(f"\n  LAYER 1 (Naive Adversary — all signals):")
        print(f"    tau_L1={self.tau_L1:.4f}  cal_FPR={cal_fpr_L1*100:.1f}%  corpus_TPR={cal_tpr_L1*100:.1f}%")
        print(f"\n  LAYER 2 (Standard Adaptive — no kw_density):")
        print(f"    tau_L2={self.tau_L2:.4f}  cal_FPR={cal_fpr_L2*100:.1f}%  corpus_TPR={cal_tpr_L2*100:.1f}%")
        print(f"\n  LAYER 3 (Compound Adaptive — no kw_density, no avg_sent_len):")
        print(f"    tau_L3={self.tau_L3:.4f}  cal_FPR={cal_fpr_L3*100:.1f}%  corpus_TPR={cal_tpr_L3*100:.1f}%")

        json.dump({"version": "v6.2.1",
                   "tau_L1": self.tau_L1, "tau_L2": self.tau_L2, "tau_L3": self.tau_L3,
                   "L1_weights": self.L1_weights, "L2_weights": self.L2_weights,
                   "L3_weights": self.L3_weights,
                   "flipped_signals": sorted(self.flipped_signals),
                   "norm_config": self.norm_config,
                   "cal_doc_ids": sorted(self.cal_doc_ids),
                   "cal_fpr_L1": cal_fpr_L1, "cal_tpr_L1": cal_tpr_L1,
                   "cal_fpr_L2": cal_fpr_L2, "cal_tpr_L2": cal_tpr_L2,
                   "cal_fpr_L3": cal_fpr_L3, "cal_tpr_L3": cal_tpr_L3,
                   "target_fpr": FPR_TARGET,
                   "cal_seed": self.cal_seed,
                   "snrs": snrs,
                   "signal_stats": signal_stats},
                  open(ck, "w", encoding="utf-8"))
        self.phase3_ok = True
        print(f"Phase 3: {time.perf_counter() - t0:.1f}s\n")

    # ---- PHASE 4: Three-Layer Evaluation ----
    def phase4(self):
        t0 = time.perf_counter()
        print(f"--- PHASE 4: v6.2 Independent Evaluation (K={K}, K_FETCH={K_FETCH}, seed={self.cal_seed}) ---")
        print(f"  L1: tau={self.tau_L1:.4f}  L2: tau={self.tau_L2:.4f}  L3: tau={self.tau_L3:.4f}")
        print(f"  Eval queries: {len(self.eval_queries)} ({sum(1 for q in self.eval_queries if not q['adv'])} benign + "
              f"{sum(1 for q in self.eval_queries if q['adv'])} adv)")
        print(f"  Cal doc IDs to decontaminate: {len(self.cal_doc_ids)}")

        self._compute_centroid()
        enc_p = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
        R_L1 = Results(); R_L2 = Results(); R_L3 = Results()

        for wq in self.eval_queries[:3]:
            qe = enc_p.encode([wq["q"]], convert_to_numpy=True, normalize_embeddings=True)
            self.idx.search(qe.astype(np.float32), K_FETCH)
        print("  Warmup done. Evaluating...")

        n_eval = len(self.eval_queries)
        decon_skip = 0  # count of docs skipped due to decontamination
        for qi, qr in enumerate(self.eval_queries):
            ts = time.perf_counter()
            qe_p = enc_p.encode([qr["q"]], convert_to_numpy=True, normalize_embeddings=True)
            qe32 = np.ascontiguousarray(qe_p, dtype=np.float32)

            _, ir_fetch = self.idx.search(qe32, K_FETCH)
            ri_fetch = [int(i) for i in ir_fetch[0] if i >= 0]
            n_fetch = len(ri_fetch)

            sims_fetch = [float(np.dot(self.pe[j], qe32[0])) for j in ri_fetch]
            top_k_indices = sorted(range(n_fetch), key=lambda x: sims_fetch[x], reverse=True)[:K]
            ri = [ri_fetch[x] for x in top_k_indices]
            rd = [self.corpus[i] for i in ri]

            # cluster_coh is now doc-level (precomputed self.doc_coh[idx]).
            # Per-query group coherence is NOT used — avoids retrieval contamination.

            flagged_L1 = [False] * len(rd)
            flagged_L2 = [False] * len(rd)
            flagged_L3 = [False] * len(rd)
            contaminated = [False] * len(rd)
            for j, (doc, idx) in enumerate(zip(rd, ri)):
                # Document-level decontamination: skip docs scored in Phase 3
                if doc["id"] in self.cal_doc_ids and not qr["adv"]:
                    contaminated[j] = True
                    decon_skip += 1
                    continue
                if sha256(doc["text"]) != self.hashes.get(doc["id"], ""):
                    flagged_L1[j] = True; R_L1.hash_c += 1
                    flagged_L2[j] = True; R_L2.hash_c += 1
                    flagged_L3[j] = True; R_L3.hash_c += 1
                else:
                    de_p = self.pe[idx]
                    drift = 1.0 - float(np.dot(de_p, self.corpus_centroid))
                    sent_unif, ttr_signal, repeat_rate, kw_density, doc_len, avg_sl, punct_s, cttr_s = \
                        text_features(doc["text"], self.norm_config)
                    doc_coh_val = float(self.doc_coh[idx])  # doc-level K-NN cohesion
                    score_L1 = self._score(drift, sent_unif, ttr_signal, repeat_rate, kw_density,
                                           doc_len, avg_sl, punct_s, cttr_s, doc_coh_val, self.L1_weights)
                    score_L2 = self._score(drift, sent_unif, ttr_signal, repeat_rate, kw_density,
                                           doc_len, avg_sl, punct_s, cttr_s, doc_coh_val, self.L2_weights)
                    score_L3 = self._score(drift, sent_unif, ttr_signal, repeat_rate, kw_density,
                                           doc_len, avg_sl, punct_s, cttr_s, doc_coh_val, self.L3_weights)
                    if score_L1 > self.tau_L1: flagged_L1[j] = True; R_L1.ml_c += 1
                    if score_L2 > self.tau_L2: flagged_L2[j] = True; R_L2.ml_c += 1
                    if score_L3 > self.tau_L3: flagged_L3[j] = True; R_L3.ml_c += 1

            lat_ms = (time.perf_counter() - ts) * 1000
            R_L1.latencies.append(lat_ms); R_L2.latencies.append(lat_ms); R_L3.latencies.append(lat_ms)

            qfp_L1 = False; qfp_L2 = False; qfp_L3 = False
            for j, doc in enumerate(rd):
                if contaminated[j]: continue  # decontaminated — skip for counting
                poi = doc["is_poisoned"]
                if flagged_L1[j]:
                    if not poi and not qr["adv"]: R_L1.fp_docs += 1; qfp_L1 = True
                else:
                    if poi and qr["adv"]: R_L1.atk_succ += 1
                if not qr["adv"] and not poi: R_L1.clean_seen += 1
                if flagged_L2[j]:
                    if not poi and not qr["adv"]: R_L2.fp_docs += 1; qfp_L2 = True
                else:
                    if poi and qr["adv"]: R_L2.atk_succ += 1
                if not qr["adv"] and not poi: R_L2.clean_seen += 1
                if flagged_L3[j]:
                    if not poi and not qr["adv"]: R_L3.fp_docs += 1; qfp_L3 = True
                else:
                    if poi and qr["adv"]: R_L3.atk_succ += 1
                if not qr["adv"] and not poi: R_L3.clean_seen += 1

            if qr["adv"]:
                att = sum(1 for j, d in enumerate(rd) if d["is_poisoned"] and not contaminated[j])
                R_L1.atk_att += att; R_L2.atk_att += att; R_L3.atk_att += att
            else:
                R_L1.benign_total += 1; R_L2.benign_total += 1; R_L3.benign_total += 1
                if qfp_L1: R_L1.benign_fp_q += 1
                if qfp_L2: R_L2.benign_fp_q += 1
                if qfp_L3: R_L3.benign_fp_q += 1

            c = qi + 1
            if c % 10 == 0:
                tt = time.perf_counter() - t0
                eta = int((n_eval - c) / (c / tt)) if c > 0 else 0
                print(f"  Q {c:>4}/{n_eval} | {lat_ms:.0f}ms | "
                      f"L1 ASR={R_L1.asr():.1f}% FPR={R_L1.fpr():.1f}% | "
                      f"L2 ASR={R_L2.asr():.1f}% FPR={R_L2.fpr():.1f}% | "
                      f"L3 ASR={R_L3.asr():.1f}% FPR={R_L3.fpr():.1f}% | ETA={eta//60}m{eta%60}s")

        del enc_p; flush()
        print(f"  Decontamination: {decon_skip} doc-encounters skipped (benign-query overlap with cal set)")
        print(f"  Effective sample: {R_L1.clean_seen} clean docs, {R_L1.atk_att} attack attempts")
        print(f"Phase 4: {time.perf_counter() - t0:.1f}s\n")
        return R_L1, R_L2, R_L3

    def report(self, R_L1, R_L2, R_L3):
        W = 64
        pct = f"{self.poison_ratio*100:.1f}%" if self.poison_ratio is not None else f"~{self.P/self.N*100:.1f}%"
        print("=" * W)
        print(f"  SEVA v6.2 RESULTS -- {self.N // 1000}k ({pct} poison) seed={self.cal_seed}".center(W))
        print("=" * W)
        print(f"  {'Corpus':<28}: {self.N:,} ({self.P:,} poisoned)")
        print(f"  {'Poison Ratio':<28}: {pct}")
        print(f"  {'FPR_TARGET (universal)':<28}: {FPR_TARGET*100:.1f}%")
        if self.norm_config:
            print(f"  {'Norm (doc_len/sent/punct)':<28}: {self.norm_config['doc_length']:.0f} / "
                  f"{self.norm_config['sent_length']:.1f} / {self.norm_config['punct_density']:.4f}")
        print(f"\n  LAYER 1 (Naive Adversary — all signals, SNR-weighted):")
        print(f"    {'tau_L1':<24}: {self.tau_L1:.4f}")
        print(f"    {'ASR':<24}: {R_L1.asr():.2f}%")
        print(f"    {'DocFPR':<24}: {R_L1.fpr():.2f}%")
        print(f"    {'QueryFPR':<24}: {R_L1.qfpr():.2f}%")
        print(f"    {'Catches (hash/ml)':<24}: {R_L1.hash_c}/{R_L1.ml_c}")
        print(f"    {'Latency (mean/p95)':<24}: {R_L1.lat():.1f}/{R_L1.p95():.1f} ms")
        print(f"\n  LAYER 2 (Standard Adaptive — no kw_density, SNR-weighted):")
        print(f"    {'tau_L2':<24}: {self.tau_L2:.4f}")
        print(f"    {'ASR':<24}: {R_L2.asr():.2f}%")
        print(f"    {'DocFPR':<24}: {R_L2.fpr():.2f}%")
        print(f"    {'QueryFPR':<24}: {R_L2.qfpr():.2f}%")
        print(f"    {'Catches (hash/ml)':<24}: {R_L2.hash_c}/{R_L2.ml_c}")
        print(f"    {'Latency (mean/p95)':<24}: {R_L2.lat():.1f}/{R_L2.p95():.1f} ms")
        print(f"\n  LAYER 3 (Compound Adaptive — no kw/avg_sent_len, SNR-weighted):")
        print(f"    {'tau_L3':<24}: {self.tau_L3:.4f}")
        print(f"    {'ASR':<24}: {R_L3.asr():.2f}%")
        print(f"    {'DocFPR':<24}: {R_L3.fpr():.2f}%")
        print(f"    {'QueryFPR':<24}: {R_L3.qfpr():.2f}%")
        print(f"    {'Catches (hash/ml)':<24}: {R_L3.hash_c}/{R_L3.ml_c}")
        print(f"    {'Latency (mean/p95)':<24}: {R_L3.lat():.1f}/{R_L3.p95():.1f} ms")
        print("=" * W)
        def _layer_out(R, weights, tau):
            """Build per-layer result dict with raw counts for post-hoc CI calculation."""
            tp = R.atk_att - R.atk_succ   # poison docs correctly flagged (in adv queries)
            fn = R.atk_succ                # poison docs missed — attack successes
            fp = R.fp_docs                 # clean docs falsely flagged (in benign queries)
            tn = R.clean_seen - R.fp_docs  # clean docs correctly passed
            return {
                "asr": R.asr(), "doc_fpr": R.fpr(), "query_fpr": R.qfpr(),
                "tau": tau,
                "weights": weights,
                # Raw confusion matrix counters (for post-hoc CI)
                "counts": {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
                           "poison_encounters": R.atk_att, "clean_seen": R.clean_seen},
                "catches": {"hash": R.hash_c, "ml": R.ml_c},
                "attacks": {"successes": R.atk_succ, "attempts": R.atk_att},
                "fp_docs": R.fp_docs, "clean_seen": R.clean_seen,
                "latency": {"mean": R.lat(), "p95": R.p95()},
            }

        out = {
            "version": "v6.2.1",
            "corpus": self.N, "poisoned": self.P,
            "poison_ratio": self.poison_ratio,
            "cal_seed": self.cal_seed,
            "benign_q": BENIGN_Q,
            "target_fpr": FPR_TARGET,
            "norm_config": self.norm_config,
            # Per-signal metadata (SNR, weights, tau per layer)
            "snrs": self.snrs,
            "signal_stats": self.signal_stats,
            # Layer-level calibration metadata
            "L1_weights": self.L1_weights, "tau_L1": self.tau_L1,
            "L2_weights": self.L2_weights, "tau_L2": self.tau_L2,
            "L3_weights": self.L3_weights, "tau_L3": self.tau_L3,
            # Layer evaluation results
            "L1": _layer_out(R_L1, self.L1_weights, self.tau_L1),
            "L2": _layer_out(R_L2, self.L2_weights, self.tau_L2),
            "L3": _layer_out(R_L3, self.L3_weights, self.tau_L3),
            "gpu": _gpu,
        }
        json.dump(out, open(self.rf, "w", encoding="utf-8"), indent=2)
        print(f"  Saved: {self.rf}")
        return out


def run(N, poison_ratio=None):
    pct = f"{poison_ratio*100:.1f}%" if poison_ratio is not None else "default"
    print(f"\n{'#' * 60}\n  {N // 1000}k EVALUATION ({pct} poison)\n{'#' * 60}")
    b = SEVABench(N, poison_ratio=poison_ratio)
    b.phase1()
    b.phase2()
    b._compute_doc_coh()   # must follow phase2 (needs self.idx + self.pe)
    b.phase3()
    if not b.phase3_ok:
        print(f"  Phase 3 stop triggered — skipping Phase 4 for {pct} tier.")
        del b; flush()
        return None
    R_L1, R_L2, R_L3 = b.phase4()
    out = b.report(R_L1, R_L2, R_L3)
    del b; flush()
    return out


def compare(results):
    W = 70
    print("\n" + "=" * W)
    print("  CROSS-SCALE STABILITY REPORT".center(W))
    print("=" * W)
    print(f"  {'Scale':<8} {'TAU':>8} {'a/b/g':>12} {'ASR%':>8} {'FPR%':>8} {'Lat':>8}")
    print("-" * W)
    for r in results:
        w = r["weights"]
        print(f"  {r['corpus']//1000}k{'':<4} {r['tau']:>8.4f} "
              f"{w['alpha']:.1f}/{w['beta']:.1f}/{w['gamma']:.1f}{'':<3} "
              f"{r['asr']:>8.2f} {r['doc_fpr']:>8.2f} {r['latency']['mean']:>8.1f}")
    asrs = [r["asr"] for r in results]
    fprs = [r["doc_fpr"] for r in results]
    ad = max(asrs) - min(asrs); fd = max(fprs) - min(fprs)
    print("-" * W)
    print(f"  ASR range: {min(asrs):.1f}% - {max(asrs):.1f}% (delta={ad:.1f}pp)")
    print(f"  FPR range: {min(fprs):.1f}% - {max(fprs):.1f}% (delta={fd:.1f}pp)")
    stable = ad < 8 and fd < 5 and all(r["asr"] < 30 for r in results)
    print(f"  STATUS: {'STABLE - 50k READY' if stable else 'NEEDS ITERATION'}")
    print("=" * W)
    json.dump({"results": results, "asr_delta": ad, "fpr_delta": fd, "stable": stable},
              open("cross_scale_v5d.json", "w", encoding="utf-8"), indent=2)


def run_multitier(N):
    poison_ratios = [0.01, 0.05, 0.10]
    tier_results = []
    tier_runtimes = []
    total_t0 = time.perf_counter()

    W = 70
    print("\n" + "=" * W)
    print("  SEVA v6.2.1 MULTI-TIER POISONING EVALUATION".center(W))
    print("=" * W)
    print(f"  {'Corpus':<20}: {N:,} documents")
    print(f"  {'GPU':<20}: {_gpu}")
    print(f"  {'Tiers':<20}: {len(poison_ratios)}")
    print(f"  {'Ratios':<20}: {', '.join(f'{r*100:.1f}%' for r in poison_ratios)}")
    print(f"  {'Started':<20}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * W)

    for ti, pr in enumerate(poison_ratios):
        tier_t0 = time.perf_counter()
        print(f"\n{'*' * W}")
        print(f"  TIER {ti+1}/{len(poison_ratios)}: {pr*100:.1f}% POISONING".center(W))
        print(f"  ({int(N * pr):,} poisoned documents out of {N:,})".center(W))
        print(f"{'*' * W}")

        out = run(N, poison_ratio=pr)
        runtime = time.perf_counter() - tier_t0
        tier_runtimes.append(runtime)
        if out is None:
            print(f"\n  >> Tier {ti+1} STOPPED (Phase 3 gate) in {runtime/60:.1f} min")
            tier_results.append({"poison_ratio": pr, "poisoned": int(N * pr),
                                  "skipped": True, "tier_runtime_s": runtime})
        else:
            out["tier_runtime_s"] = runtime
            tier_results.append(out)
            print(f"\n  >> Tier {ti+1} completed in {runtime/60:.1f} min")

        # Running summary (L1 values; skipped tiers show STOP)
        print(f"  >> Running Summary:")
        print(f"  {'Tier':<6} {'Poison%':<10} {'Docs':<8} {'L1 ASR%':<10} {'L1 FPR%':<10} {'L2 ASR%':<10} {'Runtime':<10}")
        print(f"  {'-'*64}")
        for j, tr in enumerate(tier_results):
            if tr.get("skipped"):
                print(f"  {j+1:<6} {tr['poison_ratio']*100:<10.1f} {tr['poisoned']:<8} "
                      f"  STOPPED{'':>3} {'':>10} {'':>10} {tier_runtimes[j]/60:<10.1f}m")
            else:
                print(f"  {j+1:<6} {tr['poison_ratio']*100:<10.1f} {tr['poisoned']:<8} "
                      f"{tr['L1']['asr']:<10.2f} {tr['L1']['doc_fpr']:<10.2f} "
                      f"{tr['L2']['asr']:<10.2f} {tier_runtimes[j]/60:<10.1f}m")

    total_runtime = time.perf_counter() - total_t0

    # Per-tier pass/fail targets (IEEE TIFS targets, diverse corpus)
    TIER_TARGETS = {
        0.001: {"asr": 5.0,  "fpr": 1.0},
        0.01:  {"asr": 8.0,  "fpr": 2.0},
        0.05:  {"asr": 20.0, "fpr": 5.0},
        0.10:  {"asr": 35.0, "fpr": 8.0},
    }

    print(f"\n\n{'=' * W}")
    print("  SEVA v6.2.1 MULTI-TIER CONSOLIDATED RESULTS".center(W))
    print("=" * W)
    print(f"  {'Poison%':<10} {'P.Docs':<8} | {'L1 ASR':<8} {'L1 FPR':<8} {'L1 Lat':<8} | {'L2 ASR':<8} {'L2 FPR':<8} {'L2 Lat':<8} | {'Pass?'}")
    print(f"  {'-'*80}")
    all_pass = True
    for j, tr in enumerate(tier_results):
        pr = tr["poison_ratio"]
        tgt = TIER_TARGETS.get(pr, {"asr": 50.0, "fpr": 10.0})
        if tr.get("skipped"):
            print(f"  {pr*100:<10.1f} {tr['poisoned']:<8} | STOPPED (Phase 3 gate)")
            all_pass = False
        else:
            l1 = tr["L1"]; l2 = tr["L2"]
            asr_pass = l1["asr"] <= tgt["asr"]
            fpr_pass = l1["doc_fpr"] <= tgt["fpr"]
            tier_pass = asr_pass and fpr_pass
            if not tier_pass: all_pass = False
            status = "PASS" if tier_pass else "FAIL"
            print(f"  {pr*100:<10.1f} {tr['poisoned']:<8} | "
                  f"{l1['asr']:<8.2f} {l1['doc_fpr']:<8.2f} {l1['latency']['mean']:<8.1f} | "
                  f"{l2['asr']:<8.2f} {l2['doc_fpr']:<8.2f} {l2['latency']['mean']:<8.1f} | {status}")
    print(f"  {'-'*80}")
    print(f"  Total runtime: {total_runtime/60:.1f} min")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME TIERS FAIL OR STOPPED'}")
    print("=" * W)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "gpu": _gpu,
        "corpus_size": N,
        "total_runtime_s": total_runtime,
        "total_runtime_min": total_runtime / 60,
        "tiers": tier_results,
        "tier_runtimes_s": tier_runtimes,
        "all_pass": all_pass,
    }
    sf = "seva_multitier_summary.json"
    json.dump(summary, open(sf, "w", encoding="utf-8"), indent=2)
    print(f"\n  Saved summary: {sf}")
    for tr in tier_results:
        if not tr.get("skipped"):
            ptag = poison_tag(tr["poison_ratio"])
            print(f"  Saved tier results: seva_v6_2_results_{N//1000}k_{ptag}_s{_args.cal_seed:03d}.json")

    return summary


if __name__ == "__main__":
    t0 = time.perf_counter()
    if _args.multitier:
        run_multitier(_args.mtcorpus)
    else:
        results = [run(n, poison_ratio=_args.poison_ratio) for n in _args.corpus]
        if len(results) > 1: compare(results)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")
