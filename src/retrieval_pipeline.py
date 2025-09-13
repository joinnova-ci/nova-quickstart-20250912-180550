"""Retrieval Pipeline (extended demo)

This module intentionally contains a rich set of retrieval- and RAG-related utilities
so the quickstart can demonstrate Nova fixing multiple classes of issues.

Sections:
  - Vector math and similarity
  - Ranking and top-k selection
  - Chunking and adaptive chunking
  - Quality metrics and thresholding
  - Clustering and reranking
  - Query expansion
  - Drift detection

Note: Some simple implementations are used for demonstration purposes.
"""
import numpy as np
from typing import List, Tuple, Sequence, Dict
from dataclasses import dataclass
from collections import Counter, defaultdict
import math

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with proper L2 normalization on both vectors."""
    a_norm = a / (np.linalg.norm(a) or 1.0)
    b_norm = b / (np.linalg.norm(b) or 1.0)  # BUG_HOOK_NORM
    return float(np.dot(a_norm, b_norm))

def rank(documents: List[str], query_embedding: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    sims: List[Tuple[float, int]] = [(cosine_sim(query_embedding, e), i) for i, e in enumerate(doc_embeddings)]
    sims.sort(key=lambda x: x[0], reverse=True)  # BUG_HOOK_SORT
    return [i for _, i in sims[:top_k]]  # BUG_HOOK_SLICE

def chunk_document(text: str, window: int = 100, overlap: int = 20) -> List[str]:
    """Simple fixed-window chunking with overlap."""
    if window <= 0:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + window)
        chunks.append(text[start:end])
        start = end - overlap  # BUG_HOOK_OVERLAP
        if start <= 0:
            start = end
    return chunks

def compute_embedding_quality(vectors: Sequence[np.ndarray]) -> Dict[str, float]:
    if not vectors:
        return {"mean": 0.0, "variance": 0.0}
    vs = np.array([float(np.linalg.norm(v)) for v in vectors], dtype=float)
    return {"mean": float(np.mean(vs)), "variance": float(np.var(vs))}  # BUG_HOOK_VARIANCE

def optimize_retrieval_threshold(y_true: Sequence[int], scores: Sequence[float]) -> Tuple[float, float]:
    """Return (best_threshold, best_f1)."""
    if not scores:
        return 0.0, 0.0
    thresholds = np.linspace(0.0, 1.0, 51)
    best_f1 = -1.0
    best_t = 0.0
    y_true_np = np.array(y_true, dtype=int)
    scores_np = np.array(scores, dtype=float)
    for t in thresholds:
        y_pred = (scores_np >= t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true_np == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true_np == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true_np == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  # BUG_HOOK_F1
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)

def semantic_clustering(vectors: np.ndarray, k: int = 3, iters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Tiny k-means-like clustering for demo; returns (centroids, assignments)."""
    if vectors.size == 0 or k <= 0:
        return np.empty((0, 0)), np.empty((0,), dtype=int)
    k = min(k, max(1, vectors.shape[0]))
    centroids = vectors[:k, :].copy()  # BUG_HOOK_INIT_CENTROIDS
    for _ in range(max(1, iters)):
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        for i in range(k):
            mask = assign == i
            if np.any(mask):
                centroids[i] = np.mean(vectors[mask], axis=0)
    return centroids, assign

def cross_encoder_rerank(scored: List[Tuple[float, int]], cross_encoder_scores: np.ndarray, alpha: float = 0.5) -> List[int]:
    """Blend base scores with cross-encoder scores."""
    ce = np.asarray(cross_encoder_scores, dtype=float)
    base_scores = np.array([s for s, _ in scored], dtype=float)
    combined = alpha * ce + (1 - alpha) * base_scores  # BUG_HOOK_CE_INTERACTION
    order = np.argsort(-combined)
    return [scored[i][1] for i in order.tolist()]

def query_expansion(query: str, candidates: List[Tuple[str, float]], max_expansions: int = 3) -> List[str]:
    """Select top textual expansions by weight."""
    ranked = sorted(candidates, key=lambda x: -x[1])
    expansions = [w for w, _ in ranked[:max_expansions]]  # BUG_HOOK_EXPANSION
    return [query] + expansions

def adaptive_chunking(text: str, base_window: int = 120, max_window: int = 240) -> List[str]:
    """Toy adaptive chunking based on simple punctuation density."""
    if not text:
        return []
    density = text.count(",") + text.count(";") + text.count(":")
    window = min(max_window, base_window + density)
    return chunk_document(text, window=window, overlap=20)

def embedding_drift_detection(embeddings: np.ndarray) -> Tuple[float, float]:
    """Return (score, threshold) where score>threshold indicates drift."""
    if embeddings.size == 0:
        return 0.0, 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    mean = float(np.mean(norms))
    std = float(np.std(norms))
    threshold = mean + 3 * std  # BUG_HOOK_DRIFT_THRESH
    score = float(abs(norms[-1] - mean))
    return score, threshold


def tokenize(text: str) -> List[str]:
    return [t for t in str(text).lower().split() if t]

def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_sim(a, b)

def softmax(x: Sequence[float]) -> List[float]:
    arr = np.array(list(x), dtype=float)
    if arr.size == 0:
        return []
    m = float(np.max(arr))
    ex = np.exp(arr - m)
    s = float(np.sum(ex)) or 1.0
    return (ex / s).tolist()

def moving_average(x: Sequence[float], window: int = 5) -> List[float]:
    if window <= 0:
        return []
    arr = np.array(list(x), dtype=float)
    if arr.size == 0:
        return []
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = (cumsum[window:] - cumsum[:-window]) / float(window)
    return out.tolist()

def build_inverted_index(docs: Sequence[str]) -> Dict[str, List[int]]:
    inv: Dict[str, List[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        seen = set()
        for tok in tokenize(d):
            if tok not in seen:
                inv[tok].append(i)
                seen.add(tok)
    return dict(inv)

def compute_tf_idf(tokenized_docs: Sequence[Sequence[str]]) -> Tuple[Dict[str, int], np.ndarray]:
    vocab: Dict[str, int] = {}
    # Build vocab
    for toks in tokenized_docs:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    n_docs = len(tokenized_docs)
    if n_docs == 0:
        return vocab, np.zeros((0, 0), dtype=float)
    # Term frequencies and document frequencies
    tf = np.zeros((n_docs, len(vocab)), dtype=float)
    df = np.zeros((len(vocab),), dtype=float)
    for i, toks in enumerate(tokenized_docs):
        counts = Counter(toks)
        if not counts:
            continue
        max_tf = float(max(counts.values())) or 1.0
        for t, c in counts.items():
            j = vocab[t]
            tf[i, j] = c / max_tf
        for t in set(toks):
            df[vocab[t]] += 1
    idf = np.log((n_docs + 1) / (df + 1)) + 1.0
    tfidf = tf * idf
    return vocab, tfidf

def bm25_score(query_tokens: Sequence[str], doc_tokens: Sequence[str], avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    if avgdl <= 0:
        avgdl = 1.0
    q_counts = Counter(query_tokens)
    d_counts = Counter(doc_tokens)
    dl = float(sum(d_counts.values())) or 1.0
    score = 0.0
    for term, qf in q_counts.items():
        f = float(d_counts.get(term, 0))
        if f <= 0:
            continue
        idf = math.log(1 + (1_000_000 - f + 0.5) / (f + 0.5))
        denom = f + k1 * (1 - b + b * (dl / avgdl))
        score += idf * (f * (k1 + 1)) / (denom or 1.0)
    return float(score)

def precision_at_k(true_items: Sequence[int], pred_items: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    s_true = set(true_items)
    s_pred = pred_items[:k]
    if not s_pred:
        return 0.0
    return float(len(s_true.intersection(s_pred))) / float(len(s_pred))

def recall_at_k(true_items: Sequence[int], pred_items: Sequence[int], k: int) -> float:
    s_true = set(true_items)
    if not s_true:
        return 0.0
    s_pred = set(pred_items[:k])
    return float(len(s_true.intersection(s_pred))) / float(len(s_true))

def dcg(scores: Sequence[float], k: int) -> float:
    s = 0.0
    for i, v in enumerate(scores[:k], start=1):
        s += (2**v - 1) / math.log2(i + 1)
    return float(s)

def ndcg(true_rels: Sequence[float], pred_order: Sequence[int], k: int) -> float:
    ideal = sorted(true_rels, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg <= 0:
        return 0.0
    ranked = [true_rels[i] for i in pred_order]
    return float(dcg(ranked, k) / ideal_dcg)

@dataclass
class RetrievalConfig:
    top_k: int = 5
    use_cross_encoder: bool = False
    alpha: float = 0.5

class RetrievalPipeline:
    def __init__(self, docs: Sequence[str], config: RetrievalConfig | None = None) -> None:
        self.docs = list(docs)
        self.cfg = config or RetrievalConfig()
        self.inv = build_inverted_index(self.docs)

    def search(self, query: str, doc_embs: Sequence[np.ndarray], query_emb: np.ndarray) -> List[int]:
        order = rank(self.docs, query_emb, list(doc_embs), top_k=self.cfg.top_k)
        return order

    def evaluate(self, queries: Sequence[str], gold: Sequence[Sequence[int]], doc_embs: Sequence[np.ndarray], query_embs: Sequence[np.ndarray]) -> Dict[str, float]:
        precs, recs = [], []
        for q, g, qe in zip(queries, gold, query_embs):
            r = self.search(q, doc_embs, qe)
            precs.append(precision_at_k(g, r, self.cfg.top_k))
            recs.append(recall_at_k(g, r, self.cfg.top_k))
        return {"precision@k": float(np.mean(precs) if precs else 0.0), "recall@k": float(np.mean(recs) if recs else 0.0)}

def _noop(*args, **kwargs):
    return None

def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_1(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 1
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_2(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 2
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_3(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 3
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_4(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 4
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_5(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 5
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_6(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 6
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_7(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 7
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_8(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 8
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_9(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 9
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_10(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 10
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_11(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 11
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_12(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 12
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_13(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 13
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_14(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 14
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_15(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 15
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_16(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 16
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_17(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 17
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_18(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 18
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_19(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 19
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_20(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 20
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_21(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 21
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_22(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 22
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_23(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 23
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_24(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 24
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_25(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 25
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_26(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 26
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_27(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 27
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_28(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 28
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_29(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 29
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_30(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 30
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_31(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 31
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_32(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 32
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_33(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 33
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_34(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 34
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_35(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 35
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_36(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 36
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_37(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 37
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_38(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 38
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_39(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 39
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_40(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 40
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_41(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 41
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_42(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 42
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_43(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 43
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_44(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 44
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_45(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 45
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_46(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 46
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_47(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 47
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_48(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 48
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_49(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 49
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_50(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 50
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_51(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 51
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_52(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 52
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_53(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 53
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_54(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 54
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_55(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 55
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_56(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 56
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_57(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 57
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_58(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 58
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_59(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 59
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_60(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 60
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_61(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 61
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_62(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 62
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_63(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 63
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_64(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 64
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_65(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 65
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_66(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 66
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_67(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 67
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_68(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 68
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_69(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 69
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_70(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 70
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_71(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 71
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_72(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 72
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_73(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 73
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_74(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 74
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_75(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 75
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_76(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 76
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_77(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 77
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_78(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 78
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_79(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 79
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_80(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 80
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_81(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 81
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_82(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 82
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_83(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 83
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_84(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 84
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_85(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 85
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_86(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 86
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_87(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 87
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_88(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 88
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_89(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 89
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_90(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 90
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_91(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 91
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_92(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 92
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_93(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 93
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_94(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 94
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_95(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 95
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_96(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 96
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_97(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 97
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_98(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 98
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_99(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 99
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_100(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 100
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_101(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 101
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_102(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 102
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_103(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 103
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_104(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 104
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_105(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 105
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_106(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 106
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_107(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 107
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_108(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 108
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_109(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 109
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_110(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 110
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_111(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 111
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_112(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 112
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_113(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 113
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_114(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 114
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_115(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 115
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_116(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 116
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_117(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 117
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_118(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 118
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_119(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 119
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_120(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 120
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_121(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 121
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_122(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 122
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_123(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 123
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_124(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 124
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_125(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 125
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_126(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 126
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_127(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 127
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_128(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 128
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_129(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 129
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_130(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 130
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_131(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 131
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_132(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 132
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_133(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 133
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_134(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 134
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_135(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 135
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_136(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 136
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_137(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 137
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_138(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 138
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_139(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 139
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_140(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 140
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_141(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 141
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_142(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 142
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_143(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 143
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

def _safe_topk_144(values: Sequence[float], k: int) -> List[int]:
    idx = np.argsort(-np.array(values, dtype=float))
    k = max(0, int(k))
    return idx[:k].tolist()

# helper repetition 144
def _stat_summary(x: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(x), dtype=float) if x else np.array([], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
