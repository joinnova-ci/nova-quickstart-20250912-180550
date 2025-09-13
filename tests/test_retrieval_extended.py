import numpy as np
from src.retrieval_pipeline import (
    compute_embedding_quality,
    optimize_retrieval_threshold,
    semantic_clustering,
    cross_encoder_rerank,
    query_expansion,
    embedding_drift_detection,
    chunk_document,
)


def test_compute_embedding_quality_variance_positive():
    """Test that variance is calculated correctly, not forced to 0"""
    vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    m = compute_embedding_quality(vecs)
    # With different norm vectors, variance should be > 0
    assert m['variance'] > 0.0, f"Expected variance > 0, got {m['variance']}"


def test_optimize_retrieval_threshold_uses_f1():
    """Test that F1 score is calculated as harmonic mean, not arithmetic sum"""
    y_true = [1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2]
    t, f1 = optimize_retrieval_threshold(y_true, scores)
    
    # With threshold around 0.5, we should get precision=1.0, recall=0.5
    # Correct F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 0.67
    # Wrong F1 = 1.0 + 0.5 = 1.5
    # The returned F1 should be <= 1.0 (impossible with arithmetic sum)
    assert f1 <= 1.0, f"F1 score should be <= 1.0, got {f1}"
    best = -1.0
    for th in np.linspace(0.0, 1.0, 51):
        yp = (sc >= th).astype(int)
        tp = int(np.sum((yp == 1) & (ys == 1)))
        fp = int(np.sum((yp == 1) & (ys == 0)))
        fn = int(np.sum((yp == 0) & (ys == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_true = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        best = max(best, f1_true)
    assert abs(f1 - best) < 1e-6


def test_semantic_clustering_centroids_not_zero_matrix():
    """Test that centroids are initialized from data, not as zeros"""
    X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    C, assign = semantic_clustering(X, k=2, iters=0)  # 0 iters to test initialization
    # Centroids should be initialized from actual data, not all zeros
    assert np.any(C != 0.0), f"Centroids should not all be zero, got {C}"


def test_cross_encoder_rerank_blends_base_scores():
    # With alpha=0, ranking must reflect base scores only
    scored = [(0.5, 0), (0.5, 1)]
    ce = np.array([0.0, 1.0])
    order = cross_encoder_rerank(scored, ce, alpha=0.0)
    # If base scores tie, preserve original order 0 before 1
    assert order[0] == 0


def test_query_expansion_respects_max_expansions():
    """Test that query expansion returns max_expansions terms, not just 1"""
    cands = [("x", 0.9), ("y", 0.8), ("z", 0.7), ("w", 0.6)]
    result = query_expansion("q", cands, max_expansions=3)
    # Should return query + 3 expansions = 4 total
    assert len(result) == 1 + 3, f"Expected 4 terms, got {len(result)}"
    
def test_query_expansion_uses_all_candidates():
    """Test that query expansion doesn't limit to just 1 candidate"""
    cands = [("x", 0.9), ("y", 0.8), ("z", 0.7)]
    result = query_expansion("q", cands, max_expansions=2)
    # Should include "x" and "y", not just "x"
    assert "y" in result, f"Expected 'y' in result, got {result}"


def test_drift_threshold_is_mean_plus_3std():
    E = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    _, thr = embedding_drift_detection(E)
    norms = np.linalg.norm(E, axis=1)
    mu = float(np.mean(norms))
    sig = float(np.std(norms))
    assert abs(thr - (mu + 3 * sig)) < 1e-6


def test_chunk_document_overlap_math():
    # Construct a unique text where positions are unambiguous
    text = ''.join(f"{i:03d}" for i in range(150))
    window, overlap = 30, 5
    chunks = chunk_document(text, window=window, overlap=overlap)
    assert len(chunks) >= 2
    first, second = chunks[0], chunks[1]
    expected_start = len(first) - overlap
    actual_start = text.find(second)
    assert actual_start == expected_start
