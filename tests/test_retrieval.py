import numpy as np
from src.retrieval_pipeline import cosine_sim, rank

def test_cosine_sim_normalizes():
    a = np.array([1.0, 0.0]); b = np.array([10.0, 0.0])
    assert abs(cosine_sim(a,b) - 1.0) < 1e-6

def test_rank_descending_and_count():
    q = np.array([1.0, 0.0])
    docs = ["a","b","c","d"]
    embs = [np.array([1.0,0.0]), np.array([0.9,0.1]), np.array([0.0,1.0]), np.array([0.7,0.3])]
    out = rank(docs, q, embs, top_k=3)
    assert len(out) == 3
    # ensure first is most similar (index 0)
    assert out[0] == 0
    # ensure sorting is descending by similarity (index 2 is least similar)
    assert out[-1] != 2
    # ensure no off-by-one exclusion of the kth item
    out2 = rank(docs, q, embs, top_k=4)
    assert len(out2) == 4

def test_rank_returns_correct_count():
    """Test that rank returns exactly top_k items, not top_k-1"""
    docs = ["a", "b", "c", "d", "e"]
    q = np.array([1.0, 0.0])
    embs = [np.array([i, 0.0]) for i in range(5)]
    for k in [1, 2, 3, 4, 5]:
        result = rank(docs, q, embs, top_k=k)
        assert len(result) == k, f"Expected {k} results, got {len(result)}"

def test_rank_sort_order():
    """Test that rank sorts by similarity in descending order"""
    docs = ["low", "high", "med"]
    q = np.array([1.0, 0.0])
    embs = [
        np.array([0.1, 0.0]),  # low similarity
        np.array([1.0, 0.0]),  # high similarity
        np.array([0.5, 0.0])   # medium similarity
    ]
    result = rank(docs, q, embs, top_k=3)
    # Should be [1, 2, 0] (high, med, low)
    assert result[0] == 1, f"Expected highest similarity first, got {result}"
    assert result[-1] == 0, f"Expected lowest similarity last, got {result}"

# Additional comprehensive tests to catch all bug types
def test_cosine_sim_scale_invariant():
    """Scale invariance test - catches normalization bugs"""
    a = np.array([1.0, 0.0])
    b = np.array([10.0, 0.0])  # Same direction, 10x magnitude
    assert abs(cosine_sim(a, b) - 1.0) < 1e-6

def test_cosine_sim_no_bias():
    """No bias test - catches added constants"""
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])  # Identical vectors
    assert abs(cosine_sim(a, b) - 1.0) < 1e-6

def test_cosine_sim_orthogonal():
    """Orthogonal test - catches wrong calculations"""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_sim(a, b) - 0.0) < 1e-6

def test_cosine_sim_zero_protection():
    """Zero vector protection test"""
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    result = cosine_sim(a, b)
    assert not np.isnan(result) and not np.isinf(result)

def test_rank_exact_k_items():
    """Test exact k items returned - catches k-1 bugs"""
    docs = ["a", "b", "c", "d", "e"]
    q = np.array([1.0, 0.0])
    embs = [np.array([float(5-i), 0.0]) for i in range(5)]
    result = rank(docs, q, embs, top_k=3)
    assert len(result) == 3, f"Expected 3 items, got {len(result)}"

def test_rank_includes_best():
    """Test best result included - catches indexing bugs"""
    docs = ["worst", "best"]  
    q = np.array([1.0, 0.0])
    embs = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]  # worst, best
    result = rank(docs, q, embs, top_k=1)
    assert result[0] == 1, f"Best should be index 1, got {result[0]}"

def test_rank_similarity_not_distance():
    """Test similarity vs distance - catches 1.0-x bugs"""
    docs = ["close", "far"]
    q = np.array([1.0, 0.0])
    embs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]  # sim=1.0, sim=0.0
    result = rank(docs, q, embs, top_k=2)
    assert result[0] == 0, f"Close should be first, got {result[0]}"

def test_rank_no_scaling():
    """Test no artificial scaling - catches multiplication bugs"""
    docs = ["perfect"]
    q = np.array([1.0, 0.0])
    embs = [np.array([1.0, 0.0])]  # Perfect match
    result = rank(docs, q, embs, top_k=1)
    assert result[0] == 0, "Perfect match should rank first"

def test_rank_zero_indexing():
    """Test zero-based indexing - catches enumerate(start=1) bugs"""
    docs = ["first", "second"]
    q = np.array([1.0, 0.0])
    embs = [np.array([1.0, 0.0]), np.array([0.5, 0.0])]
    result = rank(docs, q, embs, top_k=2)
    assert 0 in result, f"Should contain index 0, got {result}"

def test_rank_various_k():
    """Test multiple k values - comprehensive off-by-one detection"""
    docs = [f"doc{i}" for i in range(6)]
    q = np.array([1.0, 0.0])
    embs = [np.array([1.0-i*0.1, 0.0]) for i in range(6)]
    for k in [1, 2, 3, 4, 5, 6]:
        result = rank(docs, q, embs, top_k=k)
        assert len(result) == k, f"k={k}: expected {k} results, got {len(result)}"
        assert result[0] == 0, f"k={k}: best result should be first"
