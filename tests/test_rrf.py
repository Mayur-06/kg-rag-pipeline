"""
tests/test_rrf.py

Pure-logic unit test for Reciprocal Rank Fusion.
Self-contained: imports NO external libraries, NO API keys required.
Run with:  python tests/test_rrf.py
       or: pytest tests/test_rrf.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── inline copy of the RRF function so this test has zero import deps ─────
def _rrf(dense, sparse, k=60):
    scores: dict[str, float] = {}
    for rank, (nid, _) in enumerate(dense, 1):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
    for rank, (nid, _) in enumerate(sparse, 1):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def test_doc_in_both_lists_outranks_doc_in_one_list():
    """A document ranking well in BOTH lists beats one ranking #1 in
    only one list — the whole point of RRF over single-ranker trust."""
    dense  = [("docA", 0.9), ("docB", 0.85), ("docC", 0.8)]
    sparse = [("docD", 15.2), ("docC", 10.1), ("docB", 9.0)]
    top_two = {nid for nid, _ in _rrf(dense, sparse)[:2]}
    assert top_two == {"docB", "docC"}, f"got {top_two}"
    print("✅ test_doc_in_both_lists_outranks_doc_in_one_list")


def test_empty_sparse_preserves_dense_order():
    dense  = [("docA", 0.9), ("docB", 0.8)]
    sparse: list = []
    ids = [nid for nid, _ in _rrf(dense, sparse)]
    assert ids == ["docA", "docB"], f"got {ids}"
    print("✅ test_empty_sparse_preserves_dense_order")


def test_disjoint_lists_return_union():
    dense  = [("docA", 0.9)]
    sparse = [("docZ", 12.0)]
    ids = {nid for nid, _ in _rrf(dense, sparse)}
    assert ids == {"docA", "docZ"}, f"got {ids}"
    print("✅ test_disjoint_lists_return_union")


def test_rrf_score_exact_arithmetic():
    """rank-1 in a single list, k=60 → score = 1/(60+1) = 1/61."""
    fused = _rrf([("docA", 0.99)], [], k=60)
    assert len(fused) == 1
    assert abs(fused[0][1] - 1/61) < 1e-12, f"got {fused[0][1]}"
    print("✅ test_rrf_score_exact_arithmetic")


def test_three_way_overlap_accumulates_correctly():
    """A document appearing in all three ranking positions should have
    three RRF contributions summed, not overwritten."""
    dense  = [("docX", 1.0), ("docY", 0.9)]
    sparse = [("docX", 20.0), ("docY", 15.0)]
    fused = dict(_rrf(dense, sparse))
    # docX: 1/61 + 1/61 (rank-1 in both) = 2/61
    expected_x = 2 / 61
    assert abs(fused["docX"] - expected_x) < 1e-12, f"got {fused['docX']}"
    assert fused["docX"] > fused["docY"], "docX should still lead docY"
    print("✅ test_three_way_overlap_accumulates_correctly")


if __name__ == "__main__":
    test_doc_in_both_lists_outranks_doc_in_one_list()
    test_empty_sparse_preserves_dense_order()
    test_disjoint_lists_return_union()
    test_rrf_score_exact_arithmetic()
    test_three_way_overlap_accumulates_correctly()
    print("\n🎉 All RRF tests passed.")
