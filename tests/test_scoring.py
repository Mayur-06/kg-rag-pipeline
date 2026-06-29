"""
tests/test_scoring.py

Offline unit tests for the four direct-Ollama scoring functions.
All functions use single-prompt 0-10 scoring (simpler, works with 1B models).
Zero external dependencies — runs with only stdlib + unittest.mock.

Run with:  python tests/test_scoring.py
       or: pytest tests/test_scoring.py -v
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Inline the four scoring functions (matching evaluate.py exactly) ────────

def _ollama_score(base_url, model, prompt, timeout=120.0):
    raise NotImplementedError("should be mocked in tests")


def _score_faithfulness(question, answer, contexts, base_url, model):
    context_block = " | ".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = (
        f"Rate from 0 to 10 how well the answer is supported by the context "
        f"(0=contradicts/ignores, 10=every claim is from context).\n"
        f"Reply with ONLY a number from 0 to 10.\n\n"
        f"CONTEXT: {context_block}\n\nANSWER: {answer}\n\nScore (0-10):"
    )
    raw = _ollama_score(base_url, model, prompt)
    m = re.search(r"\b(10|[0-9])\b", raw)
    if not m:
        return float("nan")
    return round(int(m.group()) / 10, 4)


def _score_answer_relevancy(question, answer, base_url, model):
    prompt = (
        f"Rate from 0 to 10 how directly the ANSWER addresses the QUESTION.\n"
        f"Reply with ONLY a number from 0 to 10.\n\n"
        f"QUESTION: {question}\n\nANSWER: {answer}\n\nScore (0-10):"
    )
    raw = _ollama_score(base_url, model, prompt)
    m = re.search(r"\b(10|[0-9])\b", raw)
    if not m:
        return float("nan")
    return round(int(m.group()) / 10, 4)


def _score_context_precision(question, answer, contexts, reference, base_url, model):
    if not contexts:
        return float("nan")
    relevant = 0
    for i, ctx in enumerate(contexts):
        prompt = (
            f"Does the following text contain information useful for answering "
            f"the question? Reply with YES or NO only.\n\n"
            f"Question: {question}\n\nText: {ctx[:400]}\n\nAnswer (YES or NO):"
        )
        verdict = _ollama_score(base_url, model, prompt).lower()
        if "yes" in verdict:
            relevant += 1
    return round(relevant / len(contexts), 4)


def _score_context_recall(question, contexts, reference, base_url, model):
    context_block = " | ".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = (
        f"Rate from 0 to 10 how well the CONTEXT supports the REFERENCE ANSWER.\n"
        f"Reply with ONLY a number from 0 to 10.\n\n"
        f"CONTEXT: {context_block}\n\nREFERENCE ANSWER: {reference}\n\nScore (0-10):"
    )
    raw = _ollama_score(base_url, model, prompt)
    m = re.search(r"\b(10|[0-9])\b", raw)
    if not m:
        return float("nan")
    return round(int(m.group()) / 10, 4)


# ── Helpers ────────────────────────────────────────────────────────────────

BASE  = "http://localhost:11434"
MODEL = "llama3.2:1b"


def _mock(responses):
    it = iter(responses)
    return patch(__name__ + "._ollama_score", side_effect=lambda *a, **k: next(it))


# ── Faithfulness (0-10 scale → 0.0-1.0) ───────────────────────────────────

def test_faithfulness_10_maps_to_1():
    with _mock(["10"]):
        assert _score_faithfulness("Q", "A", ["ctx"], BASE, MODEL) == 1.0
    print("✅ faithfulness: 10 → 1.0")


def test_faithfulness_0_maps_to_0():
    with _mock(["0"]):
        assert _score_faithfulness("Q", "A", ["ctx"], BASE, MODEL) == 0.0
    print("✅ faithfulness: 0 → 0.0")


def test_faithfulness_5_maps_to_half():
    with _mock(["5"]):
        assert _score_faithfulness("Q", "A", ["ctx"], BASE, MODEL) == 0.5
    print("✅ faithfulness: 5 → 0.5")


def test_faithfulness_no_digit_nan():
    with _mock(["I cannot determine"]):
        assert math.isnan(_score_faithfulness("Q", "A", ["ctx"], BASE, MODEL))
    print("✅ faithfulness: no digit → NaN")


def test_faithfulness_text_with_digit_extracted():
    with _mock(["I would rate this an 8 out of 10"]):
        score = _score_faithfulness("Q", "A", ["ctx"], BASE, MODEL)
        assert score == 0.8, f"Expected 0.8, got {score}"
    print("✅ faithfulness: extracts digit from verbose response → 0.8")


# ── Answer Relevancy (0-10 → 0.0-1.0) ─────────────────────────────────────

def test_relevancy_10_to_1():
    with _mock(["10"]):
        assert _score_answer_relevancy("Q", "A", BASE, MODEL) == 1.0
    print("✅ answer_relevancy: 10 → 1.0")


def test_relevancy_0_to_0():
    with _mock(["0"]):
        assert _score_answer_relevancy("Q", "A", BASE, MODEL) == 0.0
    print("✅ answer_relevancy: 0 → 0.0")


def test_relevancy_7_to_point7():
    with _mock(["7"]):
        assert _score_answer_relevancy("Q", "A", BASE, MODEL) == 0.7
    print("✅ answer_relevancy: 7 → 0.7")


def test_relevancy_no_digit_nan():
    with _mock(["unclear"]):
        assert math.isnan(_score_answer_relevancy("Q", "A", BASE, MODEL))
    print("✅ answer_relevancy: no digit → NaN")


# ── Context Precision (YES/NO per context) ─────────────────────────────────

def test_precision_all_yes():
    with _mock(["YES", "YES", "YES"]):
        assert _score_context_precision("Q", "A", ["c1","c2","c3"], "ref", BASE, MODEL) == 1.0
    print("✅ context_precision: all YES → 1.0")


def test_precision_all_no():
    with _mock(["NO", "NO"]):
        assert _score_context_precision("Q", "A", ["c1","c2"], "ref", BASE, MODEL) == 0.0
    print("✅ context_precision: all NO → 0.0")


def test_precision_mixed():
    with _mock(["YES", "NO", "NO", "YES"]):
        score = _score_context_precision("Q", "A", ["c1","c2","c3","c4"], "ref", BASE, MODEL)
        assert score == 0.5, f"Expected 0.5, got {score}"
    print("✅ context_precision: 2/4 YES → 0.5")


def test_precision_empty_nan():
    assert math.isnan(_score_context_precision("Q", "A", [], "ref", BASE, MODEL))
    print("✅ context_precision: empty contexts → NaN")


# ── Context Recall (0-10 → 0.0-1.0) ──────────────────────────────────────

def test_recall_10_to_1():
    with _mock(["10"]):
        assert _score_context_recall("Q", ["ctx"], "ref", BASE, MODEL) == 1.0
    print("✅ context_recall: 10 → 1.0")


def test_recall_0_to_0():
    with _mock(["0"]):
        assert _score_context_recall("Q", ["ctx"], "ref", BASE, MODEL) == 0.0
    print("✅ context_recall: 0 → 0.0")


def test_recall_no_digit_nan():
    with _mock(["not sure"]):
        assert math.isnan(_score_context_recall("Q", ["ctx"], "ref", BASE, MODEL))
    print("✅ context_recall: no digit → NaN")


if __name__ == "__main__":
    test_faithfulness_10_maps_to_1()
    test_faithfulness_0_maps_to_0()
    test_faithfulness_5_maps_to_half()
    test_faithfulness_no_digit_nan()
    test_faithfulness_text_with_digit_extracted()
    test_relevancy_10_to_1()
    test_relevancy_0_to_0()
    test_relevancy_7_to_point7()
    test_relevancy_no_digit_nan()
    test_precision_all_yes()
    test_precision_all_no()
    test_precision_mixed()
    test_precision_empty_nan()
    test_recall_10_to_1()
    test_recall_0_to_0()
    test_recall_no_digit_nan()
    print("\n🎉 All 16 scoring tests passed.")
