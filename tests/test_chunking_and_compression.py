"""
tests/test_chunking_and_compression.py

Self-contained tests for BM25 tokenizer, table detection, and context
compression. Zero external library imports — run before pip install.

Run with:  python tests/test_chunking_and_compression.py
       or: pytest tests/test_chunking_and_compression.py -v
"""
from __future__ import annotations
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── inline copies of the functions under test ─────────────────────────────

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
def _is_table(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines: return False
    return sum(1 for l in lines if _TABLE_ROW.match(l)) / len(lines) > 0.5

_FILLERS = [
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.MULTILINE),
    re.compile(r"^\s*\[image\]\s*$",        re.MULTILINE | re.IGNORECASE),
    re.compile(r"\n{3,}"),
    re.compile(r"[ \t]{2,}"),
]
def _compress(text: str) -> str:
    out = _FILLERS[0].sub("", text)
    out = _FILLERS[1].sub("", out)
    out = _FILLERS[2].sub("\n\n", out)
    out = _FILLERS[3].sub(" ", out)
    return out.strip()


# ── tests ──────────────────────────────────────────────────────────────────

def test_tokenizer_preserves_clause_number():
    tokens = _tokenize("Governed by Section 4.2(a) per vpc-0a1b2c3d.")
    assert any("4.2" in t for t in tokens), f"clause token missing: {tokens}"
    assert any("vpc-0a1b2c3d" in t for t in tokens), f"hostname token missing: {tokens}"
    print("✅ test_tokenizer_preserves_clause_number")


def test_tokenizer_lowercases():
    assert "microsoft" in _tokenize("Microsoft Corporation")
    print("✅ test_tokenizer_lowercases")


def test_table_detection_positive():
    table = (
        "| Pillar | Focus Area |\n"
        "|---|---|\n"
        "| Reliability | Fault tolerance, disaster recovery |\n"
        "| Security | IAM, encryption, detection |\n"
    )
    assert _is_table(table) is True, "Markdown table should be detected"
    print("✅ test_table_detection_positive")


def test_table_detection_negative():
    prose = (
        "Activision Blizzard discloses dependence on third-party cloud infrastructure "
        "providers for hosting live-service game backends."
    )
    assert _is_table(prose) is False, "Plain prose should NOT be detected as table"
    print("✅ test_table_detection_negative")


def test_table_detection_mixed_content():
    """A chunk that's half prose / half table should NOT be treated as
    atomic table — the majority-rows threshold must hold."""
    mixed = (
        "The following table summarises the five pillars:\n"
        "This sentence has no pipes.\n"
        "Neither does this one.\n"
        "| Pillar | Goal |\n"
        "|---|---|\n"
        "| Reliability | Uptime |\n"
    )
    # 3 table rows out of 6 non-blank lines = 50%, threshold is >50%
    assert _is_table(mixed) is False, "Mixed content below threshold should not be table"
    print("✅ test_table_detection_mixed_content")


def test_compress_strips_page_footer_and_image_placeholder():
    noisy = (
        "Page 47 of 312\n\n"
        "Microsoft agreed to pay $95.00 per share.\n\n"
        "[image]\n"
    )
    out = _compress(noisy)
    assert "Page 47 of 312" not in out, "page footer should be stripped"
    assert "[image]" not in out, "image placeholder should be stripped"
    assert "$95.00 per share" in out, f"dollar figure must survive: {out}"
    print("✅ test_compress_strips_page_footer_and_image_placeholder")


def test_compress_preserves_clause_numbers_after_whitespace_collapse():
    noisy = "Pursuant to   Section   4.2(a),   the parties agreed."
    out = _compress(noisy)
    # whitespace collapsed but clause number intact
    assert "4.2(a)" in out, f"clause number must survive: {out}"
    assert "   " not in out, f"triple spaces must be collapsed: {out}"
    print("✅ test_compress_preserves_clause_numbers_after_whitespace_collapse")


def test_compress_collapses_excessive_blank_lines():
    noisy = "First paragraph.\n\n\n\n\nSecond paragraph."
    out = _compress(noisy)
    assert "\n\n\n" not in out, f"3+ blank lines should be collapsed: {repr(out)}"
    assert "First paragraph." in out
    assert "Second paragraph." in out
    print("✅ test_compress_collapses_excessive_blank_lines")


if __name__ == "__main__":
    test_tokenizer_preserves_clause_number()
    test_tokenizer_lowercases()
    test_table_detection_positive()
    test_table_detection_negative()
    test_table_detection_mixed_content()
    test_compress_strips_page_footer_and_image_placeholder()
    test_compress_preserves_clause_numbers_after_whitespace_collapse()
    test_compress_collapses_excessive_blank_lines()
    print("\n🎉 All chunking/compression tests passed.")
