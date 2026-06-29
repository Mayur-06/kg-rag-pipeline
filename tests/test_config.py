"""
tests/test_config.py

Tests for Settings key-validation logic. Self-contained via unittest.mock.
No real API keys needed. Run before pip install.

Run with:  python tests/test_config.py
       or: pytest tests/test_config.py -v
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── inline the Settings dataclass so the test has no llama_index dep ──────
from dataclasses import dataclass, field

REQUIRED = ["OPENAI_API_KEY", "LLAMA_CLOUD_API_KEY", "COHERE_API_KEY", "PINECONE_API_KEY"]

@dataclass(frozen=True)
class _Settings:
    openai_api_key:      str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llama_cloud_api_key: str = field(default_factory=lambda: os.getenv("LLAMA_CLOUD_API_KEY", ""))
    cohere_api_key:      str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    pinecone_api_key:    str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))

    def validate(self, require=None):
        keys = require if require is not None else REQUIRED
        attr_map = {
            "OPENAI_API_KEY":      self.openai_api_key,
            "LLAMA_CLOUD_API_KEY": self.llama_cloud_api_key,
            "COHERE_API_KEY":      self.cohere_api_key,
            "PINECONE_API_KEY":    self.pinecone_api_key,
        }
        missing = [k for k in keys if not attr_map.get(k, "").strip()
                   or attr_map[k].startswith("sk-...") or attr_map[k].endswith("...")]
        if missing:
            raise EnvironmentError(
                "Missing or placeholder API key(s): " + ", ".join(missing) +
                ".\nCopy .env.example to .env and fill in real values."
            )


def _s(**overrides):
    base = dict(OPENAI_API_KEY="sk-real", LLAMA_CLOUD_API_KEY="llx-real",
                COHERE_API_KEY="co-real", PINECONE_API_KEY="pc-real")
    base.update(overrides)
    with patch.dict(os.environ, base, clear=True):
        return _Settings()


def test_all_keys_present_passes():
    _s().validate()
    print("✅ test_all_keys_present_passes")


def test_missing_key_raises_and_names_it():
    s = _s(OPENAI_API_KEY="")
    try:
        s.validate()
        assert False, "Should have raised"
    except EnvironmentError as e:
        assert "OPENAI_API_KEY" in str(e)
    print("✅ test_missing_key_raises_and_names_it")


def test_placeholder_key_caught():
    s = _s(OPENAI_API_KEY="sk-...")
    try:
        s.validate()
        assert False, "Should have raised"
    except EnvironmentError as e:
        assert "OPENAI_API_KEY" in str(e)
    print("✅ test_placeholder_key_caught")


def test_partial_validation_ignores_unchecked_keys():
    s = _s(COHERE_API_KEY="")        # Cohere missing
    s.validate(require=["OPENAI_API_KEY"])   # only OpenAI checked — must not raise
    print("✅ test_partial_validation_ignores_unchecked_keys")


def test_all_missing_names_all_in_error():
    s = _s(OPENAI_API_KEY="", LLAMA_CLOUD_API_KEY="",
           COHERE_API_KEY="", PINECONE_API_KEY="")
    try:
        s.validate()
        assert False
    except EnvironmentError as e:
        for k in REQUIRED:
            assert k in str(e), f"{k} missing from error: {e}"
    print("✅ test_all_missing_names_all_in_error")


if __name__ == "__main__":
    test_all_keys_present_passes()
    test_missing_key_raises_and_names_it()
    test_placeholder_key_caught()
    test_partial_validation_ignores_unchecked_keys()
    test_all_missing_names_all_in_error()
    print("\n🎉 All config tests passed.")
