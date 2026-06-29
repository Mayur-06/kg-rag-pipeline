"""
src/config.py  —  Centralized configuration (free / local-model edition)

API keys now required:
  PINECONE_API_KEY   — still needed (Pinecone has a free tier: 1 project,
                       1 index, 2 GB storage — more than enough for this corpus)

No longer required (removed):
  OPENAI_API_KEY     — embeddings now run locally via HuggingFace
  LLAMA_CLOUD_API_KEY — LlamaParse markdown already cached; not needed again
                        unless you add new PDFs
  COHERE_API_KEY     — reranking now runs locally via FlashRank

LLM (generation) options — set LLM_BACKEND in .env:
  "ollama"   (default) — runs llama3.2 locally via Ollama. Free, private.
                         Install: https://ollama.com  then: ollama pull llama3.2
  "openai"             — falls back to gpt-4o-mini if you add credits later
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_PDF_DIR   = DATA_DIR / "raw_pdfs"
PARSED_MD_DIR = DATA_DIR / "parsed_markdown"
EVAL_RESULTS_DIR = DATA_DIR / "eval_results"
STORAGE_DIR   = DATA_DIR / "storage"

for d in (RAW_PDF_DIR, PARSED_MD_DIR, EVAL_RESULTS_DIR, STORAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

# Only Pinecone is strictly required now — everything else runs locally.
REQUIRED_KEYS = ["PINECONE_API_KEY"]


@dataclass(frozen=True)
class Settings:
    # ── Vector store (still cloud, free tier) ────────────────────────────
    pinecone_api_key:   str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "merger-kg-rag"))
    pinecone_cloud:     str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    pinecone_region:    str = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))

    # ── Local embedding model (HuggingFace, runs on CPU, no API key) ─────
    # bge-small-en-v1.5: 33M params, 384-dim, best-in-class retrieval score
    # for its size on BEIR benchmark. Downloads ~130 MB on first run to
    # ~/.cache/huggingface/hub and is cached there permanently.
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_dim:   int = field(default_factory=lambda: int(os.getenv("EMBED_DIM", "384")))

    # ── LLM backend ───────────────────────────────────────────────────────
    # "ollama"  → local llama3.2 via Ollama (free, recommended)
    # "openai"  → gpt-4o-mini via OpenAI API (requires credits)
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "ollama").lower())
    llm_model:   str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2"))
    # Separate model for Ragas scoring — can be a smaller/faster model than
    # the generation LLM since scoring only needs yes/no claim judgments.
    # llama3.2:1b runs ~4x faster than llama3.2 on CPU with acceptable
    # accuracy for faithfulness/precision/recall scoring.
    # Run: ollama pull llama3.2:1b   (only ~1.3 GB vs 2 GB for llama3.2)
    ragas_llm_model: str = field(default_factory=lambda: os.getenv("RAGAS_LLM_MODEL", "llama3.2:1b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    # Connection timeout for Ollama (seconds)
    ollama_connect_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5.0")))

    # Kept for openai fallback path
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # ── Reranker (FlashRank, local, no API key) ───────────────────────────
    # ms-marco-MiniLM-L-12-v2 is the standard FlashRank default model.
    # Downloads ~120 MB on first run, cached permanently.
    flashrank_model: str = field(default_factory=lambda: os.getenv(
        "FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2"
    ))

    # ── Retrieval tuning ─────────────────────────────────────────────────
    rerank_top_n:    int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_N",    "5")))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "20")))
    rrf_k_constant:  int = field(default_factory=lambda: int(os.getenv("RRF_K_CONSTANT",  "60")))

    # ── LlamaParse (only needed if re-ingesting new PDFs) ─────────────────
    llama_cloud_api_key: str = field(default_factory=lambda: os.getenv("LLAMA_CLOUD_API_KEY", ""))

    def validate(self, require: list[str] | None = None) -> None:
        keys_to_check = require if require is not None else REQUIRED_KEYS
        attr_map = {
            "PINECONE_API_KEY":    self.pinecone_api_key,
            "OPENAI_API_KEY":      self.openai_api_key,
            "LLAMA_CLOUD_API_KEY": self.llama_cloud_api_key,
        }
        missing = [
            k for k in keys_to_check
            if not attr_map.get(k, "").strip()
            or attr_map.get(k, "").endswith("...")
        ]
        if missing:
            raise EnvironmentError(
                "Missing or placeholder value(s): "
                + ", ".join(missing)
                + ".\nCheck your .env file."
            )


def get_settings() -> Settings:
    return Settings()


if __name__ == "__main__":
    s = get_settings()
    try:
        s.validate()
        print(f"✅ Config OK  |  embed={s.embed_model} ({s.embed_dim}d)"
              f"  |  llm_backend={s.llm_backend}  |  llm={s.llm_model}")
    except EnvironmentError as e:
        print(f"⚠️  {e}", file=sys.stderr)
        sys.exit(1)
