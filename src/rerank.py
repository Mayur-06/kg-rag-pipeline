"""
src/rerank.py  —  Local reranking edition

Reranker: FlashRank (ms-marco-MiniLM-L-12-v2)
  • Runs entirely on CPU, no API key, no cost
  • Downloads ~120 MB on first run, cached permanently
  • Cross-encoder architecture: scores query-document relevance jointly
    rather than comparing independent vectors, same principle as Cohere
  • ms-marco-MiniLM-L-12-v2 is trained on MS MARCO passage retrieval —
    exactly the right training distribution for document Q&A

Compression: same deterministic filler-stripping as before (unchanged).
"""
from __future__ import annotations

import logging
import re

from flashrank import Ranker, RerankRequest

from src.config import get_settings
from src.retrieval import RetrievedContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_FILLER_PATTERNS = [
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.MULTILINE),
    re.compile(r"^\s*\[image\]\s*$",        re.MULTILINE | re.IGNORECASE),
    re.compile(r"\n{3,}"),
    re.compile(r"[ \t]{2,}"),
]

# Module-level singleton so the model is loaded once per process,
# not once per query call.
_ranker: Ranker | None = None


def _get_ranker(model_name: str) -> Ranker:
    global _ranker
    if _ranker is None:
        logger.info(f"Loading local rerank model: {model_name} (downloads ~120 MB on first run) ...")
        _ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank_cache")
        logger.info("✅ Rerank model loaded.")
    return _ranker


def compress_text(text: str) -> str:
    out = _FILLER_PATTERNS[0].sub("", text)
    out = _FILLER_PATTERNS[1].sub("", out)
    out = _FILLER_PATTERNS[2].sub("\n\n", out)
    out = _FILLER_PATTERNS[3].sub(" ", out)
    return out.strip()


def rerank_and_compress(
    query_str: str,
    contexts: list[RetrievedContext],
    top_n: int | None = None,
) -> list[RetrievedContext]:
    """Rerank `contexts` with FlashRank, keep top_n, compress each survivor.

    Returns a new list of RetrievedContext with rrf_score overwritten by
    the FlashRank relevance score so downstream logging reflects the final
    ordering.
    """
    if not contexts:
        return []

    settings = get_settings()
    top_n    = top_n or settings.rerank_top_n
    ranker   = _get_ranker(settings.flashrank_model)

    passages = [{"id": i, "text": ctx.text} for i, ctx in enumerate(contexts)]
    request  = RerankRequest(query=query_str, passages=passages)

    logger.info(f"Reranking {len(contexts)} candidate(s) locally with FlashRank, keeping top {top_n} ...")
    results = ranker.rerank(request)

    reranked: list[RetrievedContext] = []
    for result in results[:top_n]:
        original      = contexts[result["id"]]
        compressed    = compress_text(original.text)
        reranked.append(RetrievedContext(
            node_id     = original.node_id,
            parent_id   = original.parent_id,
            text        = compressed,
            source_file = original.source_file,
            rrf_score   = float(result.get("score", 0.0)),
            dense_rank  = original.dense_rank,
            sparse_rank = original.sparse_rank,
        ))

    logger.info(f"✅ Reranked to {len(reranked)} final context(s).")
    return reranked
