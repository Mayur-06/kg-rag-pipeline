"""
src/sparse_index.py

Step 2b: Sparse keyword index (BM25) over the same child nodes used for the
dense index.

Why this matters for M&A documents specifically: dense embeddings are
excellent at "what concept is this about" but routinely fail on exact
strings that carry no semantic meaning of their own — a contract clause
ID like "Section 7.3(b)(ii)", a product SKU, a server hostname, a CVE
number, a defendant's name. BM25 matches those tokens exactly where a
dense vector might consider three different clause numbers "similar"
because they sit in similar sentences. Running both and fusing (RRF, in
retrieval.py) gets the precision of both families of failure mode covered.
"""
from __future__ import annotations

import logging
import pickle
import re

from llama_index.core.schema import BaseNode
from rank_bm25 import BM25Okapi

from src.config import STORAGE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BM25_CACHE_PATH = STORAGE_DIR / "bm25_index.pkl"

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")


def tokenize(text: str) -> list[str]:
    """Deliberately simple, deterministic tokenizer (lowercased word/number
    chunks, keeping internal dots/dashes/underscores). This is intentional:
    we want "section_4.2" or "vpc-0a1b2c3d" to survive as single tokens
    rather than being shattered by a generic NLP tokenizer, since those are
    exactly the exact-match strings BM25 is here to catch."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Thin wrapper around rank_bm25 that keeps node_id <-> node mapping
    alongside the BM25 statistics, since rank_bm25 itself only knows about
    token lists and integer positions."""

    def __init__(self, nodes: list[BaseNode]):
        self.nodes = nodes
        self.node_ids = [n.node_id for n in nodes]
        self._corpus_tokens = [tokenize(n.get_content()) for n in nodes]
        self.bm25 = BM25Okapi(self._corpus_tokens)

    def query(self, query_str: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Returns [(node_id, bm25_score), ...] sorted descending by score."""
        query_tokens = tokenize(query_str)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.node_ids, scores), key=lambda x: x[1], reverse=True)
        return [(nid, score) for nid, score in ranked[:top_k] if score > 0]

    def save(self, path=BM25_CACHE_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"nodes": self.nodes}, f)
        logger.info(f"✅ Cached BM25 index ({len(self.nodes)} nodes) -> {path}")

    @classmethod
    def load(cls, path=BM25_CACHE_PATH) -> "BM25Index":
        if not path.exists():
            raise FileNotFoundError(
                f"No cached BM25 index at {path}. Run `python -m src.build_pipeline` first."
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(payload["nodes"])


def build_bm25_index(child_nodes: list[BaseNode]) -> BM25Index:
    logger.info(f"Building BM25 index over {len(child_nodes)} child node(s) ...")
    bm25_index = BM25Index(child_nodes)
    bm25_index.save()
    return bm25_index
