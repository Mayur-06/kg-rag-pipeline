"""
src/retrieval.py

Step 2c: Hybrid retriever = dense (Pinecone) + sparse (BM25) fused with
Reciprocal Rank Fusion, then expanded from child nodes back to their
parent context.

Why RRF and not a weighted score average: dense cosine similarities and
BM25 scores live on completely different, non-comparable numeric scales
(cosine is bounded [-1,1]; BM25 is an unbounded, corpus-dependent score
that can range from 0 to dozens). Averaging or weighting them directly
requires brittle manual calibration that breaks every time the corpus
changes size. RRF sidesteps this entirely by fusing on *rank position*
instead of raw score, which is why it's the standard choice for hybrid
search in production systems (Elasticsearch, Weaviate, Azure AI Search
all implement some form of it).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.vector_stores import VectorStoreQuery

from src.chunking import build_parent_lookup
from src.config import get_settings
from src.sparse_index import BM25Index, tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """A single fused-and-expanded retrieval result, ready for reranking."""
    node_id: str  # child node id (what actually matched the query)
    parent_id: str | None
    text: str  # PARENT text if parent expansion succeeded, else child text
    source_file: str
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None


def dense_retrieve(index, query_str: str, top_k: int) -> list[tuple[str, float]]:
    """Returns [(node_id, similarity_score), ...] from Pinecone, ranked
    descending. Goes through the LlamaIndex VectorStoreIndex retriever
    rather than calling Pinecone directly so embedding model selection
    stays centralized in dense_index.py."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes_with_scores = retriever.retrieve(query_str)
    return [(n.node.node_id, n.score if n.score is not None else 0.0) for n in nodes_with_scores]


def sparse_retrieve(bm25_index: BM25Index, query_str: str, top_k: int) -> list[tuple[str, float]]:
    return bm25_index.query(query_str, top_k=top_k)


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Standard RRF: score(doc) = sum over each ranking it appears in of
    1 / (k + rank), rank starting at 1. k=60 is the value used in the
    original Cormack et al. RRF paper and most production hybrid-search
    implementations; it's a damping constant that prevents rank-1 in a
    single list from completely dominating the fused score, so a document
    that ranks well in BOTH lists (even at rank 5-10 in each) can beat a
    document that's rank-1 in only one list.

    Returns node_ids sorted descending by fused score.
    """
    fused_scores: dict[str, float] = {}

    for rank, (node_id, _score) in enumerate(dense_results, start=1):
        fused_scores[node_id] = fused_scores.get(node_id, 0.0) + 1.0 / (k + rank)

    for rank, (node_id, _score) in enumerate(sparse_results, start=1):
        fused_scores[node_id] = fused_scores.get(node_id, 0.0) + 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Owns both retrieval backends plus the parent-node lookup table, and
    exposes a single `.retrieve(query)` that returns fully fused,
    parent-expanded context ready for the reranker."""

    def __init__(self, dense_index, bm25_index: BM25Index, parent_nodes: list[BaseNode]):
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.parent_lookup = build_parent_lookup(parent_nodes)
        # node_id -> node, for child nodes we need to expand from RRF results
        self.child_lookup = {n.node_id: n for n in bm25_index.nodes}
        self.settings = get_settings()

    def retrieve(self, query_str: str, top_k: int | None = None) -> list[RetrievedContext]:
        top_k = top_k or self.settings.retrieval_top_k

        dense_results = dense_retrieve(self.dense_index, query_str, top_k=top_k)
        sparse_results = sparse_retrieve(self.bm25_index, query_str, top_k=top_k)
        logger.info(f"Dense hits: {len(dense_results)} | Sparse hits: {len(sparse_results)}")

        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=self.settings.rrf_k_constant)
        fused = fused[:top_k]

        dense_rank_map = {nid: i + 1 for i, (nid, _) in enumerate(dense_results)}
        sparse_rank_map = {nid: i + 1 for i, (nid, _) in enumerate(sparse_results)}

        contexts: list[RetrievedContext] = []
        for node_id, rrf_score in fused:
            child = self.child_lookup.get(node_id)
            if child is None:
                continue  # node existed in dense store but not in this BM25 snapshot; skip rather than guess

            parent_id = child.metadata.get("parent_id")
            parent_node = self.parent_lookup.get(parent_id) if parent_id else None

            # Parent expansion: hand the LLM the *paragraph/table*, not the
            # single sentence that happened to match. Falls back to the
            # child's own text if no parent was tracked (shouldn't happen
            # in practice, but better degraded than a crash).
            expanded_text = parent_node.get_content() if parent_node is not None else child.get_content()

            contexts.append(
                RetrievedContext(
                    node_id=node_id,
                    parent_id=parent_id,
                    text=expanded_text,
                    source_file=child.metadata.get("source_file", "unknown"),
                    rrf_score=rrf_score,
                    dense_rank=dense_rank_map.get(node_id),
                    sparse_rank=sparse_rank_map.get(node_id),
                )
            )

        logger.info(f"Fused + parent-expanded to {len(contexts)} candidate context(s).")
        return contexts

    def as_node_with_score_list(self, contexts: list[RetrievedContext]) -> list[NodeWithScore]:
        """Adapter so the rerank step (built on LlamaIndex's
        BaseNodePostprocessor interface) can consume our fused results
        without retrieval.py and rerank.py needing to share a bespoke
        data type."""
        out = []
        for ctx in contexts:
            node = TextNode(
                text=ctx.text,
                id_=ctx.node_id,
                metadata={"source_file": ctx.source_file, "parent_id": ctx.parent_id},
            )
            out.append(NodeWithScore(node=node, score=ctx.rrf_score))
        return out
