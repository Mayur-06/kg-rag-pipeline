"""
src/build_pipeline.py

Re-run safety — every stage is independently guarded:

  Stage 1   LlamaParse (PDF → Markdown)
            Skipped per-PDF if the .md file already exists. [ingest.py]

  Stage 1b  Hierarchical chunking (markdown → parent/child nodes)
            Skipped if parent_nodes.pkl exists — no longer requires
            bm25_index.pkl, which doesn't exist yet when Stage 2a crashes.
            Child nodes are re-derived from parent_nodes in-memory (fast,
            no API calls) when the BM25 pkl is absent.

  Stage 2a  Dense index (local BGE embeddings → Pinecone)
            Skipped if Pinecone already holds >= N vectors. [dense_index.py]

  Stage 2b  BM25 index
            Skipped if bm25_index.pkl exists.

Flags
─────
  --force-ingest    Re-parse PDFs even if .md files exist
  --force-chunk     Re-chunk even if parent_nodes.pkl exists
  --force-embed     Re-embed and re-upsert even if Pinecone looks full
  --force-bm25      Rebuild BM25 index even if .pkl exists
  --force           Equivalent to all four flags above (full clean rebuild)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pickle
import sys

from src.chunking import build_hierarchical_nodes, load_markdown_documents
from src.config import STORAGE_DIR, get_settings
from src.dense_index import build_dense_index
from src.ingest import ingest_all_pdfs
from src.sparse_index import BM25_CACHE_PATH, BM25Index, build_bm25_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARENT_NODES_CACHE = STORAGE_DIR / "parent_nodes.pkl"
CHILD_NODES_CACHE  = STORAGE_DIR / "child_nodes.pkl"   # new: persisted independently of BM25


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the merger RAG pipeline indexes.")
    p.add_argument("--force-ingest", action="store_true",
                   help="Re-parse PDFs via LlamaParse even if .md files already exist.")
    p.add_argument("--force-chunk",  action="store_true",
                   help="Re-chunk markdown even if parent_nodes.pkl already exists.")
    p.add_argument("--force-embed",  action="store_true",
                   help="Re-embed and re-upsert all nodes into Pinecone, even if already populated.")
    p.add_argument("--force-bm25",   action="store_true",
                   help="Rebuild BM25 index even if bm25_index.pkl already exists.")
    p.add_argument("--force", action="store_true",
                   help="Force all stages (equivalent to all four --force-* flags).")
    return p.parse_args()


def _load_cached_nodes():
    """Load parent + child nodes from their individual pkl caches.

    Both are now saved independently so this works even when bm25_index.pkl
    doesn't exist yet (i.e. Stage 2b never completed because 2a crashed).
    """
    logger.info(f"⏭️  Skipping chunking — loading cached nodes from disk")

    with open(PARENT_NODES_CACHE, "rb") as f:
        parent_nodes = pickle.load(f)

    with open(CHILD_NODES_CACHE, "rb") as f:
        child_nodes = pickle.load(f)

    logger.info(
        f"  Loaded {len(parent_nodes)} parent node(s) and "
        f"{len(child_nodes)} child node(s) from cache."
    )
    return parent_nodes, child_nodes


def main():
    args   = _parse_args()
    force_all = args.force
    settings  = get_settings()
    settings.validate()

    # ── Stage 1: Ingest (PDF → Markdown) ────────────────────────────────────
    logger.info("=== STAGE 1: Ingestion (PDF → Markdown via LlamaParse) ===")
    asyncio.run(ingest_all_pdfs(force_ingest=args.force_ingest))  # skips per-file if .md already exists

    # ── Stage 1b: Chunking ───────────────────────────────────────────────────
    logger.info("=== STAGE 1b: Hierarchical chunking ===")

    # FIX: guard on PARENT_NODES_CACHE only — child pkl is always saved
    # alongside it, so if parent exists both exist.  BM25 pkl is NOT
    # required here because it's a Stage 2b output, not a Stage 1b output.
    chunk_cache_exists = PARENT_NODES_CACHE.exists() and CHILD_NODES_CACHE.exists()

    if not (force_all or args.force_chunk) and chunk_cache_exists:
        logger.info(
            f"⏭️  Skipping chunking — cache exists "
            f"({PARENT_NODES_CACHE.name}, {CHILD_NODES_CACHE.name}). "
            f"Use --force-chunk to re-chunk."
        )
        parent_nodes, child_nodes = _load_cached_nodes()
    else:
        reason = "--force-chunk set" if (force_all or args.force_chunk) else "no cache found"
        logger.info(f"  Running chunking ({reason}) ...")

        documents = load_markdown_documents()
        parent_nodes, child_nodes = build_hierarchical_nodes(documents)

        with open(PARENT_NODES_CACHE, "wb") as f:
            pickle.dump(parent_nodes, f)
        with open(CHILD_NODES_CACHE, "wb") as f:
            pickle.dump(child_nodes, f)
        logger.info(
            f"  Cached {len(parent_nodes)} parent node(s) → {PARENT_NODES_CACHE}\n"
            f"  Cached {len(child_nodes)} child node(s)  → {CHILD_NODES_CACHE}"
        )

    # ── Stage 2a: Dense index (local BGE embeddings → Pinecone) ─────────────
    logger.info("=== STAGE 2a: Dense index (BGE-small-en-v1.5 local embeddings → Pinecone) ===")
    build_dense_index(child_nodes, force=(force_all or args.force_embed))

    # ── Stage 2b: BM25 sparse index ─────────────────────────────────────────
    logger.info("=== STAGE 2b: Sparse index (BM25) ===")
    if not (force_all or args.force_bm25) and BM25_CACHE_PATH.exists():
        logger.info(
            f"⏭️  Skipping BM25 rebuild — {BM25_CACHE_PATH.name} exists. "
            f"Use --force-bm25 to rebuild."
        )
    else:
        build_bm25_index(child_nodes)

    logger.info(
        "🎉 Pipeline build complete.\n"
        "   Query:    python -m src.main\n"
        "   Evaluate: python -m src.evaluate\n"
        "   Flags:    python -m src.build_pipeline --help"
    )


if __name__ == "__main__":
    main()
