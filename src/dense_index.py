"""
src/dense_index.py  —  Local embedding edition

Embedding model: BAAI/bge-small-en-v1.5 (HuggingFace, runs on CPU)
  • 33M parameters, 384-dimensional output
  • Downloads ~130 MB to ~/.cache/huggingface/hub on first run, cached forever
  • No API key, no quota, no cost
  • BEIR retrieval benchmark score comparable to text-embedding-ada-002

Vector store: Pinecone (free tier)
  • Still cloud-hosted — Pinecone's free tier gives 1 index, 2 GB, which
    easily holds 6783 × 384-dim float32 vectors (~10 MB total)

Dimension change note
─────────────────────
bge-small-en-v1.5 outputs 384-dim vectors. The previous Pinecone index was
created for 1536-dim (OpenAI). Pinecone indexes are fixed-dimension, so
build_dense_index() now detects a dimension mismatch, deletes the old index,
and recreates it at the correct dimension. This is automatic and only
happens once. Pinecone delete+recreate takes ~30 seconds on the free tier.
"""
from __future__ import annotations

import logging
import time

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _get_embed_model(settings) -> HuggingFaceEmbedding:
    """Load bge-small-en-v1.5 from local HuggingFace cache (or download it
    on first run). The query_prefix is required by BGE models — they were
    trained with the prefix 'Represent this sentence for searching relevant
    passages:' on query strings, which improves retrieval by ~2–4% vs no
    prefix. Document strings get no prefix (asymmetric retrieval design)."""
    logger.info(f"Loading local embedding model: {settings.embed_model}")
    return HuggingFaceEmbedding(
        model_name=settings.embed_model,
        query_instruction="Represent this sentence for searching relevant passages: ",
        text_instruction="",   # no prefix for documents at index time
        device="cpu",          # explicit; falls back to GPU if available via env
        cache_folder=None,     # uses HuggingFace default: ~/.cache/huggingface/hub
    )


def _get_index_dimension(pc: Pinecone, index_name: str) -> int | None:
    """Returns the dimension of an existing Pinecone index, or None if it
    doesn't exist. Used to detect the 1536→384 dimension mismatch."""
    existing = {idx["name"]: idx for idx in pc.list_indexes()}
    if index_name not in existing:
        return None
    try:
        return existing[index_name]["dimension"]
    except (KeyError, TypeError):
        return None


def _ensure_pinecone_index(pc: Pinecone, index_name: str, dim: int, cloud: str, region: str):
    """Create the index if it doesn't exist. If it exists at the WRONG
    dimension (i.e. was built with OpenAI 1536-dim), delete and recreate it."""
    existing_dim = _get_index_dimension(pc, index_name)

    if existing_dim is not None and existing_dim != dim:
        logger.warning(
            f"Pinecone index '{index_name}' exists at dimension={existing_dim} "
            f"but the local model needs dimension={dim}. "
            f"Deleting and recreating (this takes ~30 seconds on the free tier)..."
        )
        pc.delete_index(index_name)
        # Pinecone free tier needs a moment after delete before recreate
        time.sleep(15)

    if _get_index_dimension(pc, index_name) is None:
        logger.info(f"Creating Pinecone index '{index_name}' (dim={dim}, {cloud}/{region}) ...")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait for the index to be ready
        for _ in range(20):
            time.sleep(3)
            try:
                status = pc.describe_index(index_name).status
                if status.get("ready", False):
                    break
            except Exception:
                pass
        logger.info(f"✅ Index '{index_name}' ready.")
    else:
        logger.info(f"Reusing existing Pinecone index '{index_name}' (dim={existing_dim}).")

    return pc.Index(index_name)


def _pinecone_vector_count(pinecone_index) -> int:
    try:
        stats = pinecone_index.describe_index_stats()
        return int(stats.get("total_vector_count", 0))
    except Exception as e:
        logger.warning(f"Could not read Pinecone vector count: {e}. Assuming 0.")
        return 0


def build_dense_index(child_nodes: list[BaseNode], force: bool = False) -> VectorStoreIndex:
    """Embed child nodes locally and upsert to Pinecone.

    Skip guard: if Pinecone already holds >= len(child_nodes) vectors and
    force=False, the entire embed+upsert pass is skipped and the existing
    index is loaded instead. Safe to call on every re-run.
    """
    settings = get_settings()
    settings.validate(require=["PINECONE_API_KEY"])

    embed_model = _get_embed_model(settings)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    pinecone_index = _ensure_pinecone_index(
        pc,
        index_name=settings.pinecone_index_name,
        dim=settings.embed_dim,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )

    if not force:
        existing_count = _pinecone_vector_count(pinecone_index)
        if existing_count >= len(child_nodes):
            logger.info(
                f"⏭️  Skipping embed+upsert — Pinecone already contains "
                f"{existing_count} vectors (need {len(child_nodes)}). "
                f"Pass --force-embed to re-embed."
            )
            return load_dense_index()

    vector_store    = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info(
        f"Embedding {len(child_nodes)} child node(s) locally with "
        f"{settings.embed_model} then upserting to Pinecone ..."
    )
    logger.info("  ⏱  CPU embedding is slower than GPU (~2–5 min for 6 k nodes on a laptop).")
    index = VectorStoreIndex(
        nodes=child_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info("✅ Dense index build complete.")
    return index


def load_dense_index() -> VectorStoreIndex:
    """Reconnect to an already-populated Pinecone index using the local
    embedding model (needed at query time to embed the query string)."""
    settings = get_settings()
    settings.validate(require=["PINECONE_API_KEY"])

    embed_model   = _get_embed_model(settings)
    pc            = Pinecone(api_key=settings.pinecone_api_key)
    pinecone_index = pc.Index(settings.pinecone_index_name)
    vector_store  = PineconeVectorStore(pinecone_index=pinecone_index)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
