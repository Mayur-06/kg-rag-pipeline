"""
src/load_pipeline.py

Small shared helper: reconstitute (dense_index, bm25_index, parent_nodes)
from cached/persisted state so main.py and evaluate.py don't duplicate
this loading logic or accidentally diverge from each other.
"""
from __future__ import annotations

import pickle

from src.build_pipeline import PARENT_NODES_CACHE
from src.dense_index import load_dense_index
from src.sparse_index import BM25Index


def load_pipeline_components():
    if not PARENT_NODES_CACHE.exists():
        raise FileNotFoundError(
            f"No cached parent nodes at {PARENT_NODES_CACHE}. "
            f"Run `python -m src.build_pipeline` first to ingest documents and build the indexes."
        )
    with open(PARENT_NODES_CACHE, "rb") as f:
        parent_nodes = pickle.load(f)

    dense_index = load_dense_index()
    bm25_index = BM25Index.load()

    return dense_index, bm25_index, parent_nodes
