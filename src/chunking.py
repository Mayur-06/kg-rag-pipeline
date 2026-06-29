"""
src/chunking.py

Step 1b: Hierarchical parent/child node parsing.

The core problem this solves: a flat text splitter chops a 400-word
paragraph into a couple of arbitrary 200-token chunks with no regard for
sentence or table boundaries. When chunk boundaries cut through a table or
a multi-clause legal sentence, the LLM gets a fragment it can't reason
over — e.g. "...governed by Section 4.2 of the" with no Section 4.2 in
sight.

The fix used here is the classic small-to-big retrieval pattern:
  - PARENT nodes = whole sections/paragraphs/tables (large, full context,
    via LlamaIndex's MarkdownNodeParser which respects heading and table
    boundaries from the Markdown LlamaParse produced).
  - CHILD nodes = individual sentences within each parent (small, precise,
    these are what actually get embedded and searched).

At query time we search over child nodes (sentence-level precision finds
the exact needle), but we *return* the parent node's full text to the LLM
(paragraph/table-level context so the answer isn't built from a fragment).
This is LlamaIndex's NodeReferenceMode pattern, implemented explicitly here
so it's auditable rather than hidden behind a one-line abstraction.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode

from src.config import PARSED_MD_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Parents: split on Markdown structure (headings, tables) — NOT on a fixed
# token count, so a table is one parent node regardless of its size and
# never gets sliced down the middle.
PARENT_CHUNK_PARSER = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)

# Children: split each parent into sentence-level units. Small chunk_size
# here is a safety net for unusually long "sentences" (e.g. a dense legal
# clause); chunk_overlap=0 because parent->child is a containment
# relationship, not a sliding window — overlap would just duplicate text
# in the index.
CHILD_SENTENCE_SPLITTER = SentenceSplitter(chunk_size=256, chunk_overlap=0)

TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")


def load_markdown_documents(md_dir: Path = PARSED_MD_DIR) -> list[Document]:
    md_paths = sorted(md_dir.glob("*.md"))
    if not md_paths:
        raise FileNotFoundError(
            f"No parsed Markdown found in {md_dir}. Run `python -m src.ingest` first."
        )
    docs = []
    for path in md_paths:
        text = path.read_text(encoding="utf-8")
        docs.append(Document(text=text, metadata={"source_file": path.stem}))
    logger.info(f"Loaded {len(docs)} markdown document(s) for chunking.")
    return docs


def _is_table_block(text: str) -> bool:
    """A parent node counts as a table if a majority of its non-blank lines
    are Markdown table rows. Tables are never sentence-split — splitting a
    table "sentence-by-sentence" would shred rows and destroy the exact
    structure LlamaParse worked to preserve."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    table_lines = sum(1 for l in lines if TABLE_ROW_PATTERN.match(l))
    return table_lines / len(lines) > 0.5


def build_hierarchical_nodes(documents: list[Document]) -> tuple[list[BaseNode], list[BaseNode]]:
    """Returns (parent_nodes, child_nodes).

    Every child node carries metadata["parent_id"] pointing back to its
    parent's node_id, and a NodeRelationship.PARENT pointer for libraries
    that respect the LlamaIndex relationship graph directly. We set both
    because the custom RRF/rerank code in retrieval.py reads the metadata
    field directly (simpler, explicit), while staying compatible with any
    stock LlamaIndex component that expects the relationship graph.
    """
    parent_nodes = PARENT_CHUNK_PARSER.get_nodes_from_documents(documents)
    logger.info(f"Created {len(parent_nodes)} parent node(s) from {len(documents)} document(s).")

    all_child_nodes: list[BaseNode] = []
    table_parents = 0

    for parent in parent_nodes:
        parent_text = parent.get_content()
        if not parent_text.strip():
            continue

        if _is_table_block(parent_text):
            # Tables stay atomic: the "child" IS the parent, verbatim.
            # This guarantees a query matching any cell value retrieves
            # the entire table, never a single mangled row.
            table_parents += 1
            child = TextNode(
                text=parent_text,
                metadata={**parent.metadata, "parent_id": parent.node_id, "node_type": "table_atomic"},
            )
            child.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent.node_id)
            all_child_nodes.append(child)
            continue

        sentence_nodes = CHILD_SENTENCE_SPLITTER.get_nodes_from_documents(
            [Document(text=parent_text, metadata=parent.metadata)]
        )
        for sent_node in sentence_nodes:
            child = TextNode(
                text=sent_node.get_content(),
                metadata={**parent.metadata, "parent_id": parent.node_id, "node_type": "sentence"},
            )
            child.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent.node_id)
            all_child_nodes.append(child)

    logger.info(
        f"Created {len(all_child_nodes)} child node(s) "
        f"({table_parents} atomic table parent(s), rest sentence-level)."
    )
    return parent_nodes, all_child_nodes


def build_parent_lookup(parent_nodes: list[BaseNode]) -> dict[str, BaseNode]:
    """node_id -> parent node, used at query time to expand a matched
    child back into its full paragraph/table context before it's handed
    to the reranker / LLM."""
    return {p.node_id: p for p in parent_nodes}


if __name__ == "__main__":
    docs = load_markdown_documents()
    parents, children = build_hierarchical_nodes(docs)
    print(f"\nParents: {len(parents)} | Children: {len(children)}")
    if children:
        sample = children[0]
        print("\n--- Sample child node ---")
        print(sample.get_content()[:200])
        print(f"node_type={sample.metadata.get('node_type')} parent_id={sample.metadata.get('parent_id')}")
