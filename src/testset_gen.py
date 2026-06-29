"""
src/testset_gen.py

Step 4a: Generate a synthetic test set of 30 (question, reference_answer)
pairs from the actual child nodes the retriever searches over.

Three generation modes, deliberately mixed:

  SINGLE-HOP (40%)
    Sample one child node, ask the LLM to produce a question AND a
    one-sentence answer grounded only in that node's parent text.
    These are the precision baseline: if the pipeline can't answer
    "what was the per-share merger price" it's broken at a basic level.

  CROSS-DOCUMENT (40%)
    Sample one node from two DIFFERENT source documents, give both
    passages to the LLM, and ask for a question that genuinely requires
    both. This is the hard case the whole pipeline is designed for:
    "Does the 10-K risk factor about cloud dependency relate to any
    AWS Well-Architected reliability pillar?"

  ADVERSARIAL (20%)
    Ask the LLM to write a plausible-sounding question whose answer is
    NOT in the corpus. The correct pipeline behaviour is to say "I cannot
    find this in the documents." These test Faithfulness: a pipeline that
    hallucinates an answer here scores 0 on that question.

reference column = a clean, short, factual answer string — NOT raw
markdown. Ragas compares the pipeline's generated answer against this
string sentence-by-sentence, so it must be readable prose.
"""
from __future__ import annotations

import json
import logging
import random
import re
import textwrap
from pathlib import Path

import httpx
import pandas as pd

from src.config import EVAL_RESULTS_DIR, get_settings
from src.sparse_index import BM25Index, BM25_CACHE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TESTSET_PATH        = EVAL_RESULTS_DIR / "synthetic_testset.csv"
DEFAULT_TESTSET_SIZE = 30
random.seed(42)   # reproducible sampling

# ── Adversarial questions hard-coded — the LLM generating "unanswerable"
# questions from corpus text reliably hallucinates an answer into the
# question itself, defeating the purpose. Hand-crafted ones are better.
ADVERSARIAL_QUESTIONS = [
    {
        "user_input": "What did Microsoft's CEO Satya Nadella personally say about Activision's workplace culture during the shareholder vote?",
        "reference":  "This information is not present in the provided documents. The SEC Form 8-K details the shareholder vote procedure and results but does not contain direct quotes from Satya Nadella about workplace culture.",
    },
    {
        "user_input": "What is Activision Blizzard's current stock price as of today?",
        "reference":  "The documents do not contain current stock price information. The 10-K and 8-K filings are point-in-time documents and do not reflect real-time market data.",
    },
    {
        "user_input": "Which AWS availability zone did Activision Blizzard use to host Call of Duty servers in 2022?",
        "reference":  "This specific operational detail is not disclosed in any of the provided documents. The 10-K mentions cloud infrastructure dependency at a high level but does not name specific AWS availability zones.",
    },
    {
        "user_input": "What was Bobby Kotick's exact severance package following the Microsoft acquisition?",
        "reference":  "The provided documents do not disclose the specific terms of Bobby Kotick's severance arrangement. The 8-K covers shareholder vote mechanics, not executive compensation details post-close.",
    },
    {
        "user_input": "How many mobile game downloads did Candy Crush record in Q3 2022 according to Newzoo?",
        "reference":  "This figure is not present in the documents. The Newzoo report covers market-level trends and does not break down downloads by individual title.",
    },
    {
        "user_input": "What is the exact network latency SLA Microsoft committed to for Activision's game servers post-acquisition?",
        "reference":  "No such SLA is disclosed in the provided documents. Neither the 8-K, 10-K, nor the AWS Well-Architected Framework whitepaper contains post-acquisition infrastructure commitments specific to this merger.",
    },
]


# ── Ollama call ────────────────────────────────────────────────────────────

def _call_ollama(base_url: str, model: str, prompt: str, temperature: float = 0.3) -> str:
    """Single synchronous call to the Ollama /api/generate endpoint.
    Returns the raw response text."""
    response = httpx.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature}},
        timeout=180.0,
    )
    response.raise_for_status()
    data = response.json()
    text = data.get("response") or data.get("message", {}).get("content", "")
    return text.strip()


def _clean_question(raw: str) -> str:
    """Strip LLM preamble and keep only the question sentence.

    llama3.2 often prepends 'Here is a question based on the passage:'
    or wraps the question in quotes. This strips all of that.
    """
    raw = raw.strip()
    # Remove quoted wrappers
    raw = raw.strip('"').strip("'")
    # If there are multiple lines, prefer the last one that ends with '?'
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in reversed(lines):
        if line.endswith("?"):
            return line
    # Fallback: last sentence ending in '?'
    sentences = re.split(r"(?<=[.!?])\s+", raw)
    for s in reversed(sentences):
        if s.endswith("?"):
            return s
    return raw.split("\n")[0].strip()


def _clean_answer(raw: str) -> str:
    """Keep only the first 1-3 sentences of the answer. Ragas compares
    this string against the pipeline's generated answer claim-by-claim,
    so brevity and factual density matter more than completeness."""
    raw = raw.strip()
    # Strip markdown headers, bullet leaders
    raw = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s*[-*]\s+", "", raw, flags=re.MULTILINE)
    sentences = re.split(r"(?<=[.!?])\s+", raw)
    return " ".join(sentences[:3]).strip()


# ── Samplers ───────────────────────────────────────────────────────────────

def _sample_nodes_by_doc(nodes, n: int, exclude_doc: str | None = None):
    """Return n randomly-sampled nodes, optionally excluding one source_file."""
    pool = [nd for nd in nodes
            if nd.metadata.get("source_file", "") != (exclude_doc or "")
            and len(nd.get_content().strip()) > 120]
    return random.sample(pool, min(n, len(pool)))


def _parent_text(node, parent_lookup: dict) -> str:
    """Return the parent paragraph/table text for a child node, or fall
    back to the child's own text. Keeps context rich for question gen."""
    pid = node.metadata.get("parent_id")
    parent = parent_lookup.get(pid) if pid else None
    text = parent.get_content() if parent else node.get_content()
    # Cap at 600 chars so the LLM prompt stays well within context window
    return textwrap.shorten(text, width=600, placeholder=" [...]")


# ── Generation functions ───────────────────────────────────────────────────

def _gen_single_hop(node, parent_lookup, base_url, model) -> dict | None:
    passage = _parent_text(node, parent_lookup)
    source  = node.metadata.get("source_file", "unknown")

    prompt = (
        "You are a corporate due-diligence analyst. Read the passage below and write:\n"
        "1. ONE specific factual question that is directly and fully answerable from the passage alone.\n"
        "2. A concise factual answer (1-2 sentences maximum) using only information in the passage.\n\n"
        "Format your response as exactly two lines:\n"
        "QUESTION: <your question ending with ?>\n"
        "ANSWER: <your concise answer>\n\n"
        "Do not include any other text, explanation, or preamble.\n\n"
        f"Passage (source: {source}):\n{passage}"
    )

    raw = _call_ollama(base_url, model, prompt, temperature=0.3)

    q_match = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    a_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE | re.DOTALL)

    if not q_match or not a_match:
        logger.debug(f"single_hop: could not parse Q/A from: {raw[:120]}")
        return None

    question = _clean_question(q_match.group(1))
    answer   = _clean_answer(a_match.group(1))

    if not question.endswith("?") or len(answer) < 10:
        return None

    return {"user_input": question, "reference": answer,
            "question_type": "single_hop", "source_docs": source}


def _gen_cross_doc(node_a, node_b, parent_lookup, base_url, model) -> dict | None:
    passage_a = _parent_text(node_a, parent_lookup)
    passage_b = _parent_text(node_b, parent_lookup)
    source_a  = node_a.metadata.get("source_file", "doc_a")
    source_b  = node_b.metadata.get("source_file", "doc_b")

    prompt = (
        "You are a corporate due-diligence analyst reviewing two passages from different documents.\n"
        "Write ONE question that can ONLY be answered by combining information from BOTH passages — "
        "not from either one alone.\n"
        "Then write a concise answer (2-3 sentences) that explicitly draws on both passages.\n\n"
        "Format your response as exactly two lines:\n"
        "QUESTION: <your cross-document question ending with ?>\n"
        "ANSWER: <your answer citing both passages>\n\n"
        "Do not include any other text, explanation, or preamble.\n\n"
        f"Passage A (source: {source_a}):\n{passage_a}\n\n"
        f"Passage B (source: {source_b}):\n{passage_b}"
    )

    raw = _call_ollama(base_url, model, prompt, temperature=0.4)

    q_match = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    a_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE | re.DOTALL)

    if not q_match or not a_match:
        logger.debug(f"cross_doc: could not parse Q/A from: {raw[:120]}")
        return None

    question = _clean_question(q_match.group(1))
    answer   = _clean_answer(a_match.group(1))

    if not question.endswith("?") or len(answer) < 15:
        return None

    return {"user_input": question, "reference": answer,
            "question_type": "cross_doc",
            "source_docs": f"{source_a} + {source_b}"}


# ── Main entry point ───────────────────────────────────────────────────────

def generate_synthetic_testset(testset_size: int = DEFAULT_TESTSET_SIZE) -> pd.DataFrame:
    if TESTSET_PATH.exists():
        logger.info(f"⏭️  Loading cached testset from {TESTSET_PATH}")
        df = pd.read_csv(TESTSET_PATH)
        return df

    settings = get_settings()
    settings.validate()

    # Load child nodes — these are the actual sentence/table units the
    # retriever searches over, which is exactly what we want to generate
    # questions about (not the 4 raw markdown files truncated to 3000 chars).
    if not BM25_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No BM25 cache at {BM25_CACHE_PATH}. "
            "Run `python -m src.build_pipeline` first so child nodes exist."
        )
    bm25 = BM25Index.load()
    nodes = bm25.nodes
    logger.info(f"Loaded {len(nodes)} child nodes from BM25 cache for testset sampling.")

    # Build parent lookup for context expansion during question generation
    import pickle
    from src.build_pipeline import PARENT_NODES_CACHE
    with open(PARENT_NODES_CACHE, "rb") as f:
        parent_nodes = pickle.load(f)
    parent_lookup = {p.node_id: p for p in parent_nodes}

    # Get all unique source documents present in the corpus
    source_docs = list({nd.metadata.get("source_file", "") for nd in nodes
                        if nd.metadata.get("source_file", "")})
    logger.info(f"Source documents found: {source_docs}")

    base_url = settings.ollama_base_url
    model    = settings.llm_model

    # Target counts per type
    n_adversarial = min(6, len(ADVERSARIAL_QUESTIONS))
    n_cross       = int((testset_size - n_adversarial) * 0.50)
    n_single      = testset_size - n_adversarial - n_cross
    logger.info(f"Target: {n_single} single-hop, {n_cross} cross-doc, {n_adversarial} adversarial")

    results: list[dict] = []

    # ── Single-hop questions ──────────────────────────────────────────────
    logger.info(f"Generating {n_single} single-hop questions ...")
    attempts = 0
    while len([r for r in results if r["question_type"] == "single_hop"]) < n_single:
        if attempts > n_single * 4:
            logger.warning("Single-hop: too many failed attempts, moving on.")
            break
        attempts += 1
        node = _sample_nodes_by_doc(nodes, 1)[0]
        pair = _gen_single_hop(node, parent_lookup, base_url, model)
        if pair:
            results.append(pair)
            done = len([r for r in results if r["question_type"] == "single_hop"])
            logger.info(f"  single-hop {done}/{n_single}: {pair['user_input'][:80]}")

    # ── Cross-document questions ──────────────────────────────────────────
    if len(source_docs) >= 2:
        logger.info(f"Generating {n_cross} cross-document questions ...")
        attempts = 0
        while len([r for r in results if r["question_type"] == "cross_doc"]) < n_cross:
            if attempts > n_cross * 4:
                logger.warning("Cross-doc: too many failed attempts, moving on.")
                break
            attempts += 1
            doc_a, doc_b = random.sample(source_docs, 2)
            pool_a = _sample_nodes_by_doc(nodes, 1, exclude_doc=doc_b)
            pool_b = _sample_nodes_by_doc(nodes, 1, exclude_doc=doc_a)
            if not pool_a or not pool_b:
                continue
            pair = _gen_cross_doc(pool_a[0], pool_b[0], parent_lookup, base_url, model)
            if pair:
                results.append(pair)
                done = len([r for r in results if r["question_type"] == "cross_doc"])
                logger.info(f"  cross-doc {done}/{n_cross}: {pair['user_input'][:80]}")
    else:
        logger.warning("Only one source document found — skipping cross-doc questions.")
        n_single += n_cross

    # ── Adversarial questions ─────────────────────────────────────────────
    logger.info(f"Adding {n_adversarial} adversarial (unanswerable) questions ...")
    for q in ADVERSARIAL_QUESTIONS[:n_adversarial]:
        results.append({**q, "question_type": "adversarial", "source_docs": "N/A"})

    # ── Save ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv(TESTSET_PATH, index=False)

    counts = df["question_type"].value_counts().to_dict()
    logger.info(f"✅ Saved {len(df)} test pairs -> {TESTSET_PATH}")
    logger.info(f"   Distribution: {counts}")
    return df


if __name__ == "__main__":
    df = generate_synthetic_testset()
    print(f"\nGenerated {len(df)} rows. Distribution:\n")
    print(df["question_type"].value_counts())
    print("\nSample (one of each type):\n")
    for qtype in ["single_hop", "cross_doc", "adversarial"]:
        row = df[df["question_type"] == qtype].iloc[0]
        print(f"[{qtype}]")
        print(f"  Q: {row['user_input']}")
        print(f"  A: {row['reference'][:120]}")
        print()
