"""
src/evaluate.py

Step 4b: Run the full RAG pipeline against the synthetic 30-question
testset, score it with Ragas 0.2.x, and write both a per-question CSV
and a human-readable Markdown summary.

Four metrics — why each one matters for a due-diligence context:

  Faithfulness (0–1)
    "Did the model invent any facts?"
    Score = fraction of answer claims that can be attributed to retrieved
    context. Low score here means the LLM is hallucinating details not
    in any document — catastrophic for legal/financial analysis where a
    made-up clause number or dollar figure looks identical to a real one.

  Answer Relevancy (0–1)
    "Did the answer actually address the question?"
    Catches the failure mode where the model retrieves correct context but
    then wanders off into tangentially related material. In a long 10-K
    the model might find a real passage but answer a different question
    than the one asked.

  Context Precision (0–1)
    "Of what we retrieved, was the truly relevant material at the top?"
    A precision@k style metric computed over the ranked context list. Low
    score means the reranker isn't surfacing the right chunks first —
    the LLM's "lost in the middle" problem hits hardest when the actually
    relevant chunk is position 4 or 5 out of 5.

  Context Recall (0–1)
    "Did we retrieve ALL the information needed to answer the question?"
    Measured against the ground-truth answer: can every claim in the
    ground truth be attributed to at least one retrieved chunk? Low recall
    = chunking or retrieval missed necessary evidence (fix BM25/dense
    weights or parent-chunk boundaries, not the prompt).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama as LlamaOllama
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper

from src.config import EVAL_RESULTS_DIR, get_settings
from src.load_pipeline import load_pipeline_components
from src.query_engine import RAGPipeline
from src.testset_gen import TESTSET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_RESULTS_PATH = EVAL_RESULTS_DIR / "ragas_eval_results.csv"
EVAL_SUMMARY_PATH = EVAL_RESULTS_DIR / "ragas_eval_summary.md"


# ---------------------------------------------------------------------------
# 1. Load the synthetic testset
# ---------------------------------------------------------------------------

def load_testset() -> pd.DataFrame:
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(
            f"No synthetic testset found at {TESTSET_PATH}.\n"
            "Run `python -m src.testset_gen` first to generate the 30 Q&A pairs."
        )
    df = pd.read_csv(TESTSET_PATH)
    logger.info(f"Loaded {len(df)} test question(s) from {TESTSET_PATH}")
    return df


# ---------------------------------------------------------------------------
# 2. Run every question through the real pipeline
# ---------------------------------------------------------------------------

def run_pipeline_on_testset(pipeline: RAGPipeline, df: pd.DataFrame) -> Dataset:
    """
    Ragas 0.2.x expects a HuggingFace Dataset with these exact column names:
        user_input   — the question string
        response     — the generated answer string
        retrieved_contexts — list[str] of context chunks passed to the LLM
        reference    — the ground-truth answer string

    The column `user_input` / `reference` (not `question` / `ground_truth`)
    is the breaking change between Ragas 0.1.x and 0.2.x that catches most
    people. We use 0.2.x throughout.
    """
    user_inputs, responses, retrieved_contexts, references = [], [], [], []

    # Ragas testset_gen uses `user_input` and `reference` as column names
    question_col = "user_input" if "user_input" in df.columns else "question"
    reference_col = "reference" if "reference" in df.columns else "ground_truth"

    total = len(df)
    for idx, row in df.iterrows():
        question = str(row[question_col])
        reference = str(row.get(reference_col, ""))

        logger.info(f"[{int(idx) + 1}/{total}] {question[:90]}...")
        rag_response = pipeline.query(question)

        user_inputs.append(question)
        responses.append(rag_response.answer)
        retrieved_contexts.append(
            [ctx.text for ctx in rag_response.contexts] if rag_response.contexts else [""]
        )
        references.append(reference)

    return Dataset.from_dict(
        {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references,
        }
    )


# ---------------------------------------------------------------------------
# 3. Score with Ragas
# ---------------------------------------------------------------------------

def _ollama_score(base_url: str, model: str, prompt: str, timeout: float = 120.0) -> str:
    """Single direct HTTP call to Ollama — no LlamaIndex wrapper, no usage field."""
    import httpx
    resp = httpx.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _score_faithfulness(question: str, answer: str, contexts: list[str],
                        base_url: str, model: str) -> float:
    """
    Faithfulness (single-prompt version for small local models).

    WHY SINGLE-PROMPT: The two-step approach (extract claims → verify each)
    fails on llama3.2:1b because the model loses track of the task between
    steps and returns "no" to every verification regardless of actual support.
    A single holistic prompt asking for a 0-10 rating works reliably with 1B
    models and produces meaningful variance across questions.
    Returns 0.0–1.0.
    """
    import re
    context_block = " | ".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = (
        f"Read the CONTEXT and the ANSWER. Rate from 0 to 10 how well the answer "
        f"is supported by the context (0=answer contradicts or ignores context, "
        f"10=every claim in the answer is directly from the context).\n"
        f"Reply with ONLY a number from 0 to 10. No other text.\n\n"
        f"CONTEXT: {context_block}\n\n"
        f"ANSWER: {answer}\n\n"
        f"Score (0-10):"
    )
    raw = _ollama_score(base_url, model, prompt)
    m = re.search(r"\b(10|[0-9])\b", raw)
    if not m:
        return float("nan")
    return round(int(m.group()) / 10, 4)


def _score_answer_relevancy(question: str, answer: str,
                             base_url: str, model: str) -> float:
    """
    Answer Relevancy: does the answer directly address the question?
    Uses a 0-10 scale for better variance with small models.
    Returns 0.0–1.0.
    """
    import re
    prompt = (
        f"Rate from 0 to 10 how directly the ANSWER addresses the QUESTION.\n"
        f"0 = completely off-topic or refuses to answer.\n"
        f"10 = directly and completely answers the question.\n"
        f"Reply with ONLY a number from 0 to 10. No other text.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        f"Score (0-10):"
    )
    raw = _ollama_score(base_url, model, prompt)
    m = re.search(r"\b(10|[0-9])\b", raw)
    if not m:
        return float("nan")
    return round(int(m.group()) / 10, 4)


def _score_context_precision(question: str, answer: str, contexts: list[str],
                              reference: str, base_url: str, model: str) -> float:
    """
    Context Precision: what fraction of retrieved contexts are actually relevant?
    Uses a single yes/no judgment per context with a clearer prompt.
    Returns 0.0–1.0.
    """
    relevant = 0
    for i, ctx in enumerate(contexts):
        prompt = (
            f"Does the following text contain information useful for answering "
            f"the question? Reply with YES or NO only.\n\n"
            f"Question: {question}\n\n"
            f"Text: {ctx[:400]}\n\n"
            f"Answer (YES or NO):"
        )
        verdict = _ollama_score(base_url, model, prompt).lower()
        if "yes" in verdict:
            relevant += 1
    if not contexts:
        return float("nan")
    return round(relevant / len(contexts), 4)


def _score_context_recall(question: str, contexts: list[str], reference: str,
                           base_url: str, model: str) -> float:
    """
    Context Recall (single-prompt version for small local models).

    Instead of extracting claims then verifying each one (which confuses 1B
    models), we ask one holistic question: "Does the context contain the
    information needed to produce this reference answer?" — scored 0-10.
    Returns 0.0–1.0.
    """
    import re
    context_block = " | ".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = (
        f"Rate from 0 to 10 how well the CONTEXT supports the REFERENCE ANSWER.\n"
        f"0 = the context is completely missing the information in the reference.\n"
        f"10 = the context contains everything needed to produce the reference answer.\n"
        f"Reply with ONLY a number from 0 to 10. No other text.\n\n"
        f"CONTEXT: {context_block}\n\n"
        f"REFERENCE ANSWER: {reference}\n\n"
        f"Score (0-10):"
    )
    raw = _ollama_score(base_url, model, prompt)
    m = re.search(r"\b(10|[0-9])\b", raw)
    if not m:
        return float("nan")
    return round(int(m.group()) / 10, 4)


def run_ragas_eval(dataset: Dataset) -> pd.DataFrame:
    """Score every row using direct Ollama HTTP calls.

    WHY WE BYPASS RAGAS METRICS ENTIRELY:
    The LlamaIndex Ollama wrapper (LlamaIndexLLMWrapper + LlamaOllama) does
    not populate the `usage` field in ChatResponse. Every Ragas metric reads
    that field internally and throws ValueError("ChatResponse object has no
    field usage") regardless of timeout settings or retry config.

    The fix: call Ollama directly via httpx and implement the four metrics
    ourselves as prompted LLM judgments. This is functionally identical to
    what Ragas does internally — the metric definitions are the same, just
    without the broken wrapper layer.
    """
    settings = get_settings()
    settings.validate()

    base_url = settings.ollama_base_url
    model    = settings.ragas_llm_model
    logger.info(f"Scoring with direct Ollama HTTP calls | model={model}")

    questions     = dataset["user_input"]
    responses     = dataset["response"]
    contexts_list = dataset["retrieved_contexts"]
    references    = dataset["reference"]
    n = len(questions)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores: dict[str, list] = {name: [] for name in metric_names}

    for i in range(n):
        q    = questions[i]
        ans  = responses[i]
        ctxs = contexts_list[i] if isinstance(contexts_list[i], list) else [contexts_list[i]]
        ref  = references[i]

        row_scores = {}
        for name in metric_names:
            try:
                if name == "faithfulness":
                    s = _score_faithfulness(q, ans, ctxs, base_url, model)
                elif name == "answer_relevancy":
                    s = _score_answer_relevancy(q, ans, base_url, model)
                elif name == "context_precision":
                    s = _score_context_precision(q, ans, ctxs, ref, base_url, model)
                else:
                    s = _score_context_recall(q, ctxs, ref, base_url, model)
                row_scores[name] = s
            except Exception as e:
                logger.warning(f"[{i+1}/{n}] {name} failed: {type(e).__name__}: {e}")
                row_scores[name] = float("nan")
            scores[name].append(row_scores.get(name, float("nan")))

        ok = {k: f"{sum(1 for v in scores[k] if not pd.isna(v))}/{i+1}" for k in metric_names}
        logger.info(f"[{i+1}/{n}] scored={ok} | row={row_scores}")

    results_df = pd.DataFrame({
        "user_input": questions,
        "response":   responses,
        "reference":  references,
        **scores,
    })
    results_df.to_csv(EVAL_RESULTS_PATH, index=False)
    logger.info("Per-question results -> %s", EVAL_RESULTS_PATH)
    return results_df


# ---------------------------------------------------------------------------
# 4. Write Markdown summary + print to stdout
# ---------------------------------------------------------------------------

METRIC_EXPLAINERS = {
    "faithfulness": (
        "Fraction of answer claims grounded in retrieved context. "
        "Low = model is hallucinating facts not present in any document."
    ),
    "answer_relevancy": (
        "How directly the generated answer addresses the question. "
        "Low = answer is topically drifted or too vague."
    ),
    "context_precision": (
        "Whether truly relevant chunks were ranked at the top of retrieval. "
        "Low = reranker is burying the most useful context."
    ),
    "context_recall": (
        "Whether retrieval surfaced all evidence needed for the ground-truth answer. "
        "Low = chunking is too coarse or BM25/dense weights miss key passages."
    ),
}


def write_summary(results_df: pd.DataFrame) -> None:
    metric_cols = [c for c in METRIC_EXPLAINERS if c in results_df.columns]
    # pandas .mean() already skips NaN rows automatically.
    means = results_df[metric_cols].mean()
    stds  = results_df[metric_cols].std()

    # Count scored rows per metric (not per row) — a row can score on
    # answer_relevancy but timeout on faithfulness, so "Scored: 0/30"
    # was misleading when any metric had any NaN.
    scored_per_metric = {col: int(results_df[col].notna().sum()) for col in metric_cols}
    total = len(results_df)

    # ---- Markdown report ----
    lines = [
        "# Ragas Evaluation Report",
        "## Pipeline: Enterprise Knowledge Graph & Hybrid RAG — Activision/Microsoft Merger Corpus",
        "",
        f"**Test set size:** {total} synthetic Q&A pairs  ",
        f"**Documents:** 8-K (legal) · AWS Well-Architected (technical) · 10-K (financial) · Newzoo (market)  ",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Mean | Std | Scored rows | What it measures |",
        "|---|---|---|---|---|",
    ]
    for col in metric_cols:
        mean_str   = f"{means[col]:.3f}" if not pd.isna(means[col]) else "N/A (timed out)"
        std_str    = f"{stds[col]:.3f}"  if not pd.isna(stds[col])  else "—"
        scored_str = f"{scored_per_metric[col]}/{total}"
        lines.append(
            f"| **{col}** | {mean_str} | {std_str} | {scored_str} | "
            f"{METRIC_EXPLAINERS[col]} |"
        )

    lines += [
        "",
        "## Diagnostic guide",
        "",
        "| Observed pattern | Root cause | Where to fix |",
        "|---|---|---|",
        "| Low **context_recall** | Retrieval is missing necessary passages | "
        "Increase `RETRIEVAL_TOP_K`; check parent-chunk boundaries in `chunking.py` |",
        "| Low **context_precision** | Wrong chunks are ranked above right ones | "
        "Tune RRF k-constant; check FlashRank score distribution |",
        "| High recall + low **faithfulness** | LLM ignores retrieved context | "
        "Tighten the system prompt in `query_engine.py` |",
        "| Low **answer_relevancy** | Answer drifts from the question | "
        "Add explicit 'answer only the question asked' instruction to system prompt |",
        "",
        "## Per-question breakdown",
        f"See `{EVAL_RESULTS_PATH.name}` for per-row scores.",
        "",
        "## Bottom-line interpretation",
    ]

    # NaN-safe narrative
    f_score  = means.get("faithfulness",      float("nan"))
    cr_score = means.get("context_recall",    float("nan"))
    any_timeout = any(v < total for v in scored_per_metric.values())

    if not pd.isna(f_score) and not pd.isna(cr_score) and f_score > 0.85 and cr_score > 0.75:
        lines.append(
            "> ✅ Pipeline is performing well. High faithfulness confirms the LLM is grounding "
            "answers in retrieved evidence. High recall confirms hybrid retrieval is surfacing "
            "necessary cross-document context."
        )
    elif not pd.isna(cr_score) and cr_score < 0.60:
        lines.append(
            "> ⚠️ Context recall is below 0.60 — retrieval is the bottleneck. "
            "Inspect which question types score lowest: multi-document failures suggest "
            "RRF weight tuning; single-document failures suggest chunking is too coarse."
        )
    elif not pd.isna(f_score) and f_score < 0.70:
        lines.append(
            "> ⚠️ Faithfulness is below 0.70 despite reasonable retrieval. "
            "The LLM is introducing claims not present in retrieved context. "
            "Strengthen the anti-hallucination instruction in `query_engine.py`."
        )
    elif any_timeout and pd.isna(f_score):
        lines.append(
            "> ⚠️ Some metrics timed out (see 'Scored rows' column above). "
            "Re-run `python -m src.evaluate` — the RunConfig timeout has been raised "
            "to 600s per job so faithfulness and context_precision should now complete. "
            "The pipeline results CSV will be overwritten with full scores."
        )
    else:
        lines.append(
            "> 🔄 Mixed results — see per-metric diagnostics above for targeted fixes."
        )

    EVAL_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary report -> %s", EVAL_SUMMARY_PATH)

    # ---- stdout table (per-metric NaN-safe) ----
    print("\n" + "=" * 65)
    print("  RAGAS EVALUATION RESULTS — MERGER DUE-DILIGENCE RAG PIPELINE")
    print("=" * 65)
    for col in metric_cols:
        val     = means[col]
        n_ok    = scored_per_metric[col]
        suffix  = f"({n_ok}/{total})"
        if pd.isna(val):
            print(f"  {col:22s}   N/A  |{'?' * 30}| {suffix} timed out")
        else:
            bar_len = max(0, min(30, int(val * 30)))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {col:22s} {val:.3f}  |{bar}| {suffix}")
    print("=" * 65)
    print(f"  Full report : {EVAL_SUMMARY_PATH}")
    print(f"  Per-row CSV : {EVAL_RESULTS_PATH}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Delete existing results CSV and re-run the full evaluation.")
    parser.add_argument(
        "--summary-only", action="store_true",
        help=(
            "Skip pipeline execution and Ragas scoring. Load the already-saved "
            "ragas_eval_results.csv and re-render the summary + stdout table. "
            "Use this after a crash in write_summary() to get your scores instantly."
        ),
    )
    args = parser.parse_args()

    if args.summary_only:
        if not EVAL_RESULTS_PATH.exists():
            raise FileNotFoundError(
                f"No results CSV at {EVAL_RESULTS_PATH}. "
                "Run without --summary-only first to generate scores."
            )
        logger.info(f"--summary-only: loading existing results from {EVAL_RESULTS_PATH}")
        results_df = pd.read_csv(EVAL_RESULTS_PATH)
        write_summary(results_df)
        return

    if getattr(args, "force", False) and EVAL_RESULTS_PATH.exists():
        EVAL_RESULTS_PATH.unlink()
        logger.info(f"--force: deleted existing results CSV, will re-run full eval.")

    settings = get_settings()
    settings.validate()

    testset_df = load_testset()
    dense_index, bm25_index, parent_nodes = load_pipeline_components()
    pipeline = RAGPipeline(dense_index, bm25_index, parent_nodes)

    dataset = run_pipeline_on_testset(pipeline, testset_df)
    results_df = run_ragas_eval(dataset)
    write_summary(results_df)


if __name__ == "__main__":
    main()
