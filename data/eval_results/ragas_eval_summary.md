# Ragas Evaluation Report
## Pipeline: Enterprise Knowledge Graph & Hybrid RAG — Activision/Microsoft Merger Corpus

**Test set size:** 30 synthetic Q&A pairs  
**Documents:** 8-K (legal) · AWS Well-Architected (technical) · 10-K (financial) · Newzoo (market)  

## Aggregate scores

| Metric | Mean | Std | Scored rows | What it measures |
|---|---|---|---|---|
| **faithfulness** | 0.223 | 0.176 | 30/30 | Fraction of answer claims grounded in retrieved context. Low = model is hallucinating facts not present in any document. |
| **answer_relevancy** | 0.230 | 0.170 | 30/30 | How directly the generated answer addresses the question. Low = answer is topically drifted or too vague. |
| **context_precision** | 0.100 | 0.188 | 30/30 | Whether truly relevant chunks were ranked at the top of retrieval. Low = reranker is burying the most useful context. |
| **context_recall** | 0.213 | 0.101 | 30/30 | Whether retrieval surfaced all evidence needed for the ground-truth answer. Low = chunking is too coarse or BM25/dense weights miss key passages. |

## Diagnostic guide

| Observed pattern | Root cause | Where to fix |
|---|---|---|
| Low **context_recall** | Retrieval is missing necessary passages | Increase `RETRIEVAL_TOP_K`; check parent-chunk boundaries in `chunking.py` |
| Low **context_precision** | Wrong chunks are ranked above right ones | Tune RRF k-constant; check FlashRank score distribution |
| High recall + low **faithfulness** | LLM ignores retrieved context | Tighten the system prompt in `query_engine.py` |
| Low **answer_relevancy** | Answer drifts from the question | Add explicit 'answer only the question asked' instruction to system prompt |

## Per-question breakdown
See `ragas_eval_results.csv` for per-row scores.

## Bottom-line interpretation
> ⚠️ Context recall is below 0.60 — retrieval is the bottleneck. Inspect which question types score lowest: multi-document failures suggest RRF weight tuning; single-document failures suggest chunking is too coarse.