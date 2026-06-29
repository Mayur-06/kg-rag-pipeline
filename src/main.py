"""
src/main.py

Interactive CLI for querying the merger due-diligence RAG pipeline.
Run `python -m src.build_pipeline` first to populate the indexes.

Example session:
  > Which legal clause governs the cloud infrastructure mentioned in the technical SOP?
  > What are the termination conditions in the vendor contract?
  > exit
"""
from __future__ import annotations

from src.load_pipeline import load_pipeline_components
from src.query_engine import RAGPipeline


def print_answer(response):
    print("\n" + "=" * 80)
    print("ANSWER:\n")
    print(response.answer)
    print("\n" + "-" * 80)
    print(f"Sources used ({len(response.contexts)}):")
    for i, ctx in enumerate(response.contexts, start=1):
        preview = ctx.text[:120].replace("\n", " ")
        print(f"  [{i}] {ctx.source_file} (relevance={ctx.rrf_score:.3f}) — {preview}...")
    print("=" * 80 + "\n")


def main():
    print("Loading dense index (Pinecone), sparse index (BM25), and parent nodes ...")
    dense_index, bm25_index, parent_nodes = load_pipeline_components()
    pipeline = RAGPipeline(dense_index, bm25_index, parent_nodes)
    print("✅ Ready. Ask a question about your merger documents (or type 'exit').\n")

    while True:
        try:
            query_str = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query_str:
            continue
        if query_str.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        response = pipeline.query(query_str)
        print_answer(response)


if __name__ == "__main__":
    main()
