"""
setup.py — makes `src` importable as an installed package and lets you run
the pipeline modules with clean `python -m src.build_pipeline` invocations
without fiddling with PYTHONPATH.

Install in editable mode for development:
    pip install -e .
"""
from setuptools import find_packages, setup

setup(
    name="merger-rag-pipeline",
    version="0.1.0",
    description=(
        "Enterprise Knowledge Graph & Hybrid RAG Pipeline for Corporate Mergers. "
        "Hierarchical chunking + Pinecone (dense) + BM25 (sparse) + RRF fusion + "
        "FlashRank local rerank + Ragas evaluation — built over the Activision/Microsoft M&A corpus. "
        "Embedding: BAAI/bge-small-en-v1.5 (local). Generation: Ollama llama3.2 (local). "
        "Only external API: Pinecone free tier."
    ),
    author="Your Name",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "llama-index==0.11.23",
        "llama-index-core==0.11.23",
        "llama-index-embeddings-huggingface==0.3.1",
        "llama-index-llms-ollama==0.3.4",
        "llama-index-vector-stores-pinecone==0.2.1",
        "llama-index-readers-llama-parse==0.3.0",
        "llama-parse==0.5.12",
        "pinecone-client==5.0.1",
        "rank-bm25==0.2.2",
        "sentence-transformers==3.2.1",
        "flashrank==0.2.9",
        "torch==2.4.1 ",
        "ragas==0.2.3",
        "langchain==1.3.10",
        "langchain-community==0.3.31",
        "datasets==3.0.1",
        "python-dotenv==1.0.1",
        "pydantic==2.9.2",
        "tqdm==4.66.5",
        "nltk==3.9.1",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "httpx==0.27.2",
        "tenacity==8.5.0",
        "fastapi==0.115.5",
        "uvicorn[standard]==0.32.0",
    ],
    entry_points={
        "console_scripts": [
            "merger-build=src.build_pipeline:main",
            "merger-query=src.main:main",
            "merger-eval=src.evaluate:main",
            "merger-viz=src.visualize_results:main",
            "merger-api=src.api:start",
        ]
    },
)