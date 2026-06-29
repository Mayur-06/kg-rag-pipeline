"""
src/query_engine.py  —  Local LLM edition

Generation backend: Ollama (llama3.2, runs locally, free)
  Install Ollama: https://ollama.com/download  (Windows/Mac/Linux)
  Pull the model: ollama pull llama3.2        (~2 GB download, cached)
  Start server:   ollama serve                (or it auto-starts on install)

Fallback: if LLM_BACKEND=openai is set in .env, uses gpt-4o-mini instead
  (requires a valid OPENAI_API_KEY with credits).

The RAGPipeline class is unchanged — same interface used by main.py and
evaluate.py. Only the _build_llm() factory switches between backends.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import get_settings
from src.retrieval import HybridRetriever, RetrievedContext
from src.rerank import rerank_and_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a due-diligence research assistant helping a consulting team \
(Bain) analyze documents from a corporate merger: SOPs, technical documentation, legal \
contracts, and market research.

Rules:
- Answer ONLY using the provided context. If the context does not contain the answer, \
say so explicitly — do not guess or use outside knowledge.
- When you state a fact, cite which source document it came from using the [source_file] \
tag provided with each context block.
- If different context blocks conflict (e.g. two contract versions), surface the conflict \
rather than silently picking one.
- Be precise about clause numbers, product names, and figures — these are due-diligence \
documents where exact references matter.
- Answer in 1-2 sentences, directly addressing the question and nothing else.
"""


@dataclass
class RAGResponse:
    answer: str
    contexts: list[RetrievedContext]


MAX_CONTEXT_CHARS = 1200


def _truncate_text_for_llm(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    if "\n" in truncated:
        truncated = truncated.rsplit("\n", 1)[0]
    return truncated + "\n\n[...TRUNCATED FOR LOCAL INFERENCE SPEED...]"


def build_context_block(contexts: list[RetrievedContext]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        snippet = _truncate_text_for_llm(ctx.text)
        blocks.append(f"[Context {i} | source_file: {ctx.source_file}]\n{snippet}")
    return "\n\n---\n\n".join(blocks)


def _build_llm(settings):
    """Factory: returns whichever LLM backend is configured."""
    if settings.llm_backend == "openai":
        # ── OpenAI fallback ────────────────────────────────────────────
        if not settings.openai_api_key or settings.openai_api_key.endswith("..."):
            raise EnvironmentError(
                "LLM_BACKEND=openai but OPENAI_API_KEY is missing/placeholder. "
                "Either add credits and set the key, or switch to LLM_BACKEND=ollama."
            )
        from openai import OpenAI as _OpenAI
        logger.info(f"LLM backend: OpenAI ({settings.llm_model})")
        return ("openai", _OpenAI(api_key=settings.openai_api_key))
    else:
        # ── Ollama (default) ───────────────────────────────────────────
        # We use httpx directly rather than the ollama Python package so
        # there's one fewer dependency and the streaming/error behaviour
        # is transparent.
        import httpx
        # Quick connectivity check — fail fast with a helpful message if
        # Ollama isn't running rather than timing out silently mid-query.
        try:
            timeout = httpx.Timeout(settings.ollama_connect_timeout, connect=settings.ollama_connect_timeout)
            r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout)
            r.raise_for_status()
        except Exception:
            raise RuntimeError(
                f"Cannot reach Ollama at {settings.ollama_base_url}.\n"
                "  1. Install Ollama: https://ollama.com/download\n"
                f"  2. Pull the model: ollama pull {settings.llm_model}\n"
                "  3. Start the server: ollama serve\n"
                "  Then re-run your query."
            )
        logger.info(f"LLM backend: Ollama ({settings.llm_model} @ {settings.ollama_base_url})")
        return ("ollama", settings)  # pass settings through; query() uses httpx directly


def _call_llm(backend_type: str, backend, system_prompt: str, user_message: str, settings) -> str:
    """Unified call interface for both backends."""
    if backend_type == "openai":
        completion = backend.chat.completions.create(
            model=settings.llm_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return completion.choices[0].message.content
    else:
        # Ollama REST API via /api/generate is more reliable for local llama3.2.
        import httpx
        payload = {
            "model": settings.llm_model,
            "prompt": f"{system_prompt}\n\n{user_message}",
            "options": {
                "temperature": 0.0,
                "max_tokens": 512,
            },
            "stream": False,
        }
        timeout = httpx.Timeout(None, connect=settings.ollama_connect_timeout)
        r = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        body = r.json()
        return body.get("response", "").strip()

class RAGPipeline:
    def __init__(self, dense_index, bm25_index, parent_nodes):
        self.retriever = HybridRetriever(dense_index, bm25_index, parent_nodes)
        self.settings  = get_settings()
        backend_type, backend = _build_llm(self.settings)
        self._backend_type = backend_type
        self._backend      = backend

    def query(self, query_str: str) -> RAGResponse:
        candidates = self.retriever.retrieve(query_str, top_k=self.settings.retrieval_top_k)
        if not candidates:
            return RAGResponse(
                answer="I couldn't find any relevant context in the indexed documents for this question.",
                contexts=[],
            )

        final_contexts = rerank_and_compress(query_str, candidates, top_n=self.settings.rerank_top_n)
        context_block  = build_context_block(final_contexts)
        user_message   = f"Context:\n\n{context_block}\n\n---\n\nQuestion: {query_str}"

        answer = _call_llm(
            self._backend_type, self._backend,
            SYSTEM_PROMPT, user_message, self.settings,
        )
        return RAGResponse(answer=answer, contexts=final_contexts)
