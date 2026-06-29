"""
src/api.py

FastAPI REST API for the Enterprise Knowledge Graph & Hybrid RAG Pipeline.

Endpoints
─────────
  GET  /health                   Liveness probe — no deps, always fast
  GET  /status                   Readiness probe — checks indexes are built
  POST /query                    Run a question through the full RAG pipeline
  POST /build                    Trigger pipeline build (background job)
  GET  /build/status/{job_id}    Poll build job progress
  POST /evaluate                 Trigger Ragas evaluation (background job)
  GET  /evaluate/status/{job_id} Poll evaluation job progress

Design decisions
────────────────
  - RAGPipeline is loaded ONCE at startup via FastAPI lifespan and stored in
    app.state. Loading takes ~30-60s (Pinecone connect + BM25 deserialize +
    BGE model load). Re-loading per request would be unusable.

  - Build and Evaluate run in a ThreadPoolExecutor because both are blocking
    CPU/IO-heavy operations (embedding, BM25 construction, Ollama calls). The
    event loop stays responsive for /query and /status during long builds.

  - Job state is in-memory (dict). Sufficient for a single-process deployment.
    For multi-replica AWS deployments, swap job_store for Redis/DynamoDB.

  - /query itself is synchronous but fast enough (< 5s on CPU) that running
    it in a thread pool via run_in_executor is fine. It's kept sync here to
    avoid masking the real async story (build/evaluate ARE long-running).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.build_pipeline import PARENT_NODES_CACHE
from src.config import get_settings
from src.sparse_index import BM25_CACHE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"


@dataclass
class Job:
    job_id:    str
    kind:      str   # "build" | "evaluate"
    status:    JobStatus = JobStatus.PENDING
    message:   str       = ""
    result:    Any       = None
    started_at: float    = field(default_factory=time.time)
    finished_at: float | None = None


# In-memory store: {job_id: Job}
_job_store: dict[str, Job] = {}
_executor  = ThreadPoolExecutor(max_workers=2)


def _new_job(kind: str) -> Job:
    job = Job(job_id=str(uuid.uuid4()), kind=kind)
    _job_store[job.job_id] = job
    return job


def _get_job(job_id: str, kind: str) -> Job:
    job = _job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job.kind != kind:
        raise HTTPException(status_code=400, detail=f"Job {job_id!r} is a {job.kind!r} job, not {kind!r}.")
    return job


# ---------------------------------------------------------------------------
# Pipeline state — loaded once at startup
# ---------------------------------------------------------------------------

_pipeline = None   # RAGPipeline | None
_pipeline_load_error: str | None = None


def _try_load_pipeline() -> None:
    """Attempt to load the pipeline at startup. Failures are non-fatal —
    the API still starts so /health and /build are reachable."""
    global _pipeline, _pipeline_load_error
    try:
        logger.info("Loading pipeline components (Pinecone + BM25 + parent nodes) ...")
        from src.load_pipeline import load_pipeline_components
        from src.query_engine import RAGPipeline
        dense_index, bm25_index, parent_nodes = load_pipeline_components()
        _pipeline = RAGPipeline(dense_index, bm25_index, parent_nodes)
        logger.info("✅ Pipeline ready.")
    except FileNotFoundError as e:
        _pipeline_load_error = (
            f"Pipeline indexes not built yet: {e}. "
            "Call POST /build to build them."
        )
        logger.warning(_pipeline_load_error)
    except Exception as e:
        _pipeline_load_error = f"Pipeline load failed: {type(e).__name__}: {e}"
        logger.error(_pipeline_load_error)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load pipeline in the thread pool so the event loop isn't blocked
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _try_load_pipeline)
    yield
    # Shutdown: nothing to clean up (Pinecone is stateless HTTP)
    _executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise RAG Pipeline API",
    description=(
        "Hybrid RAG pipeline for M&A due-diligence document retrieval. "
        "Hierarchical chunking + Pinecone (dense) + BM25 (sparse) + RRF + FlashRank rerank."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to ask the RAG pipeline.")
    top_k:    int = Field(20, ge=1, le=100, description="Number of candidates to retrieve before reranking.")


class SourceContext(BaseModel):
    source_file: str
    rrf_score:   float
    preview:     str   # first 200 chars of the retrieved + compressed context


class QueryResponse(BaseModel):
    answer:      str
    sources:     list[SourceContext]
    duration_ms: float


class BuildRequest(BaseModel):
    force:         bool = Field(False, description="Re-run ALL stages (equivalent to --force).")
    force_ingest:  bool = Field(False, description="Re-parse PDFs even if .md files exist.")
    force_chunk:   bool = Field(False, description="Re-chunk even if parent_nodes.pkl exists.")
    force_embed:   bool = Field(False, description="Re-embed and re-upsert to Pinecone.")
    force_bm25:    bool = Field(False, description="Rebuild BM25 index even if .pkl exists.")


class EvaluateRequest(BaseModel):
    force: bool = Field(False, description="Delete existing results CSV and re-run full evaluation.")


class JobResponse(BaseModel):
    job_id:  str
    status:  JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id:      str
    kind:        str
    status:      JobStatus
    message:     str
    result:      Any = None
    started_at:  float
    finished_at: float | None = None
    elapsed_s:   float | None = None


class StatusResponse(BaseModel):
    pipeline_ready:     bool
    parent_nodes_cache: bool
    bm25_cache:         bool
    pipeline_error:     str | None = None


# ---------------------------------------------------------------------------
# Background job runners
# ---------------------------------------------------------------------------

def _run_build(job: Job, req: BuildRequest) -> None:
    """Blocking function — called from the thread pool."""
    global _pipeline, _pipeline_load_error
    job.status  = JobStatus.RUNNING
    job.message = "Build in progress ..."
    try:
        import argparse
        import asyncio

        # Build pipeline stages
        from src.chunking import build_hierarchical_nodes, load_markdown_documents
        from src.build_pipeline import PARENT_NODES_CACHE, CHILD_NODES_CACHE, _load_cached_nodes
        from src.dense_index import build_dense_index
        from src.ingest import ingest_all_pdfs
        from src.sparse_index import BM25_CACHE_PATH, build_bm25_index
        from src.config import STORAGE_DIR
        import pickle

        force_all = req.force

        # Stage 1: Ingest
        job.message = "Stage 1/4: PDF ingestion ..."
        asyncio.run(ingest_all_pdfs(force_ingest=(force_all or req.force_ingest)))

        # Stage 1b: Chunking
        job.message = "Stage 2/4: Hierarchical chunking ..."
        chunk_cache_exists = PARENT_NODES_CACHE.exists() and CHILD_NODES_CACHE.exists()
        if not (force_all or req.force_chunk) and chunk_cache_exists:
            parent_nodes, child_nodes = _load_cached_nodes()
        else:
            documents = load_markdown_documents()
            parent_nodes, child_nodes = build_hierarchical_nodes(documents)
            with open(PARENT_NODES_CACHE, "wb") as f:
                pickle.dump(parent_nodes, f)
            with open(CHILD_NODES_CACHE, "wb") as f:
                pickle.dump(child_nodes, f)

        # Stage 2a: Dense index
        job.message = "Stage 3/4: Embedding → Pinecone ..."
        build_dense_index(child_nodes, force=(force_all or req.force_embed))

        # Stage 2b: BM25
        job.message = "Stage 4/4: Building BM25 index ..."
        if not (force_all or req.force_bm25) and BM25_CACHE_PATH.exists():
            logger.info("⏭️  BM25 cache exists, skipping rebuild.")
        else:
            build_bm25_index(child_nodes)

        # Reload the pipeline in-process so /query works immediately
        job.message = "Reloading pipeline into memory ..."
        from src.load_pipeline import load_pipeline_components
        from src.query_engine import RAGPipeline
        dense_index, bm25_index, parent_nodes = load_pipeline_components()
        _pipeline = RAGPipeline(dense_index, bm25_index, parent_nodes)
        _pipeline_load_error = None

        job.status       = JobStatus.DONE
        job.message      = "Pipeline build complete. /query is now ready."
        job.finished_at  = time.time()
        logger.info("✅ Build job %s complete.", job.job_id)

    except Exception as e:
        job.status      = JobStatus.FAILED
        job.message     = f"{type(e).__name__}: {e}"
        job.finished_at = time.time()
        logger.error("Build job %s failed:\n%s", job.job_id, traceback.format_exc())


def _run_evaluate(job: Job, req: EvaluateRequest) -> None:
    """Blocking function — called from the thread pool."""
    job.status  = JobStatus.RUNNING
    job.message = "Evaluation in progress (this takes ~3 min/row on CPU) ..."
    try:
        from src.evaluate import (
            EVAL_RESULTS_PATH,
            load_testset,
            run_pipeline_on_testset,
            run_ragas_eval,
            write_summary,
        )

        if req.force and EVAL_RESULTS_PATH.exists():
            EVAL_RESULTS_PATH.unlink()
            logger.info("--force: deleted existing results CSV.")

        if _pipeline is None:
            raise RuntimeError(
                "Pipeline not loaded. Call POST /build first to build the indexes, "
                "then retry POST /evaluate."
            )

        settings = get_settings()
        settings.validate()

        testset_df   = load_testset()
        dataset      = run_pipeline_on_testset(_pipeline, testset_df)
        results_df   = run_ragas_eval(dataset)
        write_summary(results_df)

        # Build a compact result dict for the status endpoint
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        import math
        means = {}
        for col in metric_cols:
            if col in results_df.columns:
                val = float(results_df[col].mean())
                means[col] = None if math.isnan(val) else round(val, 4)

        job.status      = JobStatus.DONE
        job.message     = f"Evaluation complete. Scored {len(results_df)} rows."
        job.result      = {"scores": means, "n_rows": len(results_df)}
        job.finished_at = time.time()
        logger.info("✅ Evaluate job %s complete.", job.job_id)

    except Exception as e:
        job.status      = JobStatus.FAILED
        job.message     = f"{type(e).__name__}: {e}"
        job.finished_at = time.time()
        logger.error("Evaluate job %s failed:\n%s", job.job_id, traceback.format_exc())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe. Returns 200 immediately — no dependency checks."""
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse, tags=["ops"])
def status():
    """Readiness probe. Reports whether the pipeline indexes are built and
    the in-memory pipeline is ready to serve /query requests."""
    return StatusResponse(
        pipeline_ready     = _pipeline is not None,
        parent_nodes_cache = PARENT_NODES_CACHE.exists(),
        bm25_cache         = BM25_CACHE_PATH.exists(),
        pipeline_error     = _pipeline_load_error,
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    """Run a question through the full RAG pipeline.

    Retrieves relevant context via hybrid search (Pinecone + BM25 + RRF),
    reranks with FlashRank, and generates an answer with Ollama llama3.2.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                _pipeline_load_error
                or "Pipeline not ready. Call POST /build first, then wait for it to complete."
            ),
        )

    t0 = time.time()
    settings = get_settings()

    # Override top_k per request if caller wants fewer/more candidates
    original_top_k = settings.retrieval_top_k
    try:
        response = _pipeline.query(req.question)
    except Exception as e:
        logger.error("Query failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {e}")

    duration_ms = round((time.time() - t0) * 1000, 1)

    sources = [
        SourceContext(
            source_file = ctx.source_file,
            rrf_score   = round(ctx.rrf_score, 4),
            preview     = ctx.text[:200].replace("\n", " "),
        )
        for ctx in response.contexts
    ]

    return QueryResponse(
        answer      = response.answer,
        sources     = sources,
        duration_ms = duration_ms,
    )


@app.post("/build", response_model=JobResponse, status_code=202, tags=["pipeline"])
def build(req: BuildRequest):
    """Trigger the pipeline build in the background.

    Runs all four stages: PDF ingestion → hierarchical chunking →
    Pinecone embed/upsert → BM25 index. Returns a job_id to poll.

    Use the force_* flags to selectively re-run specific stages.
    Caches guard every stage — re-running without force flags is safe.
    """
    job = _new_job("build")
    _executor.submit(_run_build, job, req)
    return JobResponse(
        job_id  = job.job_id,
        status  = job.status,
        message = "Build job queued. Poll GET /build/status/{job_id} for progress.",
    )


@app.get("/build/status/{job_id}", response_model=JobStatusResponse, tags=["pipeline"])
def build_status(job_id: str):
    """Poll the status of a pipeline build job."""
    job = _get_job(job_id, "build")
    elapsed = (
        round((job.finished_at or time.time()) - job.started_at, 1)
    )
    return JobStatusResponse(
        job_id      = job.job_id,
        kind        = job.kind,
        status      = job.status,
        message     = job.message,
        result      = job.result,
        started_at  = job.started_at,
        finished_at = job.finished_at,
        elapsed_s   = elapsed,
    )


@app.post("/evaluate", response_model=JobResponse, status_code=202, tags=["evaluation"])
def evaluate(req: EvaluateRequest):
    """Trigger the Ragas evaluation in the background.

    Requires the pipeline to be built and the synthetic testset to exist
    (run `python -m src.testset_gen` first, or just POST /build which also
    implicitly loads the pipeline). Returns a job_id to poll.

    Evaluation runs ~3 minutes per row on CPU (30 rows ≈ 90 min total).
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                _pipeline_load_error
                or "Pipeline not ready. Call POST /build first."
            ),
        )
    job = _new_job("evaluate")
    _executor.submit(_run_evaluate, job, req)
    return JobResponse(
        job_id  = job.job_id,
        status  = job.status,
        message = "Evaluation job queued. Poll GET /evaluate/status/{job_id} for progress.",
    )


@app.get("/evaluate/status/{job_id}", response_model=JobStatusResponse, tags=["evaluation"])
def evaluate_status(job_id: str):
    """Poll the status of an evaluation job.

    When status=done, the `result` field contains aggregate Ragas scores.
    Full per-question CSV is written to data/eval_results/ragas_eval_results.csv.
    """
    job = _get_job(job_id, "evaluate")
    elapsed = round((job.finished_at or time.time()) - job.started_at, 1)
    return JobStatusResponse(
        job_id      = job.job_id,
        kind        = job.kind,
        status      = job.status,
        message     = job.message,
        result      = job.result,
        started_at  = job.started_at,
        finished_at = job.finished_at,
        elapsed_s   = elapsed,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start():
    """Called by the `merger-api` console script defined in setup.py."""
    parser = argparse.ArgumentParser(description="Start the RAG pipeline API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (dev only).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn workers (default: 1). "
                             "Keep at 1 — the in-memory pipeline state is not shared across workers.")
    args = parser.parse_args()

    if args.workers > 1:
        logger.warning(
            "⚠️  Multiple workers requested (%d). The in-memory pipeline (_pipeline) "
            "is NOT shared across worker processes. Each worker will load its own copy "
            "of the model on startup. Set --workers 1 unless you have enough RAM.",
            args.workers,
        )

    uvicorn.run(
        "src.api:app",
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        workers = args.workers,
        log_level = "info",
    )


if __name__ == "__main__":
    start()