# ============================================================
# Enterprise Knowledge Graph & Hybrid RAG Pipeline
# FastAPI application image
#
# Build:  docker build -t rag-pipeline .
# Run:    docker compose up   (preferred — starts Ollama too)
# ============================================================

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.4.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
COPY setup.py .
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# ── Data directory skeleton ──────────────────────────────────────────────────
# Owned by root so the process can always write to mounted volumes.
RUN mkdir -p \
        data/raw_pdfs \
        data/parsed_markdown \
        data/storage \
        data/eval_results \
        .cache/huggingface \
        .cache/flashrank

# ── Runtime config ───────────────────────────────────────────────────────────
# Running as root inside the container so writes to Docker named volumes
# always succeed. The container is isolated from the host — root inside
# the container is not root on the host machine.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV OLLAMA_BASE_URL=http://ollama:11434

CMD ["python", "-m", "src.api"]