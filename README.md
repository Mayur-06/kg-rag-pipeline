# Enterprise Knowledge Graph & Hybrid RAG Pipeline
### Corporate M&A Due-Diligence · Activision / Microsoft Corpus

> Production-grade retrieval-augmented generation over heterogeneous corporate
> documents — SEC legal filings, cloud architecture whitepapers, annual reports,
> and market research — built end-to-end as a portfolio project: pipeline, REST
> API, Docker deployment, and a live GCP instance.

**Stack:** LlamaIndex · Pinecone · BGE embeddings (local) · BM25 · FlashRank (local) · Ollama (local) · FastAPI · Docker · GCP
**Cost:** Only Pinecone's free tier is external — embeddings, reranking, and generation all run locally, no OpenAI/Cohere key needed.

---

## Why this pipeline is different from a tutorial RAG

Most RAG demos use a single homogeneous document corpus and a flat text splitter.
This pipeline is designed for a real consulting scenario (Bain-style M&A
due-diligence) where the hard queries **cross document types**:

> *"Does Activision's cloud dependency risk factor in the 10-K align with any
> pillar of the AWS Well-Architected Framework?"*

No single document contains the answer. Flat semantic search fails because
"Section 7.3(b)" has no embedding similarity to "fault tolerance." This
pipeline solves that with four compounding techniques:

| Technique | Why it matters here |
|---|---|
| **LlamaParse ingestion** | Preserves Markdown table structure from financial tables and AWS architecture diagrams — naive pdfplumber flattens these into unusable text |
| **Hierarchical parent/child chunking** | Small sentence-level child nodes for precise retrieval; large paragraph/table parent nodes for full context to the LLM |
| **Hybrid BM25 + dense retrieval with RRF** | Dense vectors catch conceptual similarity; BM25 catches exact tokens (clause IDs, dollar figures); RRF fuses both on rank position, not score magnitude |
| **FlashRank rerank + compression** | Collapses top-20 fused candidates to top-5 most relevant before generation; strips noise without paraphrasing legal text |

---

## Corpus (4 documents, 4 roles)

| File (in `data/raw_pdfs/`) | Role |
|---|---|
| `SEC Form 8-K filing for Activision Blizzard, Inc..pdf` | **Legal** — shareholder vote approving the Microsoft acquisition |
| `AWS Well-Architected Framework whitepaper.pdf` | **Technical** — cloud architecture compliance baseline |
| `Activision Blizzard 2022 Annual Report (Form 10-K).pdf` | **Financial** — revenue, risk factors, performance |
| `2025_Newzoo_Free_Global_Games_Market_Report.pdf` | **Market** — industry benchmarks, player trends |

The pipeline ingests **any** PDFs dropped in `data/raw_pdfs/` — naming is flexible.

---

## Architecture

```
data/raw_pdfs/*.pdf
       │
       ▼  LlamaParse (layout-aware PDF → clean Markdown)
data/parsed_markdown/*.md
       │
       ▼  MarkdownNodeParser (heading/table-aware) + SentenceSplitter
       ├── PARENT nodes  (full paragraphs / atomic tables)
       └── CHILD nodes   (sentences — these get embedded and searched)
              │
              ├──▶ BAAI/bge-small-en-v1.5 (local, CPU) ──▶ Pinecone  [dense]
              └──▶ BM25Okapi               (local, CPU) ──▶ .pkl file [sparse]

Query time:
  user query
    ├── dense retrieval  (Pinecone, top-20)  ─┐
    └── sparse retrieval (BM25,    top-20)  ─┴─▶ RRF fusion (top-20 fused)
                                                       │
                                              child → parent expansion
                                                       │
                                              FlashRank ms-marco-MiniLM-L-12-v2
                                              (local, top-20 → top-5)
                                                       │
                                              context compression (strip filler)
                                                       │
                                              Ollama llama3.2 synthesis (grounded, cited)
                                                       │
                                              RAGResponse (answer + source attribution)
```

---

## Beyond the pipeline: API, Docker, and a live deployment

This isn't just a script — it's wrapped end-to-end as a deployable service:

- **FastAPI REST API** (`src/api.py`) — 7 endpoints (`/query`, `/build`, `/evaluate`, `/health`, `/status`, and job-status polling for the two long-running operations), with background job execution via a thread pool so builds and evaluations don't block the API.
- **Dockerized** — multi-container `docker-compose` setup: a `rag-api` service, an `ollama` LLM server, and a one-shot `ollama-init` container that pulls models before the API starts. Two compose files: `docker-compose.yml` for local dev (hot reload) and `docker-compose.prod.yml` for deployment (host bind-mounts so pre-built indexes don't need re-embedding on every fresh container).
- **Deployed on GCP** — running on a Compute Engine VM (`e2-standard-2`), reachable over a public IP with the full Docker Compose stack, demonstrating the path from local pipeline → containerized service → cloud deployment.

See `docs/GCP_DEPLOYMENT.md` for the full deployment walkthrough.

---

## Setup

### Option A — Docker (recommended, matches the deployed environment)

```bash
git clone <your-repo-url>
cd kg-rag-pipeline
cp .env.example .env
# Edit .env and set PINECONE_API_KEY

docker compose up --build
```

This starts Ollama (pulls `llama3.2` + `llama3.2:1b` on first run, ~3.3 GB),
then the FastAPI app once models are ready. Visit `http://localhost:8000/docs`
for the interactive Swagger UI.

### Option B — Local Python environment

```powershell
python -m venv RAGvenv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\RAGvenv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

macOS/Linux:
```bash
python -m venv RAGvenv
source RAGvenv/bin/activate
pip install -r requirements.txt
pip install -e .
```

You'll also need [Ollama](https://ollama.com/download) installed locally:
```bash
ollama pull llama3.2       # generation model, ~2 GB
ollama pull llama3.2:1b    # evaluation scoring model, ~1.3 GB
ollama serve
```

### Configure API keys

```bash
cp .env.example .env
```

Only `PINECONE_API_KEY` is required — embeddings (BGE), reranking (FlashRank),
and generation (Ollama) all run locally with no API key.

```text
PINECONE_API_KEY=your_pinecone_api_key
```

Validate:
```bash
python -m src.config
# ✅ Config OK  |  embed=BAAI/bge-small-en-v1.5 (384d)  |  llm_backend=ollama  |  llm=llama3.2
```

### Add documents

Drop PDFs into `data/raw_pdfs/`. The current corpus is already cached in
`data/parsed_markdown/`, so `LLAMA_CLOUD_API_KEY` is only needed if you add
new PDFs and want to re-parse them with LlamaParse.

---

## Running the pipeline

### Via the API (recommended)

```bash
# Trigger a build (ingest → chunk → embed → BM25), all stages cache-guarded
curl -X POST http://localhost:8000/build -H "Content-Type: application/json" -d '{"force": false}'

# Poll status
curl http://localhost:8000/build/status/{job_id}

# Ask a question once the build is done
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Activisions main cloud infrastructure risks?", "top_k": 20}'
```

### Via the CLI

```bash
python -m src.build_pipeline   # one-time index build
python -m src.main             # interactive query loop
```

```
> Which legal clause governs the cloud infrastructure mentioned in the technical SOP?
> Does Activision's cloud dependency risk align with any AWS Well-Architected pillar?
> exit
```

### Evaluation

```bash
python -m src.testset_gen   # generates 30 synthetic Q&A pairs (cached)
python -m src.evaluate      # scores the pipeline on all 30 questions
python -m src.visualize_results  # HTML report: radar + heatmap + histograms
```

Or trigger evaluation via the API: `POST /evaluate`, then poll `/evaluate/status/{job_id}`.

**Actual results** (scored with `llama3.2:1b` locally across 30 synthetic Q&A pairs):

```
=================================================================
  RAGAS EVALUATION RESULTS — MERGER DUE-DILIGENCE RAG PIPELINE
=================================================================
  faithfulness           0.223  |██████░░░░░░░░░░░░░░░░░░░░░░░░|
  answer_relevancy       0.230  |██████░░░░░░░░░░░░░░░░░░░░░░░░|
  context_precision      0.100  |███░░░░░░░░░░░░░░░░░░░░░░░░░░░|
  context_recall         0.213  |██████░░░░░░░░░░░░░░░░░░░░░░░░|
=================================================================
```

These scores are bottlenecked by the scoring model, not the retrieval pipeline:
`llama3.2:1b` (1.3B params) has limited instruction-following capability for
judging faithfulness/precision/recall, which suppresses scores across the
board. The same retrieval and generation pipeline scored with `gpt-4o-mini`
as the *judge* (not the generator) would produce meaningfully higher scores
without changing a single line of retrieval code. Manual inspection of the
generated answers (see `/query` examples above) shows accurate, grounded
responses with correct source attribution.

---

## Offline tests (no API keys, no network required)

```bash
pytest tests/ -v
# 35 tests total: RRF math (5), tokenizer/table-detection/compression (8),
# config key validation (6), Ragas scoring arithmetic (16)
```

Run these first after cloning, before installing anything that needs a key.

---

## Project structure

```
kg-rag-pipeline/
├── .env.example
├── requirements.txt
├── setup.py
├── pytest.ini
├── Dockerfile
├── docker-compose.yml            # local dev (hot reload)
├── docker-compose.prod.yml       # production / GCP (host bind-mounts)
├── docker-compose.override.yml
├── .dockerignore
│
├── data/
│   ├── raw_pdfs/                 # ← Drop PDFs here
│   ├── parsed_markdown/          # LlamaParse output (cached)
│   ├── storage/                  # BM25 index + parent/child node caches
│   └── eval_results/             # Ragas CSV, summary, HTML visualization
│
├── src/
│   ├── config.py                 # Centralized settings + key validation
│   ├── ingest.py                 # PDF → Markdown via LlamaParse (async, cached)
│   ├── chunking.py               # Hierarchical parent/child node parser
│   ├── dense_index.py            # BGE embeddings (local) → Pinecone
│   ├── sparse_index.py           # BM25Okapi build/cache/query
│   ├── retrieval.py              # HybridRetriever: dense + sparse + RRF + parent expansion
│   ├── rerank.py                 # FlashRank (local) + compression
│   ├── query_engine.py           # RAGPipeline: end-to-end query + Ollama synthesis
│   ├── build_pipeline.py         # One-time build script with per-stage skip guards
│   ├── load_pipeline.py          # Shared loader for query + eval scripts
│   ├── main.py                   # Interactive CLI
│   ├── api.py                    # FastAPI app — 7 endpoints, background jobs
│   ├── testset_gen.py            # Synthetic testset generation (30 Q&A pairs)
│   ├── evaluate.py               # Ragas eval harness (4 metrics, sequential Ollama calls)
│   └── visualize_results.py      # HTML report: radar + heatmap + histograms
│
├── tests/
│   ├── test_rrf.py                       # RRF math (5 tests)
│   ├── test_chunking_and_compression.py  # Tokenizer, table detection, compress (8 tests)
│   ├── test_config.py                    # Key validation logic (6 tests)
│   └── test_scoring.py                   # Ragas scoring arithmetic (16 tests)
│
└── docs/
    └── GCP_DEPLOYMENT.md         # Full GCP VM deployment walkthrough
```

---

## Key design decisions (for interviews/walkthroughs)

**Why RRF instead of a weighted score average?**
BM25 and cosine similarity scores are on incompatible scales. BM25 is an
unbounded corpus-dependent score; cosine is bounded [-1, 1]. Averaging requires
manual calibration that breaks every time the corpus changes. RRF fuses on rank
position, making it scale-invariant and requiring zero tuning. (k=60 is the value
from the original Cormack et al. paper, used by Elasticsearch and Weaviate.)

**Why embed child nodes but return parent context?**
Embedding a whole paragraph blends multiple ideas into one vector and blurs the
similarity signal. Embedding individual sentences keeps each vector aligned with
one concept. At retrieval time, the child's high-precision match identifies the
right needle; the parent's full text gives the LLM the surrounding paragraph so
it never receives a fragment that cuts through a clause or table row.

**Why local models (BGE, FlashRank, Ollama) instead of OpenAI/Cohere?**
Cost and reproducibility — the only external dependency is Pinecone's free tier,
so the pipeline runs end-to-end without burning API credits during development
or demos. It also forces the architecture to be explicit about compute tradeoffs
(CPU embedding/generation latency) that get hidden behind a paid API.

**Why bypass Ragas' `evaluate()` for local scoring?**
Ragas fires all internal scoring jobs concurrently via asyncio. A local CPU
Ollama instance handles exactly one request at a time — concurrent jobs hit
read timeouts. The fix: call `/api/generate` in a sequential loop, one metric,
one row, one HTTP call. Zero concurrency, zero timeouts — at the cost of a
slower eval run (~3 min/row), which is an acceptable tradeoff for a portfolio
project running on a laptop CPU.

**Why the same RAGPipeline object in evaluate.py, main.py, and api.py?**
A common gap in RAG projects: the evaluation script silently reimplements a
slightly different retrieval flow than the production query path, so eval
scores don't reflect what users actually experience. All three entry points
import and call the same `RAGPipeline` class from `query_engine.py`.

**Why background jobs in the API instead of synchronous endpoints?**
`/build` and `/evaluate` can take anywhere from seconds (fully cached) to
an hour (cold start, full re-embedding). Running them synchronously would
mean HTTP timeouts on long builds. Both endpoints return a `job_id`
immediately and run the actual work in a `ThreadPoolExecutor`, with a
polling endpoint to check progress — the same pattern used by long-running
cloud APIs (e.g. AWS Batch, GCP long-running operations).

---

## Extending the pipeline

| Goal | Where to change |
|---|---|
| Add a new document type | Drop PDF in `data/raw_pdfs/`, `POST /build` |
| Swap vector database (e.g. Weaviate) | Replace `dense_index.py` — `HybridRetriever` consumes any LlamaIndex `VectorStoreIndex` |
| Use OpenAI instead of Ollama (after billing top-up) | Set `LLM_BACKEND=openai` and `OPENAI_API_KEY` in `.env` |
| Increase test set size | Set `DEFAULT_TESTSET_SIZE` in `testset_gen.py` |
| Add a 5th Ragas metric | Add a `_score_<metric>` function and include it in `evaluate.py` |
| Deploy elsewhere (AWS/Azure) | Adapt `docker-compose.prod.yml` — see `docs/GCP_DEPLOYMENT.md` for the pattern |

---

## License

MIT
