# Enterprise Knowledge Graph & Hybrid RAG Pipeline
### Corporate M&A Due-Diligence · Activision / Microsoft Corpus

> Built as a portfolio project demonstrating production-grade retrieval-augmented
> generation over heterogeneous corporate documents: SEC legal filings, cloud
> architecture whitepapers, annual reports, and market research.

---

## Why this pipeline is different from a tutorial RAG

Most RAG demos use a single homogeneous document corpus and a flat text splitter.
This pipeline is designed for a real consulting scenario (Bain-style M&A
due-diligence) where the hard queries **cross document types**:

> *"Which legal clause governs the cloud infrastructure obligation mentioned
> in Activision's 10-K risk factors — and does that infrastructure meet the
> AWS Well-Architected Reliability pillar?"*

No single document contains the answer. Flat semantic search fails because
"Section 7.3(b)" has no embedding similarity to "fault tolerance." This
pipeline solves that with four compounding techniques:

| Technique | Why it matters here |
|---|---|
| **LlamaParse ingestion** | Preserves Markdown table structure from financial tables and AWS architecture diagrams — naive pdfplumber flattens these into unusable text |
| **Hierarchical parent/child chunking** | Small sentence-level child nodes for precise retrieval; large paragraph/table parent nodes for full context to the LLM |
| **Hybrid BM25 + dense retrieval with RRF** | Dense vectors catch conceptual similarity; BM25 catches exact tokens (clause IDs, SKUs, dollar figures); RRF fuses both on rank position, not score magnitude |
| **Cohere rerank + compression** | Collapses top-20 fused candidates to top-5 most relevant before generation; strips noise without paraphrasing legal text |

---

## Corpus (4 documents, 4 roles)

| File (place in `data/raw_pdfs/`) | Role |
|---|---|
| `activision_8k_merger_vote.pdf` | **Legal** — SEC Form 8-K, shareholder vote approving Microsoft acquisition |
| `aws_well_architected_framework.pdf` | **Technical** — AWS Well-Architected Framework whitepaper (cloud compliance baseline) |
| `activision_blizzard_2022_10k.pdf` | **Financial** — Activision Blizzard 2022 Annual Report (10-K) |
| `newzoo_games_market_report.pdf` | **Market** — Newzoo Games Market Report (industry benchmarks) |

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
              ├──▶ OpenAI text-embedding-3-small ──▶ Pinecone (dense index)
              └──▶ BM25Okapi                     ──▶ BM25Index (sparse, cached .pkl)

Query time:
  user query
    ├── dense retrieval  (Pinecone, top-20)  ─┐
    └── sparse retrieval (BM25,    top-20)  ─┴─▶ RRF fusion (top-20 fused)
                                                       │
                                              child → parent expansion
                                                       │
                                              Cohere Rerank (top-20 → top-5)
                                                       │
                                              context compression (strip filler)
                                                       │
                                              gpt-4o-mini synthesis (grounded, cited)
                                                       │
                                              RAGResponse (answer + source attribution)
```

---

## Setup

### 1. Clone and install

```powershell
git clone <your-repo-url>
cd kg-rag-pipeline
python -m venv RAGvenv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\RAGvenv\Scripts\Activate.ps1
pip install -r requirements.txt
# For development (imports `src` as editable package):
pip install -e .
```

On macOS/Linux:

```bash
python -m venv RAGvenv
source RAGvenv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> Note: `requirements.txt` now pins `langchain==1.3.10` and `langchain-community==0.3.31` for compatibility with `ragas==0.2.3`.

### 2. Configure API keys

```bash
copy .env .env.local  # Windows PowerShell
# or
cp .env .env.local   # macOS/Linux
```

Edit `.env.local` and set your Pinecone key:

```text
PINECONE_API_KEY=your_pinecone_api_key
```

For the free local-model edition, only `PINECONE_API_KEY` is strictly required. The current corpus already includes cached Markdown in `data/parsed_markdown/`, so `LLAMA_CLOUD_API_KEY` is only needed if you add new PDFs and want to re-parse them with LlamaParse.

Validate your environment with:

```bash
python -m src.config
# ✅ Config OK  |  embed=...  |  llm_backend=...  |  llm=...
```

Validate your keys are set correctly (no placeholders, no empty values):

```bash
python -m src.config
# ✅ All required API keys are present.
```

### 3. Add documents

Drop your four PDFs into `data/raw_pdfs/`. See the README in that folder for
expected document roles.

---

## Running the pipeline

### Step 1 — Build indexes (run once)

```bash
python -m src.build_pipeline
```

This runs the full ingestion → chunking → embedding → indexing sequence and
caches the results. Re-running skips documents already parsed by LlamaParse
and does **not** re-embed already-upserted nodes (Pinecone upsert is idempotent).

Approximate time and cost (4-document corpus):
- LlamaParse parsing: ~2–5 min (depends on document length and server load)
- OpenAI embedding: ~$0.05–0.15 (text-embedding-3-small at $0.02/1M tokens)
- BM25 build: <10 seconds (pure Python, no API calls)

### Step 2 — Query interactively

```bash
python -m src.main
```

```
> Which legal clause governs the cloud infrastructure mentioned in the technical SOP?
> What operational risks in the 10-K would Microsoft need to remediate post-close?
> exit
```

Or use the installed entry point:
```bash
merger-query
```

### Step 3 — Generate synthetic testset (run once)

```bash
python -m src.testset_gen
```

Generates 30 synthetic Q&A pairs from the corpus using Ragas' TestsetGenerator,
distributed across:
- 35% single-hop specific (factual lookups)
- 35% multi-hop specific (cross-document factual)
- 30% multi-hop abstract (cross-document reasoning)

Cached to `data/eval_results/synthetic_testset.csv`. Re-running loads the cache.

### Step 4 — Evaluate

```bash
python -m src.evaluate
```

Runs the full pipeline on all 30 test questions and computes Ragas scores:

```
=================================================================
  RAGAS EVALUATION RESULTS — MERGER DUE-DILIGENCE RAG PIPELINE
=================================================================
  faithfulness           0.XXX  |████████████████████░░░░░░░░░░|
  answer_relevancy       0.XXX  |███████████████████████░░░░░░░|
  context_precision      0.XXX  |█████████████████████░░░░░░░░░|
  context_recall         0.XXX  |████████████████████░░░░░░░░░░|
=================================================================
```

Writes:
- `data/eval_results/ragas_eval_results.csv` — per-question scores (30 rows × 4 metrics)
- `data/eval_results/ragas_eval_summary.md` — Markdown report with diagnostic guide

### Step 5 — Visualize results

```bash
python -m src.visualize_results
```

Generates `data/eval_results/eval_visualization.html` — open in any browser:
- **Spider chart**: four-metric balance at a glance
- **Per-question heatmap**: which question types fail and on which metrics
- **Score distribution histograms**: consistency vs outlier pattern diagnosis

---

## Offline tests (no API keys required)

These validate the pure logic — RRF math, tokenizer, table detection, config
validation — and should be the first thing you run after cloning:

```bash
python tests/test_rrf.py
python tests/test_chunking_and_compression.py
python tests/test_config.py

# Or all at once:
pytest tests/ -v
```

All 18 tests are self-contained (no external library imports, no network).

---

## Project structure

```
kg-rag-pipeline/
├── .env.example                 # Copy to .env and fill in keys
├── requirements.txt
├── setup.py
├── pytest.ini
│
├── data/
│   ├── raw_pdfs/                # ← Drop your 4 PDFs here
│   ├── parsed_markdown/         # LlamaParse output (auto-generated, cached)
│   ├── storage/                 # BM25 index + parent node cache (auto-generated)
│   └── eval_results/            # Ragas CSV, Markdown summary, HTML visualization
│
├── src/
│   ├── config.py                # Centralized settings + key validation
│   ├── ingest.py                # PDF → Markdown via LlamaParse (async, cached)
│   ├── chunking.py              # Hierarchical parent/child node parser
│   ├── dense_index.py           # OpenAI embeddings → Pinecone upsert/load
│   ├── sparse_index.py          # BM25Okapi build/cache/query
│   ├── retrieval.py             # HybridRetriever: dense + sparse + RRF + parent expansion
│   ├── rerank.py                # Cohere Rerank (top-20 → top-5) + compression
│   ├── query_engine.py          # RAGPipeline: end-to-end query + gpt-4o-mini synthesis
│   ├── build_pipeline.py        # One-time build script (ingest → chunk → index)
│   ├── load_pipeline.py         # Shared loader for query + eval scripts
│   ├── main.py                  # Interactive CLI
│   ├── testset_gen.py           # Ragas synthetic testset generation (30 Q&A pairs)
│   ├── evaluate.py              # Ragas eval harness (4 metrics)
│   └── visualize_results.py     # HTML report: radar + heatmap + histograms
│
├── tests/
│   ├── test_rrf.py              # RRF math (5 tests, self-contained)
│   ├── test_chunking_and_compression.py  # Tokenizer, table detection, compress (8 tests)
│   └── test_config.py           # Key validation logic (5 tests)
│
└── docs/
    └── SAMPLE_QUERIES.md        # 12 curated queries for the demo, with rationale
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

**Why not use an LLM-based context compressor?**
LLM-based summarization of retrieved chunks risks paraphrasing "Section 4.2(b)(ii)"
into "a contract clause" — which for M&A due-diligence destroys the exact reference
the analyst needs. The compression here is deterministic filler-stripping only
(page footers, image placeholders, excessive whitespace); it reduces token cost
without touching the legal/technical specifics.

**Why the same RAGPipeline object in evaluate.py as in main.py?**
A common gap in RAG projects: the evaluation script silently reimplements a
slightly different retrieval flow than the production query path, so eval scores
don't reflect what users actually experience. Both `evaluate.py` and `main.py`
import and call the same `RAGPipeline` class from `query_engine.py`.

---

## Extending the pipeline

| Goal | Where to change |
|---|---|
| Add a new document type | Drop PDF in `data/raw_pdfs/`, re-run `build_pipeline` |
| Swap vector database (e.g. Weaviate) | Replace `dense_index.py` — `HybridRetriever` consumes any LlamaIndex `VectorStoreIndex` |
| Try a different reranker (FlashRank, BGE) | Replace the `cohere.Client` call in `rerank.py` |
| Increase test set size | Set `DEFAULT_TESTSET_SIZE` in `testset_gen.py` |
| Add a 5th Ragas metric (e.g. `answer_correctness`) | Add to the `metrics` list in `evaluate.py` |
