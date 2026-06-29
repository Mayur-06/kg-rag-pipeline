"""
src/ingest.py

Step 1a: Turn messy corporate PDFs (SOPs, contracts, technical whitepapers)
into clean Markdown using LlamaParse.

Why LlamaParse instead of PyPDF/pdfplumber: merger due-diligence docs are
full of multi-column layouts, embedded tables, and figures with captions.
Naive text extraction interleaves table cells with body text and destroys
row/column structure. LlamaParse's layout model keeps tables as Markdown
tables and preserves heading hierarchy, which is exactly what the
hierarchical parser in chunking.py depends on downstream.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from llama_parse import LlamaParse

from src.config import PARSED_MD_DIR, RAW_PDF_DIR, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Instructs LlamaParse's layout model what to preserve. This matters more
# than people expect — generic parsing flattens tables; this prompt keeps
# them as pipe-delimited Markdown tables so a downstream LLM can still
# read rows/columns correctly.
PARSING_INSTRUCTION = """
This is a corporate document (technical SOP, legal contract, or market
research report) used in an M&A due-diligence context.

- Preserve every table as a proper Markdown table (header row + |---| separator).
- Preserve numbered/lettered legal clause structure (e.g. "Section 4.2(a)") as-is, do not renumber.
- Preserve code blocks, config snippets, or architecture diagrams' captions verbatim in fenced code blocks.
- Use Markdown heading levels (#, ##, ###) that mirror the document's actual section hierarchy.
- Do not summarize or omit any content. This is for downstream retrieval, not human reading.
"""


def get_pdf_paths(raw_dir: Path = RAW_PDF_DIR) -> list[Path]:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {raw_dir}. Drop 3-5 corporate documents "
            f"(SOPs, contracts, technical whitepapers) there before running ingestion."
        )
    return pdfs


async def parse_single_pdf(parser: LlamaParse, pdf_path: Path, out_dir: Path) -> Path:
    """Parse one PDF to Markdown and write it to disk, caching the result.

    Caching matters here because LlamaParse calls cost money and time per
    page; re-running the pipeline during development shouldn't re-parse
    documents that haven't changed.
    """
    out_path = out_dir / f"{pdf_path.stem}.md"
    if out_path.exists():
        logger.info(f"⏭️  Skipping {pdf_path.name} — cached markdown already exists at {out_path}")
        return out_path

    logger.info(f"📄 Parsing {pdf_path.name} via LlamaParse ...")
    documents = await parser.aload_data(str(pdf_path))
    full_markdown = "\n\n---\n\n".join(doc.text for doc in documents)

    header = f"<!-- source_file: {pdf_path.name} -->\n\n"
    out_path.write_text(header + full_markdown, encoding="utf-8")
    logger.info(f"✅ Wrote {len(full_markdown):,} chars -> {out_path}")
    return out_path


async def ingest_all_pdfs(
    raw_dir: Path = RAW_PDF_DIR,
    out_dir: Path = PARSED_MD_DIR,
    force_ingest: bool = False,
) -> list[Path]:
    settings = get_settings()
    pdf_paths = get_pdf_paths(raw_dir)
    md_paths = [out_dir / f"{p.stem}.md" for p in pdf_paths]
    needs_parsing = force_ingest or any(not md_path.exists() for md_path in md_paths)

    if not needs_parsing:
        logger.info(
            "⏭️  Skipping ingestion — all PDFs already have cached Markdown."
        )
        return md_paths

    settings.validate(require=["LLAMA_CLOUD_API_KEY"])
    parser = LlamaParse(
        api_key=settings.llama_cloud_api_key,
        result_type="markdown",
        parsing_instruction=PARSING_INSTRUCTION,
        num_workers=4,
        verbose=True,
        language="en",
    )

    logger.info(f"Found {len(pdf_paths)} PDF(s) to ingest: {[p.name for p in pdf_paths]}")
    tasks = [parse_single_pdf(parser, p, out_dir) for p in pdf_paths]
    md_paths = await asyncio.gather(*tasks)
    return md_paths


def main():
    md_paths = asyncio.run(ingest_all_pdfs())
    print(f"\n🎉 Ingestion complete. {len(md_paths)} Markdown files in {PARSED_MD_DIR}")
    for p in md_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
