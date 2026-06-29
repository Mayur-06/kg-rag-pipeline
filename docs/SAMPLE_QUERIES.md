# Sample Queries — Activision Blizzard / Microsoft Merger Corpus

This corpus mirrors a real Bain-style due-diligence drop:

| # | Document | Role in the corpus |
|---|---|---|
| 1 | Activision Blizzard SEC **Form 8-K** | Legal layer — shareholder vote & deal approval |
| 2 | **AWS Well-Architected Framework** whitepaper | Technical standard — cloud architecture compliance baseline |
| 3 | Activision Blizzard **2022 Annual Report (10-K)** | Target company financials, franchises, risk factors |
| 4 | **Newzoo Games Market Report** | External market/industry benchmarks |

Place the four PDFs in `data/raw_pdfs/` before running `python -m src.build_pipeline`.

The point of this query set is to deliberately stress **cross-document fusion** —
the exact thing flat semantic search fails at, per the project brief. Single-document
factual lookups are included too, as a precision baseline.

---

## Single-document lookups (precision baseline — BM25 should dominate these)

1. "What was the exact per-share cash price Microsoft agreed to pay for Activision Blizzard?"
   *(8-K — exact dollar figure, a BM25-favorable exact-match case)*
2. "What percentage of shareholder votes were cast in favor of the merger proposal?"
   *(8-K — numeric, exact-match)*
3. "List Activision Blizzard's core game franchises mentioned in the 2022 Annual Report."
   *(10-K — named entities: Call of Duty, World of Warcraft, Candy Crush, etc.)*
4. "What is the global games market revenue forecast according to Newzoo?"
   *(Market report — numeric benchmark)*
5. "Which AWS Well-Architected Framework pillar addresses fault tolerance and disaster recovery?"
   *(AWS whitepaper — "Reliability" pillar, technical terminology)*

## Cross-document reasoning (the hard cases — hybrid + rerank should matter here)

6. "Does Activision's 10-K risk factors section mention reliance on third-party cloud
   infrastructure, and if so, which AWS Well-Architected pillar would address that risk?"
   *(10-K risk factors <-> AWS whitepaper — requires fusing legal/financial risk language
   with technical architecture vocabulary; classic semantic-search blind spot.)*

7. "Given the shareholder approval terms in the 8-K, what operational or technology risks
   disclosed in the 10-K would Microsoft need to address post-acquisition?"
   *(8-K <-> 10-K — deal mechanics linked to disclosed risk factors)*

8. "How does Activision's revenue performance in the 10-K compare to the overall games
   market growth trends reported by Newzoo?"
   *(10-K <-> Market report — company-specific financials vs. macro benchmark)*

9. "If Activision's game services are hosted on infrastructure resembling the AWS
   Well-Architected reference architecture, which operational excellence or security
   pillar gaps might be flagged given the risk factors disclosed in the 10-K?"
   *(10-K <-> AWS whitepaper, multi-hop — this is the kind of question Ragas'
   `multi_context`/`reasoning` evolutions should generate variants of.)*

10. "Considering Newzoo's player engagement trends and Activision's franchise lineup in
    the 10-K, which franchises are best positioned given the market report's growth
    segments (e.g. mobile vs. console)?"
    *(Market report <-> 10-K — strategic synthesis question, good Faithfulness stress-test
    since the model must avoid inventing a strategic conclusion the documents don't state)*

## Adversarial / "should refuse" cases (Faithfulness stress tests)

11. "What did Microsoft's CEO say about Activision's culture during the shareholder vote?"
    *(Likely not in the 8-K's procedural text — checks the pipeline says "not found in
    context" rather than hallucinating a quote.)*
12. "What is Activision's stock price today?"
    *(Out of scope for a static 10-K/8-K — should be refused, not guessed.)*

---

Run any of these via:
```bash
python -m src.main
> Does Activision's 10-K risk factors section mention reliance on third-party cloud infrastructure...
```

Or as part of the automated suite — `src/generate_testset.py` produces 30 *additional*
synthetic variants of this same pattern automatically; this file is for manual,
narratable demo purposes (e.g. walking the Bain Gurugram team through the live CLI).
