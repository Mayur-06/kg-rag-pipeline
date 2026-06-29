# Expected source documents

Drop these 4 PDFs here before running `python -m src.build_pipeline`:

1. **activision_8k_merger_vote.pdf** — SEC Form 8-K, Activision Blizzard Inc.
   shareholder vote/approval of the Microsoft acquisition proposal. (Legal layer)
2. **aws_well_architected_framework.pdf** — AWS Well-Architected Framework
   whitepaper. (Technical standard / cloud compliance baseline)
3. **activision_blizzard_2022_10k.pdf** — Activision Blizzard 2022 Annual
   Report (Form 10-K): financials, franchises, risk factors. (Target company asset profile)
4. **newzoo_games_market_report.pdf** — Newzoo Games Market Report: industry
   benchmarks, player trends, macro metrics. (Market context)

File naming is flexible — the pipeline ingests every .pdf in this folder
regardless of name — but the eval question templates in src/testset_gen.py
reference these four documents by role (legal / technical / asset / market),
so keeping one document per role makes the synthetic test set's
cross-document questions meaningful.
