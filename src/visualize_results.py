"""
src/visualize_results.py

Reads the Ragas CSV output from evaluate.py and produces a self-contained
HTML report with three panels:

  Panel 1 — Radar (spider) chart: the four metric aggregate scores at a
            glance. This is the "executive slide" view: one shape that
            immediately shows whether the pipeline is retrieval-limited
            (recall low, one wing collapsed inward) vs generation-limited
            (faithfulness low).

  Panel 2 — Per-question heatmap: each row is one test question; each
            column is one metric; color from red (0) to green (1). Makes
            it immediately obvious which question *types* (multi-hop
            cross-document vs single-document factual) are hardest.

  Panel 3 — Score distribution histograms: one per metric, showing
            whether scores cluster near 1.0 (most queries easy, a few
            hard outliers) or scatter (systematic retrieval gap).

No external charting library required at runtime — everything is rendered
with Plotly CDN loaded inline, so the output HTML is completely portable:
email it to the Bain Gurugram team, open it on any browser, zero setup.
"""
from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import pandas as pd

from src.config import EVAL_RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_RESULTS_PATH = EVAL_RESULTS_DIR / "ragas_eval_results.csv"
VIZ_OUTPUT_PATH   = EVAL_RESULTS_DIR / "eval_visualization.html"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}


def load_results(path: Path = EVAL_RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No eval results at {path}. Run `python -m src.evaluate` first."
        )
    df = pd.read_csv(path)
    present = [m for m in METRICS if m in df.columns]
    if not present:
        raise ValueError(
            f"CSV at {path} has no recognized metric columns. "
            f"Expected one of {METRICS}; found {list(df.columns)}."
        )
    return df


def _radar_trace(means: dict[str, float]) -> dict:
    labels = [METRIC_LABELS[m] for m in METRICS if m in means]
    values = [round(means[m], 4) for m in METRICS if m in means]
    # Close the polygon
    labels += [labels[0]]
    values += [values[0]]
    return {
        "type": "scatterpolar",
        "r": values,
        "theta": labels,
        "fill": "toself",
        "name": "Aggregate score",
        "line": {"color": "#4f7be8", "width": 2},
        "fillcolor": "rgba(79,123,232,0.18)",
    }


def _heatmap_trace(df: pd.DataFrame, present_metrics: list[str]) -> dict:
    z_rows, y_labels = [], []
    q_col = "user_input" if "user_input" in df.columns else "question"
    for i, row in df.iterrows():
        z_rows.append([round(float(row.get(m, 0)), 3) for m in present_metrics])
        raw_q = str(row.get(q_col, f"Q{i}"))
        y_labels.append(textwrap.shorten(raw_q, width=70, placeholder="..."))

    return {
        "type": "heatmap",
        "z": z_rows,
        "x": [METRIC_LABELS[m] for m in present_metrics],
        "y": y_labels,
        "colorscale": [
            [0.0,  "#d73027"],
            [0.25, "#fc8d59"],
            [0.5,  "#fee08b"],
            [0.75, "#d9ef8b"],
            [1.0,  "#1a9850"],
        ],
        "zmin": 0,
        "zmax": 1,
        "colorbar": {"title": "Score", "thickness": 14},
    }


def _histogram_traces(df: pd.DataFrame, present_metrics: list[str]) -> list[dict]:
    colors = ["#4f7be8", "#e8874f", "#4fe8a0", "#e84f7b"]
    traces = []
    for i, m in enumerate(present_metrics):
        vals = df[m].dropna().tolist()
        traces.append({
            "type": "histogram",
            "x": vals,
            "name": METRIC_LABELS[m],
            "marker": {"color": colors[i % len(colors)], "opacity": 0.78},
            "xbins": {"start": 0, "end": 1, "size": 0.1},
            "xaxis": f"x{i+1}" if i > 0 else "x",
            "yaxis": f"y{i+1}" if i > 0 else "y",
        })
    return traces


def build_html(df: pd.DataFrame) -> str:
    present = [m for m in METRICS if m in df.columns]
    means   = {m: float(df[m].mean()) for m in present}
    stds    = {m: float(df[m].std())  for m in present}

    # --- Radar figure ---
    radar_fig = {
        "data": [_radar_trace(means)],
        "layout": {
            "title": {"text": "Aggregate RAG Scores — Spider Chart", "font": {"size": 18}},
            "polar": {"radialaxis": {"visible": True, "range": [0, 1]}},
            "showlegend": False,
            "paper_bgcolor": "white",
            "height": 420,
        },
    }

    # --- Heatmap figure ---
    heatmap_fig = {
        "data": [_heatmap_trace(df, present)],
        "layout": {
            "title": {"text": "Per-Question Score Heatmap", "font": {"size": 18}},
            "xaxis": {"tickangle": -20},
            "yaxis": {"autorange": "reversed", "tickfont": {"size": 10}},
            "height": max(350, 22 * len(df) + 80),
            "margin": {"l": 420, "r": 30, "t": 60, "b": 60},
            "paper_bgcolor": "white",
        },
    }

    # --- Histogram figure (subplot grid 2×2) ---
    hist_traces = _histogram_traces(df, present)
    n = len(present)
    hist_layout: dict = {
        "title": {"text": "Score Distributions per Metric", "font": {"size": 18}},
        "paper_bgcolor": "white",
        "height": 420,
        "showlegend": False,
        "grid": {"rows": 1, "columns": n, "pattern": "independent"},
    }
    for i, m in enumerate(present):
        ax_suf = str(i + 1) if i > 0 else ""
        hist_layout[f"xaxis{ax_suf}"] = {"title": METRIC_LABELS[m], "range": [0, 1]}
        hist_layout[f"yaxis{ax_suf}"] = {"title": "Count" if i == 0 else ""}
    hist_fig = {"data": hist_traces, "layout": hist_layout}

    # --- Summary table rows ---
    tbl_rows = "".join(
        f"<tr><td><b>{METRIC_LABELS[m]}</b></td>"
        f"<td style='text-align:center'>{means[m]:.3f}</td>"
        f"<td style='text-align:center'>{stds[m]:.3f}</td>"
        f"<td style='text-align:center'>"
        f"{'🟢' if means[m]>=0.80 else ('🟡' if means[m]>=0.60 else '🔴')}"
        f"</td></tr>"
        for m in present
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ragas Eval — Merger RAG Pipeline</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#f7f8fc;margin:0;padding:0}}
  header{{background:#1a2340;color:#fff;padding:28px 40px}}
  header h1{{margin:0;font-size:1.5rem;font-weight:700}}
  header p{{margin:6px 0 0;opacity:.75;font-size:.9rem}}
  .container{{max-width:1100px;margin:0 auto;padding:30px 24px}}
  .card{{background:#fff;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.08);
         padding:28px 28px 18px;margin-bottom:28px}}
  h2{{color:#1a2340;font-size:1.15rem;margin:0 0 18px;border-bottom:2px solid #e8ecf4;
      padding-bottom:8px}}
  table{{border-collapse:collapse;width:100%}}
  th{{background:#e8ecf4;color:#1a2340;padding:9px 14px;text-align:left;font-size:.88rem}}
  td{{padding:8px 14px;border-bottom:1px solid #f0f2f8;font-size:.88rem}}
  .note{{font-size:.82rem;color:#666;margin-top:8px}}
  footer{{text-align:center;color:#999;padding:24px;font-size:.8rem}}
</style>
</head>
<body>
<header>
  <h1>Enterprise Merger RAG Pipeline — Ragas Evaluation Report</h1>
  <p>Corpus: Activision 8-K (legal) · AWS Well-Architected (technical) · 
     Activision 10-K (financial) · Newzoo (market) &nbsp;|&nbsp;
     Test set: {len(df)} synthetic Q&amp;A pairs</p>
</header>
<div class="container">

  <!-- Summary table -->
  <div class="card">
    <h2>Aggregate Scores</h2>
    <table>
      <tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Status</th></tr>
      {tbl_rows}
    </table>
    <p class="note">
      🟢 ≥ 0.80 &nbsp; 🟡 0.60–0.79 &nbsp; 🔴 &lt; 0.60 &nbsp;|&nbsp;
      All scores are on a 0–1 scale (higher = better).
    </p>
  </div>

  <!-- Radar -->
  <div class="card">
    <h2>Spider Chart — Four-Metric Balance</h2>
    <div id="radar"></div>
    <p class="note">
      A perfectly balanced pipeline fills all four quadrants equally. 
      A collapsed wing identifies the failure mode to address first.
    </p>
  </div>

  <!-- Heatmap -->
  <div class="card">
    <h2>Per-Question Heatmap</h2>
    <div id="heatmap"></div>
    <p class="note">
      Red cells are individual questions where that metric failed. 
      Horizontal red bands = one hard question across all metrics (likely a 
      genuinely ambiguous cross-document question). Vertical red bands = 
      systematic metric-level failure (fix retrieval or generation layer).
    </p>
  </div>

  <!-- Histograms -->
  <div class="card">
    <h2>Score Distributions</h2>
    <div id="histograms"></div>
    <p class="note">
      Ideal distribution: tall bar at 0.9–1.0 with a small tail. 
      A flat distribution signals high variance — the pipeline is 
      inconsistent across question types. A bimodal shape (peaks near 0 
      and 1) often means the pipeline handles one question type well and 
      fails another entirely.
    </p>
  </div>

</div>
<footer>Generated by src/visualize_results.py — Enterprise Knowledge Graph &amp; Hybrid RAG Pipeline</footer>

<script>
const radarFig    = {json.dumps(radar_fig)};
const heatmapFig  = {json.dumps(heatmap_fig)};
const histFig     = {json.dumps(hist_fig)};

Plotly.newPlot('radar',      radarFig.data,    radarFig.layout,    {{responsive:true}});
Plotly.newPlot('heatmap',    heatmapFig.data,  heatmapFig.layout,  {{responsive:true}});
Plotly.newPlot('histograms', histFig.data,     histFig.layout,     {{responsive:true}});
</script>
</body>
</html>"""
    return html


def main():
    df = load_results()
    logger.info(f"Loaded {len(df)} rows from {EVAL_RESULTS_PATH}")
    html = build_html(df)
    VIZ_OUTPUT_PATH.write_text(html, encoding="utf-8")
    logger.info(f"✅ Visualization written -> {VIZ_OUTPUT_PATH}")
    print(f"\n🎨 Open this file in any browser:\n   {VIZ_OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
