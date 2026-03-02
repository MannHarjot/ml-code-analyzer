"""Batch analysis component: multi-file upload and results table."""

import io
import csv
from typing import Any

import pandas as pd
import numpy as np
import streamlit as st

from src.features.ast_extractor import extract_all_features


def _risk_level(score: float) -> str:
    """Map risk score to level string."""
    if score < 0.3:
        return "Low"
    if score < 0.6:
        return "Medium"
    if score < 0.8:
        return "High"
    return "Critical"


def _level_color(level: str) -> str:
    """Map risk level to display color."""
    return {
        "Low": "#10B981",
        "Medium": "#F59E0B",
        "High": "#EF4444",
        "Critical": "#7F1D1D",
    }.get(level, "#6B7280")


def render_batch_analysis(model: Any, feature_names: list[str]) -> None:
    """Render the batch analysis tab: upload multiple files, show risk table.

    Args:
        model: Loaded scikit-learn classifier.
        feature_names: Feature names the model expects.
    """
    st.markdown("### Batch File Analysis")
    st.info("Upload multiple Python files to analyze their risk scores all at once.")

    uploaded_files = st.file_uploader(
        "Upload Python files (.py)",
        type=["py"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if not uploaded_files:
        st.markdown(
            """
            <div style="text-align:center; padding:40px; color:#6B7280;">
                📁 Upload .py files above to begin batch analysis
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    results: list[dict] = []
    progress = st.progress(0, text="Analyzing files...")

    for i, uploaded in enumerate(uploaded_files):
        try:
            source = uploaded.read().decode("utf-8", errors="ignore")
            features = extract_all_features(source)
            x = np.array(
                [float(features.get(name, 0.0)) for name in feature_names],
                dtype=np.float64,
            ).reshape(1, -1)
            score = float(model.predict_proba(x)[0][1])
            level = _risk_level(score)
            results.append({
                "filename": uploaded.name,
                "risk_score": score,
                "risk_level": level,
                "lines": features.get("total_lines", 0),
                "complexity": features.get("cyclomatic_complexity", 0),
                "docstring_cov": features.get("docstring_coverage", 0.0),
                "features": features,
            })
        except Exception as exc:
            results.append({
                "filename": uploaded.name,
                "risk_score": None,
                "risk_level": "Error",
                "lines": 0,
                "complexity": 0,
                "docstring_cov": 0.0,
                "error": str(exc),
                "features": {},
            })

        progress.progress((i + 1) / len(uploaded_files), text=f"Analyzed: {uploaded.name}")

    progress.empty()

    # Sort by risk score descending
    valid_results = [r for r in results if r["risk_score"] is not None]
    valid_results.sort(key=lambda r: r["risk_score"], reverse=True)

    # Summary metrics
    if valid_results:
        scores = [r["risk_score"] for r in valid_results]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Files Analyzed", len(valid_results))
        col2.metric("Avg Risk Score", f"{sum(scores)/len(scores):.2%}")
        col3.metric("High/Critical", sum(1 for r in valid_results if r["risk_level"] in ("High", "Critical")))
        col4.metric("Low Risk", sum(1 for r in valid_results if r["risk_level"] == "Low"))

    # Results table
    st.markdown("#### Results (sorted by risk)")
    for result in valid_results:
        level = result["risk_level"]
        color = _level_color(level)
        score_pct = f"{result['risk_score']:.1%}"

        with st.expander(
            f"{level} | {score_pct} — {result['filename']}",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk Score", score_pct)
            c2.metric("Lines", int(result["lines"]))
            c3.metric("Complexity", int(result["complexity"]))
            c4.metric("Docstring Cov.", f"{result['docstring_cov']:.0%}")

    # Download CSV
    if valid_results:
        st.markdown("---")
        csv_data = _results_to_csv(valid_results)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_data,
            file_name="batch_analysis_results.csv",
            mime="text/csv",
        )


def _results_to_csv(results: list[dict]) -> bytes:
    """Convert batch results to CSV bytes for download.

    Args:
        results: List of analysis result dictionaries.

    Returns:
        CSV content as UTF-8 encoded bytes.
    """
    output = io.StringIO()
    fieldnames = ["filename", "risk_score", "risk_level", "lines", "complexity", "docstring_cov"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue().encode("utf-8")
