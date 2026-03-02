"""ML Code Analyzer — Streamlit Web Dashboard.

A polished, production-style web interface for the ML-powered code quality
analyzer. Provides live code analysis, batch processing, model insights,
and educational content about how the system works.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure project root is on Python path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.ast_extractor import extract_all_features
from src.models.classifier import load_model
from src.data.synthetic_dataset import FEATURE_NAMES
from app.components.code_input import render_code_input
from app.components.risk_dashboard import render_risk_gauge, render_recommendations
from app.components.feature_display import render_feature_table, render_top_features_chart
from app.components.batch_analysis import render_batch_analysis

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Code Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = ROOT / "models" / "pretrained"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #F8FAFC; }
    .sidebar .sidebar-content { background-color: #1E293B; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .stButton > button {
        background: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover { background: #1D4ED8; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model...")
def load_analyzer_model():
    """Load the pre-trained model from disk (cached across sessions)."""
    try:
        model, feature_names, metrics = load_model(MODEL_DIR)
        if not feature_names:
            feature_names = FEATURE_NAMES
        return model, feature_names, metrics, None
    except FileNotFoundError as exc:
        return None, FEATURE_NAMES, {}, str(exc)


def predict_risk(source_code: str, model, feature_names: list[str]) -> dict:
    """Run feature extraction and model inference on source code.

    Args:
        source_code: Raw Python source code string.
        model: Loaded scikit-learn classifier.
        feature_names: Feature names the model was trained on.

    Returns:
        Dictionary with risk_score, risk_level, features, top_factors.
    """
    features = extract_all_features(source_code)
    x = np.array(
        [float(features.get(name, 0.0)) for name in feature_names],
        dtype=np.float64,
    ).reshape(1, -1)

    score = float(model.predict_proba(x)[0][1])

    if score < 0.3:
        level = "Low"
    elif score < 0.6:
        level = "Medium"
    elif score < 0.8:
        level = "High"
    else:
        level = "Critical"

    top_factors = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        ranked = sorted(
            zip(feature_names, importances),
            key=lambda t: t[1],
            reverse=True,
        )
        top_factors = [
            {"feature": name, "importance": float(imp), "value": features.get(name, 0)}
            for name, imp in ranked[:3]
        ]

    return {
        "risk_score": score,
        "risk_level": level,
        "features": features,
        "top_factors": top_factors,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(model_loaded: bool, metrics: dict) -> str:
    """Render the sidebar navigation and project branding."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding:10px 0 20px;">
                <div style="font-size:2.5rem;">🔍</div>
                <div style="font-size:1.3rem; font-weight:800; color:#1E293B;">ML Code Analyzer</div>
                <div style="font-size:0.8rem; color:#64748B; margin-top:4px;">
                    AST · Random Forest · Streamlit
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        page = st.radio(
            "Navigation",
            ["🔬 Analyze Code", "📦 Batch Analysis", "📊 Model Insights", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.divider()

        if model_loaded:
            st.success("✅ Model loaded")
        else:
            st.error("⚠️ No model — run: python scripts/train.py")

        if metrics:
            st.markdown("**Model Performance**")
            tm = metrics.get("test_metrics", {})
            if tm:
                st.metric("Accuracy", f"{tm.get('accuracy', 0):.3f}")
                st.metric("F1 Score", f"{tm.get('f1_weighted', 0):.3f}")
                st.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.3f}")

        st.divider()
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#94A3B8; text-align:center;">
            Built for AMD ML Internship Demo<br>
            <a href="https://github.com" style="color:#60A5FA;">View on GitHub</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ── Page: Analyze Code ─────────────────────────────────────────────────────────
def page_analyze(model, feature_names: list[str], model_loaded: bool) -> None:
    """Render the main code analysis page."""
    st.markdown("## 🔬 Code Quality Analyzer")
    st.markdown(
        "Paste Python code or upload a file to instantly predict bug risk "
        "using our trained machine learning model."
    )

    if not model_loaded:
        st.error(
            "⚠️ No trained model found. Run `python scripts/train.py` to train one first."
        )
        return

    source_code = render_code_input()

    st.markdown("---")
    analyze_col, _ = st.columns([1, 3])
    with analyze_col:
        analyze_btn = st.button("🔍 Analyze Code", key="analyze_btn")

    if analyze_btn:
        if not source_code or not source_code.strip():
            st.warning("Please provide some Python code to analyze.")
            return

        with st.spinner("Extracting features and computing risk score..."):
            result = predict_risk(source_code, model, feature_names)

        score = result["risk_score"]
        level = result["risk_level"]
        features = result["features"]

        st.markdown("---")
        st.markdown("## Analysis Results")

        # Gauge + top factors side by side
        gauge_col, factors_col = st.columns([2, 1])

        with gauge_col:
            render_risk_gauge(score, level)

        with factors_col:
            st.markdown("#### Top Risk Factors")
            for factor in result["top_factors"]:
                st.markdown(
                    f"""
                    <div style="background:#F1F5F9; border-radius:8px;
                                padding:10px 14px; margin-bottom:8px;">
                        <div style="font-weight:600; font-size:0.9rem;">
                            {factor['feature'].replace('_', ' ').title()}
                        </div>
                        <div style="color:#64748B; font-size:0.82rem;">
                            Value: {factor['value']:.3f} &nbsp;|&nbsp;
                            Importance: {factor['importance']:.3f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Recommendations
        render_recommendations(features)

        # Feature breakdown + chart
        st.markdown("---")
        feat_tab, chart_tab = st.tabs(["📋 Feature Breakdown", "📊 Importance Chart"])

        with feat_tab:
            render_feature_table(features)

        with chart_tab:
            render_top_features_chart(model, feature_names, top_n=12)

        # Raw JSON for developers
        with st.expander("🛠️ Raw Feature JSON (for developers)"):
            st.json(features)


# ── Page: Model Insights ───────────────────────────────────────────────────────
def page_model_insights(model, feature_names: list[str], metrics: dict, model_loaded: bool) -> None:
    """Render the model insights page with performance metrics and charts."""
    st.markdown("## 📊 Model Insights")

    if not model_loaded:
        st.error("No model loaded. Run training first.")
        return

    test_metrics = metrics.get("test_metrics", {})
    model_name = metrics.get("model_name", "Unknown")

    st.markdown(f"**Active Model:** `{model_name.replace('_', ' ').title()}`")
    st.divider()

    # Key metrics cards
    st.markdown("### Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.4f}")
    m2.metric("F1 (Weighted)", f"{test_metrics.get('f1_weighted', 0):.4f}")
    m3.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.4f}")
    m4.metric("Precision", f"{test_metrics.get('precision_weighted', 0):.4f}")

    st.divider()

    # Feature importance
    st.markdown("### Feature Importance (Top 15)")
    st.markdown(
        "Features ranked by how much they contribute to the model's predictions. "
        "Higher importance = stronger predictor of bug risk."
    )
    render_top_features_chart(model, feature_names, top_n=15)

    # All model comparison
    all_results = metrics.get("all_model_results", {})
    if len(all_results) > 1:
        st.divider()
        st.markdown("### Model Comparison")
        comparison_data = []
        for mname, mmetrics in all_results.items():
            comparison_data.append({
                "Model": mname.replace("_", " ").title(),
                "Accuracy": f"{mmetrics.get('accuracy', 0):.4f}",
                "F1 Weighted": f"{mmetrics.get('f1_weighted', 0):.4f}",
                "ROC-AUC": f"{mmetrics.get('roc_auc', 0):.4f}",
                "Precision": f"{mmetrics.get('precision_weighted', 0):.4f}",
                "Recall": f"{mmetrics.get('recall_weighted', 0):.4f}",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # CV metrics
    cv = metrics.get("cv_metrics", {})
    if cv:
        st.divider()
        st.markdown("### Cross-Validation Results (5-Fold)")
        cv_cols = st.columns(4)
        cv_cols[0].metric("CV Accuracy", f"{cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}")
        cv_cols[1].metric("CV F1", f"{cv.get('mean_f1', 0):.4f} ± {cv.get('std_f1', 0):.4f}")
        cv_cols[2].metric("CV ROC-AUC", f"{cv.get('mean_roc_auc', 0):.4f} ± {cv.get('std_roc_auc', 0):.4f}")
        cv_cols[3].metric("CV Precision", f"{cv.get('mean_precision', 0):.4f} ± {cv.get('std_precision', 0):.4f}")


# ── Page: About ───────────────────────────────────────────────────────────────
def page_about() -> None:
    """Render the About page with project description and tech stack."""
    st.markdown("## ℹ️ About ML Code Analyzer")

    st.markdown(
        """
        **ML Code Analyzer** is a machine learning-powered tool that automatically assesses
        Python source code quality and predicts bug risk using Abstract Syntax Tree (AST)
        feature extraction and trained classification models.
        """
    )

    st.divider()
    st.markdown("### How It Works")

    c1, c2, c3 = st.columns(3)
    steps = [
        ("1️⃣ Parse Code", "AST Parser", "Python's built-in `ast` module parses source code into an Abstract Syntax Tree — a structured representation of code semantics."),
        ("2️⃣ Extract Features", "Feature Engine", "30+ metrics are computed: cyclomatic complexity, nesting depth, docstring coverage, type hint coverage, import patterns, and more."),
        ("3️⃣ Predict Risk", "ML Classifier", "A trained Random Forest or Gradient Boosting model predicts the probability that a file contains bugs, outputting a 0–100% risk score."),
    ]
    for col, (title, subtitle, desc) in zip([c1, c2, c3], steps):
        col.markdown(
            f"""
            <div style="background:white; border-radius:12px; padding:20px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.08); text-align:center; height:200px;">
                <div style="font-size:2rem; margin-bottom:8px;">{title.split()[0]}</div>
                <div style="font-weight:700; font-size:1rem; margin-bottom:4px;">{subtitle}</div>
                <div style="color:#64748B; font-size:0.85rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Tech Stack")
    tech = {
        "Core ML": "`scikit-learn` — Random Forest, Gradient Boosting, cross-validation",
        "Feature Extraction": "`ast` (stdlib) — zero-dependency AST parsing",
        "Data": "`pandas`, `numpy` — feature engineering and dataset management",
        "Web UI": "`streamlit` — interactive web dashboard",
        "Visualization": "`matplotlib`, `seaborn` — charts and heatmaps",
        "Deployment": "`Docker` — containerized, one-command deployment",
    }
    for name, desc in tech.items():
        st.markdown(f"- **{name}**: {desc}")

    st.divider()
    st.markdown("### Feature Categories")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("**Structural**")
        st.markdown("- Lines of code\n- Num functions/classes\n- Import counts\n- Function lengths")
    with fc2:
        st.markdown("**Complexity**")
        st.markdown("- Cyclomatic complexity\n- Max nesting depth\n- Branch/loop counts\n- Try-except depth")
    with fc3:
        st.markdown("**Quality**")
        st.markdown("- Docstring coverage\n- Type hint coverage\n- Identifier lengths\n- Star imports")

    st.divider()
    st.markdown(
        """
        ### Future Improvements
        - **C/C++ support** via `tree-sitter` — directly relevant for AMD's HPC/GPU codebase
        - **AMD ROCm acceleration** for large-scale repository scanning on GPU
        - **Deep learning models** (CodeBERT, GraphCodeBERT) for semantic analysis
        - **IDE plugin** (VS Code extension) for real-time feedback
        - **CI/CD integration** — GitHub Actions quality gate
        """
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Main entry point for the Streamlit application."""
    model, feature_names, metrics, error = load_analyzer_model()
    model_loaded = model is not None

    page = render_sidebar(model_loaded, metrics)

    if page == "🔬 Analyze Code":
        page_analyze(model, feature_names, model_loaded)

    elif page == "📦 Batch Analysis":
        if not model_loaded:
            st.error("No model loaded. Run `python scripts/train.py` first.")
        else:
            render_batch_analysis(model, feature_names)

    elif page == "📊 Model Insights":
        page_model_insights(model, feature_names, metrics, model_loaded)

    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
