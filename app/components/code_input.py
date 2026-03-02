"""Code input component: text paste and file upload for Streamlit."""

from pathlib import Path
from typing import Optional

import streamlit as st


def render_code_input() -> Optional[str]:
    """Render a code input widget with paste and file upload options.

    Displays a tabbed interface allowing users to either paste Python code
    directly or upload a .py file. File upload takes precedence if both
    are provided.

    Returns:
        Source code string if provided, else None.
    """
    st.markdown("### Input Python Code")
    tab_paste, tab_upload, tab_sample = st.tabs(
        ["📝 Paste Code", "📁 Upload File", "🧪 Load Sample"]
    )

    source_code: Optional[str] = None

    with tab_paste:
        pasted = st.text_area(
            "Paste your Python code here:",
            height=320,
            placeholder="def my_function():\n    ...",
            key="code_paste_area",
        )
        if pasted and pasted.strip():
            source_code = pasted

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a Python file (.py)",
            type=["py"],
            key="file_uploader",
        )
        if uploaded is not None:
            try:
                source_code = uploaded.read().decode("utf-8", errors="ignore")
                st.success(f"Loaded: **{uploaded.name}** ({len(source_code):,} chars)")
                with st.expander("Preview", expanded=False):
                    st.code(source_code[:800] + ("..." if len(source_code) > 800 else ""), language="python")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    with tab_sample:
        sample_dir = Path("sample_data")
        sample_options = {
            "clean_example.py — Low Risk (well-documented)": "clean_example.py",
            "medium_risk_example.py — Medium Risk (mixed quality)": "medium_risk_example.py",
            "buggy_example.py — High Risk (poor quality)": "buggy_example.py",
        }
        selected_label = st.selectbox("Choose a sample file:", list(sample_options.keys()))
        selected_file = sample_dir / sample_options[selected_label]

        if st.button("Load Sample", key="load_sample_btn"):
            if selected_file.exists():
                source_code = selected_file.read_text(encoding="utf-8")
                st.session_state["loaded_sample"] = source_code
                st.success(f"Loaded: **{selected_file.name}**")
            else:
                st.warning(f"Sample file not found: {selected_file}")

        if "loaded_sample" in st.session_state and not source_code:
            source_code = st.session_state["loaded_sample"]
            with st.expander("Preview loaded sample", expanded=False):
                st.code(source_code[:600] + "...", language="python")

    return source_code
