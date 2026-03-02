# 🔍 ML Code Analyzer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ML-powered source code quality analyzer** — extracts 30+ AST features from Python code and predicts bug risk using trained Random Forest and Gradient Boosting classifiers. Includes a live Streamlit web dashboard for interactive analysis.

> Built as a portfolio project demonstrating ML engineering, code analysis, and productionization skills relevant to AMD's source code analysis work.

---

## 📸 Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────┐
│  🔍 ML Code Analyzer                                            │
│  ─────────────────────────────────────────────────────────────  │
│  [Analyze Code] [Batch Analysis] [Model Insights] [About]       │
│                                                                 │
│  ┌──────────────────────────┐  ┌───────────────────────────┐   │
│  │  🔴 72% High Risk        │  │  Top Risk Factors         │   │
│  │  ████████████░░░░░░░░    │  │  ▶ Cyclomatic Complexity  │   │
│  │                          │  │  ▶ Max Nesting Depth      │   │
│  │  Recommendations:        │  │  ▶ Docstring Coverage     │   │
│  │  ▶ Reduce nesting depth  │  └───────────────────────────┘   │
│  │  ▶ Add docstrings        │                                   │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/harjotsinghmann/ml-code-analyzer.git
cd ml-code-analyzer

# 2. Create virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 3. Train the model (takes ~30 seconds, synthetic data — no internet needed)
python scripts/train.py

# 4. Launch the dashboard
streamlit run app/streamlit_app.py
# → Open http://localhost:8501
```

---

## 🐳 Docker Quick Start

```bash
docker-compose up
# → Open http://localhost:8501
```

The container auto-trains the model on startup.

---

## 🏗️ Architecture

```
Source Code
    │
    ▼
AST Parser (ast module)
    │
    ▼
Feature Extractor (30+ metrics)
    │  ┌─ Structural: LOC, functions, classes, imports
    │  ├─ Complexity: cyclomatic, nesting depth, branches
    │  └─ Quality: docstrings, type hints, star imports
    ▼
ML Classifier (Random Forest / Gradient Boosting)
    │
    ▼
Risk Report (0–100% score + recommendations)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **30+ Code Metrics** | AST-based structural, complexity, and quality features |
| **Two Classifiers** | Random Forest and Gradient Boosting with cross-validation |
| **Live Dashboard** | Streamlit web UI with risk gauge, charts, recommendations |
| **Batch Analysis** | Analyze entire directories or multiple uploaded files |
| **CLI Tool** | `python scripts/analyze.py --file mycode.py` |
| **No Internet Required** | Synthetic dataset generator for offline training |
| **Docker Ready** | One-command deployment with docker-compose |

---

## 📊 Model Performance

Trained on 2,500 synthetic samples generated from realistic code quality distributions:

| Metric | Random Forest | Gradient Boosting |
|---|---|---|
| Accuracy | 0.89+ | 0.87+ |
| F1 (Weighted) | 0.89+ | 0.87+ |
| ROC-AUC | 0.95+ | 0.94+ |
| Precision | 0.89+ | 0.87+ |

*Scores vary slightly by random seed. See `models/pretrained/metrics.json` for your exact results.*

---

## 📁 Project Structure

```
ml-code-analyzer/
├── app/
│   ├── streamlit_app.py          # Main Streamlit dashboard
│   └── components/               # Modular UI components
├── src/
│   ├── features/
│   │   ├── ast_extractor.py      # Core AST feature extraction
│   │   ├── complexity.py         # Cyclomatic complexity, nesting
│   │   └── code_metrics.py       # Docstrings, type hints, globals
│   ├── data/
│   │   ├── synthetic_dataset.py  # Offline dataset generator
│   │   ├── repo_miner.py         # GitHub repo mining
│   │   └── build_dataset.py      # Pipeline orchestrator
│   ├── models/
│   │   ├── classifier.py         # RF + GBT model factory
│   │   └── trainer.py            # CV training pipeline
│   ├── evaluation/
│   │   └── metrics.py            # F1, ROC-AUC, confusion matrix
│   └── visualization/
│       └── plots.py              # Charts and heatmaps
├── scripts/
│   ├── train.py                  # Train models from CLI
│   ├── analyze.py                # Analyze files from CLI
│   └── build_dataset.py         # Build dataset from CLI
├── models/pretrained/            # Saved model artifacts
├── sample_data/                  # Demo Python files
├── tests/                        # pytest test suite
├── config/default_config.yaml   # Centralized configuration
├── Dockerfile
└── docker-compose.yml
```

---

## 🛠️ CLI Usage

**Analyze a single file:**
```bash
python scripts/analyze.py --file sample_data/buggy_example.py
python scripts/analyze.py --file mycode.py --output json
```

**Analyze a directory:**
```bash
python scripts/analyze.py --dir src/ --threshold 0.5 --output table
python scripts/analyze.py --dir . --output csv > report.csv
```

**Train with options:**
```bash
python scripts/train.py --synthetic --samples 5000
python scripts/train.py --grid-search   # hyperparameter tuning
python scripts/train.py --models random_forest gradient_boosting
```

---

## 📐 Feature Extraction Details

### Structural Features
`total_lines` · `code_lines` · `blank_lines` · `comment_lines` · `comment_ratio`
`num_functions` · `num_classes` · `num_methods` · `avg_function_length` · `max_function_length`
`num_imports` · `num_unique_imports`

### Complexity Features
`cyclomatic_complexity` · `max_nesting_depth` · `avg_nesting_depth`
`num_branches` · `num_loops` · `num_try_except` · `num_assertions` · `num_lambda_functions`

### Code Quality Features
`has_docstrings` · `docstring_coverage` · `has_type_hints` · `type_hint_coverage`
`avg_identifier_length` · `num_global_variables` · `num_nested_functions`
`num_return_statements` · `uses_star_import`

---

## 🧪 Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --tb=short
pytest tests/ --cov=src --cov-report=html
```

---

## 🔮 Future Improvements

- **C/C++ support** via `tree-sitter` — directly applicable to AMD's GPU computing codebase (HIP, ROCm)
- **AMD ROCm acceleration** — GPU-accelerated feature extraction for large-scale repo scanning
- **Deep learning models** — CodeBERT / GraphCodeBERT for semantic understanding beyond syntax
- **VS Code extension** — real-time risk indicators in the editor gutter
- **CI/CD integration** — GitHub Actions quality gate that blocks PRs above risk threshold
- **Multi-language support** — JavaScript, Rust, Go via tree-sitter unified AST

---

## 📄 License

MIT © 2025 Harjot Singh Mann
