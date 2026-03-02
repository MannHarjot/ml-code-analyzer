# ML Code Analyzer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

I built this to get better at applying ML to something more concrete than toy datasets. The idea: parse Python source files with the `ast` module, pull out ~30 features (complexity, nesting depth, docstring coverage, etc.), and train a classifier to predict whether a file is likely to contain bugs.

The web dashboard lets you paste any Python code and get a risk score back in seconds.

---

## How it works

```
your .py file
     │
     ▼
AST parser  →  30+ features extracted
                  ├─ structural  (LOC, functions, imports)
                  ├─ complexity  (cyclomatic, nesting depth)
                  └─ quality     (docstrings, type hints)
     │
     ▼
Random Forest / Gradient Boosting
     │
     ▼
risk score + what's causing it
```

The model was trained on synthetic data generated from realistic distributions — high-complexity, undocumented code maps to buggy; well-structured, type-annotated code maps to clean. No internet required to train.

---

## Running it locally

```bash
git clone https://github.com/MannHarjot/ml-code-analyzer.git
cd ml-code-analyzer

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# model is already pre-trained, just launch:
streamlit run app/streamlit_app.py
```

Or with Docker:

```bash
docker-compose up
```

Both open at `http://localhost:8501`.

---

## CLI

```bash
# single file
python scripts/analyze.py --file mycode.py

# whole directory, export to csv
python scripts/analyze.py --dir src/ --output csv > report.csv

# retrain from scratch
python scripts/train.py --synthetic --samples 5000
```

---

## What gets extracted

**Structural** — total lines, code vs comment ratio, number of functions/classes/methods, import count, average and max function length

**Complexity** — cyclomatic complexity, max nesting depth, branch count, loop count, try/except blocks, lambda usage

**Quality signals** — docstring coverage, type hint coverage, global variable count, star imports, average identifier length

---

## Project layout

```
ml-code-analyzer/
├── app/
│   ├── streamlit_app.py
│   └── components/
├── src/
│   ├── features/        ← ast_extractor, complexity, code_metrics
│   ├── data/            ← synthetic dataset + github repo miner
│   ├── models/          ← classifier factory, training pipeline
│   ├── evaluation/
│   └── visualization/
├── scripts/             ← train.py, analyze.py, build_dataset.py
├── models/pretrained/   ← shipped model, no training needed to demo
├── sample_data/         ← clean / medium / buggy example files
├── tests/               ← 45 pytest tests
└── config/
```

---

## Model performance

Trained with 5-fold cross-validation on 2,500 samples:

| | Random Forest | Gradient Boosting |
|---|---|---|
| Accuracy | 0.89+ | 0.87+ |
| F1 | 0.89+ | 0.87+ |
| ROC-AUC | 0.95+ | 0.94+ |

Exact numbers after training are in `models/pretrained/metrics.json`.

---

## Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

45 tests covering feature extraction, complexity calculations, model save/load, and prediction output.

---

## What I'd add next

- **C/C++ support via tree-sitter** — same pipeline but for systems code, relevant for GPU kernel work (HIP/ROCm)
- **CodeBERT embeddings** on top of the AST features for better semantic understanding
- **GitHub Actions integration** — flag PRs that push a file's risk score above a threshold
- **VS Code extension** — show risk score inline while writing

---

MIT © 2025 Harjot Singh Mann
