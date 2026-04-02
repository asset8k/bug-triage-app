# Project Context: CyberTriage (MSc Thesis)

## 1. Project Overview
**Project Name:** CyberTriage - Bug Severity Classification and Triage  
**Type:** Master's Thesis (Nazarbayev University)  
**Author:** Asset Kanatov  
**Advisor:** Professor Askar Boranbayev

### Core Mission
Build a production-like local system that classifies software bug severity into four classes:
- Critical
- Major
- Minor
- Trivial

The project compares:
1. **Baseline models:** TF-IDF + classical ML + CodeBERT
2. **LLM models:** Zero-shot and fine-tuned local GGUF models

### Thesis Value
The system does not only provide class labels; for LLM flows it also returns structured explanatory output (reasoning, summary, description) and supports evaluation, comparison, and reproducibility.

---

## 2. Current Runtime Stack

### Application Runtime
- **Backend:** FastAPI (`api/server.py`)
- **Frontend:** React + Vite + Tailwind (`frontend/`)
- **Model engines:** Python engine adapters (`src/engines/`)

### ML and LLM Stack
- **Classical ML:** scikit-learn, XGBoost, joblib
- **CodeBERT inference:** transformers + torch (local `codebert_model/`)
- **Zero-shot LLMs:** Ollama local inference
- **Fine-tuned LLMs:** GGUF via `llama-cpp-python`

### Data and Utilities
- pandas, numpy, scikit-learn metrics
- pypdf and python-docx for document parsing

---

## 3. Current Architecture (Implemented)

```text
/thesis-bug-triage
├── api/
│   └── server.py                    # FastAPI endpoints (predict, models, eval, history, parse)
├── frontend/                        # React SPA (login, ingest, result, evaluation, history)
├── src/
│   └── engines/
│       ├── base_engine.py           # Base engine interface
│       ├── baseline.py              # Classical TF-IDF model engine
│       ├── codebert.py              # CodeBERT engine
│       ├── llama_zeroshot.py        # Llama zero-shot (Ollama)
│       ├── llama_finetuned.py       # Llama fine-tuned GGUF
│       ├── qwen_zeroshot.py         # Qwen zero-shot (Ollama)
│       ├── qwen_finetuned.py        # Qwen fine-tuned GGUF
│       └── registry.py              # Model registry (single source of truth)
├── scripts/                         # Training/evaluation scripts
├── models/                          # Local model artifacts
│   ├── baseline/                    # .pkl + tfidf + config
│   └── llm/                         # GGUF files
├── codebert_model/                  # Local CodeBERT checkpoint files
├── dataset/                         # Local datasets/splits
├── results/                         # Evaluation outputs + persisted history JSON
└── README.md                        # Project entry documentation
```

---

## 4. Current Model Inventory

### Baseline / Traditional
- `lr` - Logistic Regression
- `svm` - SVM
- `rf` - Random Forest
- `nb` - Naive Bayes
- `xgb` - XGBoost
- `ensemble` - Voting Ensemble
- `codebert` - CodeBERT

### LLMs
- `ollama` - Llama 3 8B (Zero)
- `qwen` - Qwen 2.5 7B (Zero)
- `finetuned` - Llama 3 8B (Fine-tuned GGUF)
- `qwen_finetuned` - Qwen 2.5 7B (Fine-tuned GGUF)

---

## 5. Key Architectural Principles

1. **Registry-driven model integration**  
   API and UI query `list_models()` and use `get_engine(model_id)`; adding a model means implementing an engine and registering it.

2. **Decoupled reasoning flow for fine-tuned experts**  
   Fine-tuned experts predict severity first; API then calls a zero-shot reasoner to produce explanation fields.

3. **Persistent local history**  
   Ingestion history is saved by backend in `results/history_entries.json`, not session-only memory.

4. **Batch evaluation as first-class capability**  
   API can trigger evaluation scripts, and frontend displays metrics tables for baseline and LLM groups.