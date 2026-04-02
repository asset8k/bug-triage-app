# Methodology (Current Implemented System)

This document describes the methodology of the current CyberTriage implementation and aligns with the real repository state.

## 1. Objective

CyberTriage classifies bug reports into four severity levels: Critical, Major, Minor, and Trivial.

The thesis compares three model families in one integrated application:
1. Classical baselines (TF-IDF + ML)
2. Transformer baseline (CodeBERT)
3. LLM approaches (zero-shot and fine-tuned GGUF)

## 2. System Architecture

### 2.1 Runtime architecture
- Frontend: React + Vite + Tailwind
- Backend: FastAPI
- Inference layer: engine adapters in `src/engines/`
- Model discovery and loading: registry (`list_models`, `get_engine`)

### 2.2 Design principle
Model-agnostic orchestration:
- UI and API do not hardcode model internals.
- New model integration follows: implement engine -> register engine.

## 3. Data and Labeling Method

### 3.1 Operational datasets
The operational pipeline uses prepared local dataset files (for example, `dataset/train.csv` and `dataset/test.csv`) with a shared label space.

### 3.2 Label space
All models are mapped to the same four-class severity taxonomy:
- Critical
- Major
- Minor
- Trivial

### 3.3 Reproducibility controls
- Fixed seeds where sampling or subsampling is used
- Shared metric computation for comparable outputs
- Persisted result CSVs in `results/`

## 4. Baseline Methodology

### 4.1 Classical baseline models
Training/evaluation scripts support:
- Logistic Regression (`lr`)
- SVM (`svm`)
- Random Forest (`rf`)
- Naive Bayes (`nb`)
- XGBoost (`xgb`)
- Voting Ensemble (`ensemble`)

Typical flow:
1. Vectorize text with TF-IDF
2. Train/load model
3. Predict four-class severity
4. Evaluate with macro metrics

### 4.2 CodeBERT baseline
CodeBERT is treated as a baseline model for comparative evaluation and UI grouping:
- Model ID: `codebert`
- Local artifacts under `codebert_model/`
- Used in ingestion and baseline batch evaluation

## 5. LLM Methodology

### 5.1 Zero-shot models
- Llama zero-shot via Ollama (`ollama`)
- Qwen zero-shot via Ollama (`qwen`)

### 5.2 Fine-tuned models
- Llama fine-tuned GGUF (`finetuned`)
- Qwen fine-tuned GGUF (`qwen_finetuned`)

### 5.3 Decoupled reasoning architecture
For fine-tuned expert models:
1. Stage 1 predicts severity
2. Stage 2 reasoner generates reasoning, summary, and description

This keeps severity classification deterministic while still providing explanatory output for ingestion UX.

## 6. Evaluation Methodology

### 6.1 Metrics
Evaluation reports:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Sample count (`n_val`)
- Duration (`duration_sec`)

### 6.2 Evaluation execution
- Baseline: `scripts/evaluate_baselines.py` (single or multiple model runs)
- LLMs: dedicated scripts per model family
- API endpoint `/api/evaluation/run` orchestrates script execution from the app

### 6.3 Result aggregation
- CSV outputs are stored in `results/`
- `/api/evaluation` merges baseline and LLM metrics for frontend tables

## 7. User-Facing Workflow Method

1. User enters bug text or uploads a document.
2. User selects a model from the registry-sourced list.
3. API predicts severity.
4. Result page renders baseline or LLM-oriented output.
5. Entry is persisted to local history (`results/history_entries.json`).
6. Batch evaluation can be run and compared in-app.

## 8. Validity and Practical Constraints

- Local model files are large and treated as runtime artifacts, not source code.
- Evaluation duration differs substantially between baseline and LLM models.
- Frontend and backend remain loosely coupled through explicit REST contracts.

## 9. Extensibility Protocol

To add a new model:
1. Implement a `BaseEngine`-compatible engine under `src/engines/`.
2. Register model ID and display name in `registry.py`.
3. Add optional evaluation script and API wiring.
4. Update docs and UI expectations accordingly.

This protocol preserves architectural consistency while enabling incremental research extensions.
