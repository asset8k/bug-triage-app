# Model Architecture (Current System)

## 1. Architecture Overview

CyberTriage uses a registry-based multi-engine architecture:
- Each model is implemented as an engine under `src/engines/`
- `src/engines/registry.py` is the single source of truth for available models
- API requests use `model_id` and resolve engines through `get_engine(model_id)`

This decouples frontend and API logic from model-specific implementation details.

---

## 2. Model Groups

### 2.1 Baseline / Classical Group
- `lr`, `svm`, `rf`, `nb`, `xgb`, `ensemble`
- Inference engine: `BaselineEngine` (`src/engines/baseline.py`)
- Artifacts: `models/baseline/tfidf.pkl`, `{model_id}.pkl`, `config.json`

### 2.2 Transformer Baseline
- `codebert`
- Inference engine: `CodeBERTEngine` (`src/engines/codebert.py`)
- Artifacts: `codebert_model/` (config, tokenizer files, model weights)

### 2.3 LLM Group
- `ollama` (Llama 3 8B zero-shot)
- `qwen` (Qwen 2.5 7B zero-shot)
- `finetuned` (Llama GGUF)
- `qwen_finetuned` (Qwen GGUF)

Engines:
- `src/engines/llama_zeroshot.py`
- `src/engines/qwen_zeroshot.py`
- `src/engines/llama_finetuned.py`
- `src/engines/qwen_finetuned.py`

---

## 3. Baseline Pipeline

### Training
Training script: `scripts/train_baseline.py`

Core flow:
1. Load prepared train split from `dataset/train.csv`
2. Build TF-IDF features
3. Train selected baseline models
4. Save vectorizer, model weights, and label config in `models/baseline/`

### Inference
Inference path for baseline model IDs:
1. Load vectorizer and classifier
2. Transform text to TF-IDF space
3. Predict severity
4. Return severity, confidence/probabilities, and model metadata

---

## 4. CodeBERT Pipeline

`CodeBERTEngine` performs transformer inference using local model artifacts:
1. Load tokenizer + sequence classification model from `codebert_model/`
2. Tokenize input bug description
3. Predict class logits
4. Map prediction to the four severity labels

This model is integrated as a baseline option in ingestion and baseline evaluation UI.

---

## 5. LLM Pipeline

### Zero-shot models
Zero-shot engines query local Ollama and normalize outputs into consistent fields.

### Fine-tuned models
Fine-tuned GGUF engines provide severity classification.  
During ingestion, API applies a decoupled Stage 2 reasoner:
- Stage 1: fine-tuned expert predicts severity
- Stage 2: matching zero-shot model generates reasoning, summary, and description

This orchestration is implemented in `api/server.py`.

---

## 6. Evaluation Architecture

### Baseline evaluation
- Script: `scripts/evaluate_baselines.py`
- Supports single-model runs (`--model`) and optional subsample (`--batch-size`)
- Writes/updates `results/baseline_comparison.csv`

### LLM evaluation
- Scripts:
  - `scripts/evaluate_llama_zeroshot.py`
  - `scripts/evaluate_qwen_zeroshot.py`
  - `scripts/evaluate_llama_finetuned.py`
  - `scripts/evaluate_qwen_finetuned.py`
- Write metrics/predictions CSVs under `results/`

API endpoint `GET /api/evaluation` merges these outputs for frontend tables.

---

## 7. Persistence and Interfaces

- **Model interface:** all engines conform to `BaseEngine`
- **Registry interface:** `list_models()` and `get_engine(model_id)`
- **History persistence:** `results/history_entries.json` via `/api/history`

The resulting system is modular, reproducible, and aligned with thesis comparison goals across classical ML, transformer baseline, and LLM approaches.
