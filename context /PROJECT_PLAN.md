# Project Plan (Current and Practical)

**Project:** CyberTriage  
**Thesis:** MSc Computer Science, Nazarbayev University

This document is now a practical maintenance and completion plan aligned with the current codebase (FastAPI + React, registry-driven engines, local artifact workflow).

---

## A. What Is Already Implemented

### Application architecture
- FastAPI backend (`api/server.py`)
- React frontend (`frontend/`)
- Registry-driven engine loading (`src/engines/registry.py`)

### Model coverage
- Baseline models: `lr`, `svm`, `rf`, `nb`, `xgb`, `ensemble`, `codebert`
- LLM models: `ollama`, `qwen`, `finetuned`, `qwen_finetuned`

### Evaluation and persistence
- Batch evaluation scripts for baseline and LLM models
- Evaluation APIs and tables in UI
- Local history persistence in `results/history_entries.json`

---

## B. Current Workstream 1 - Documentation Consistency

### Goal
Ensure all context/thesis docs describe actual implementation, not historical Streamlit design.

### Tasks
1. Keep `context ` docs synchronized with code and README.
2. Ensure model lists match `registry.py`.
3. Ensure endpoint documentation matches `api/server.py`.
4. Keep figure/document references consistent with `thesis_images` generation scripts.

---

## C. Current Workstream 2 - Reproducibility and Evaluation Discipline

### Goal
Make every evaluation result traceable and repeatable.

### Tasks
1. Standardize run commands and batch sizes in docs.
2. Ensure result CSV schema remains stable (`accuracy`, `macro_f1`, `macro_precision`, `macro_recall`, `n_val`, `duration_sec`).
3. Validate that single-model re-runs overwrite/refresh rows correctly.
4. Keep local artifacts separated from source code via `.gitignore`.

---

## D. Current Workstream 3 - GitHub Preparation (Minimal Risk)

### Goal
Publish clean source code without large local assets.

### Tasks
1. Keep local-only assets ignored:
   - dataset files
   - model binaries/weights (`.gguf`, `.pkl`, `.safetensors`, etc.)
   - generated results
   - thesis images and scripts if intentionally excluded
2. Keep explanatory READMEs tracked in `dataset/`, `models/`, `results/`.
3. Avoid structural refactors that risk runtime breakage near submission deadline.

---

## E. Final Thesis Readiness Checklist

- [x] Multi-model baseline + LLM architecture implemented
- [x] Registry-based extensibility implemented
- [x] Ingestion + result + evaluation + history flow implemented
- [x] Persistent local history implemented
- [x] Evaluation scripts integrated into backend endpoints
- [ ] Final documentation consistency pass
- [ ] Final reproducibility pass
- [ ] Final thesis narrative alignment with implemented system

---

## F. Extension Rule (for Future Models)

To add a new model safely:
1. Implement an engine class under `src/engines/`.
2. Register it in `src/engines/registry.py`.
3. If needed, add corresponding evaluation script and include it in `api/server.py`.
4. Update README and context docs.

No major frontend architecture change is required for model insertion.
