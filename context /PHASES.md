# Project Roadmap and Execution Phases (Current Status)

This document reflects the implemented architecture and current project maturity.

**Project:** CyberTriage  
**Timeline target:** Jan 20, 2026 - Mar 31, 2026

---

## Phase 1 - Foundation and Baselines
**Status:** Completed

### Delivered
- Baseline model training/evaluation pipeline
- Registry-based model integration
- Baseline model support: LR, SVM, RF, NB, XGB, Ensemble
- CodeBERT integrated as baseline model
- FastAPI backend and React frontend runtime established

---

## Phase 2 - LLM Core
**Status:** Completed

### Delivered
- Zero-shot LLM support (Llama and Qwen via Ollama)
- Fine-tuned GGUF support (Llama and Qwen)
- Robust parsing/normalization of model outputs
- Evaluation scripts for all LLM variants

---

## Phase 3 - End-to-End Product Integration
**Status:** Completed

### Delivered
- Full ingestion flow with model selection
- Separate result rendering for baseline and LLM outputs
- Batch evaluation screen and backend trigger endpoint
- Local history persistence (`results/history_entries.json`)
- History screen improvements (view/delete/search/layout)

---

## Phase 4 - Evaluation and Thesis Assets
**Status:** In progress

### Delivered
- Batch evaluation scripts with metrics CSV outputs
- Comparative tables and figure-generation scripts
- Updated baseline and LLM evaluation visualizations

### Remaining focus
- Final thesis narrative alignment with implemented architecture
- Final reproducibility checks before repository publication

---

## Current Immediate Priorities

1. Keep all architecture and context docs synchronized with codebase reality.
2. Finalize repository cleanup for GitHub (ignore large local artifacts).
3. Stabilize evaluation reproducibility and final thesis figures/text references.