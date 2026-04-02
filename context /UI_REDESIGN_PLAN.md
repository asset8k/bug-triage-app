# CyberTriage Frontend Plan (Implemented State + Maintenance)

This file now documents the current frontend architecture and maintenance rules.

---

## 1. Current Frontend Stack

- React 18
- Vite
- Tailwind CSS
- React Router
- Context-based state (auth/history) with backend-backed history persistence

Runtime entry: `frontend/` served via `npm run dev`.

---

## 2. Current UX Structure

### Main screens
- `LoginScreen`
- `IngestScreen`
- `ResultRouter` (`BaselineResult` or `LLMResult`)
- `EvaluationScreen`
- `HistoryScreen`

### Shared layout
- Persistent top navigation (`TopBar`)
- Unified project branding with custom `AppLogo`
- Light theme and thesis-friendly styling

---

## 3. Backend Contract in Use

Frontend relies on:
- `GET /api/models`
- `POST /api/predict`
- `GET /api/evaluation`
- `POST /api/evaluation/run`
- `POST /api/parse-document`
- `GET /api/history`
- `POST /api/history`
- `DELETE /api/history/{entry_id}`

History is no longer session-only; it is persisted by backend in `results/history_entries.json`.

---

## 4. Model Presentation Rules (Current)

### Baseline list in UI
- `lr`, `svm`, `rf`, `nb`, `xgb`, `ensemble`, `codebert`

### LLM list in UI
- `ollama`, `qwen`, `finetuned`, `qwen_finetuned`

CodeBERT must remain categorized with baseline models in both ingestion and evaluation flows.

---

## 5. Existing UX Decisions to Preserve

1. Academic login subtitle and neutral branding language.
2. Consistent button styling in history actions (View/Delete).
3. Baseline history table optimized for width and readability.
4. Working cross-field history search.
5. Browser tab icon linked to `frontend/favicon.svg`.

---

## 6. Maintenance Checklist

When updating frontend:
1. Keep model grouping synced with `src/engines/registry.py`.
2. Keep API client methods aligned with backend response schema.
3. Preserve history persistence behavior (API-first, resilient fallback only if needed).
4. Avoid introducing presentation text that is informal or non-academic.
5. Re-check routing and result rendering for both baseline and LLM paths.

---

## 7. Scope Boundaries

Frontend changes should not alter:
- model inference internals in `src/engines/`
- training/evaluation script logic in `scripts/`
- artifact formats in `results/` unless coordinated with backend
