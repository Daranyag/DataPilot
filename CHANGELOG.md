# CHANGELOG

## Changes in this session
* **`app/main.py`**:
  * Fixed `SyntaxError` caused by unclosed `try` blocks and incorrect indentation.
  * Fixed `NameError` where `cols_to_drop` was passed into `evaluate_models` but no longer defined (now defaults to `[]`).
  * Fixed an `ImportError` where `get_clustering_models` was imported from `automl.registry` instead of its true home in `automl.models`.
  * Wrapped the AutoML execution in an exception handler that writes tracebacks to `logs/app.log` and displays them via a "Show technical details" expander to help identify future bugs easily.
* **`automl/fit_check.py`**:
  * Fixed `NameError` for `df` and `target` when running a provisional sample run via the "Run on a sample" button. They now safely pull from `st.session_state`.
* **`automl/leakage.py`**:
  * Cleaned up unused variable `n_cols` (Ruff `F841`).
* **`tests/test_smoke.py`**:
  * Added programmatic, mock-LLM tests to run the full AutoML pipeline against multiple edge-case datasets (missing values, multi-class rare, clustering, target leakage).
* **`tests/test_app_headless.py`**:
  * Added headless Streamlit UI testing (`AppTest`) to automatically click through the interface and verify no red error boxes appear. Extended to run Regression, Classification, and Clustering full UI workflows successfully.
* **`tests/test_import_contract.py`**:
  * Added an AST-based "import contract" test. It parses the entire `app/`, `automl/`, and `agent/` directories to dynamically verify that every function or object imported by any file actually exists at the source. This ensures future refactors will be caught if they break imports.
