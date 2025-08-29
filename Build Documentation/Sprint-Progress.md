Sprint 1 Progress

- Ticket-1001: Project setup and initial environment
  - Status: Completed
  - Actions:
    - Created directories: data/{01_raw,02_interim,03_processed}, notebooks, src, tests, reports/images
    - Created Python virtual environment and installed base packages (pandas, numpy, datasets, pytest)
    - Added sanity tests for structure and imports
    - Test run: 2 tests passed (via `python -m pytest -q`)
  - Next: Proceed to Ticket-1002

- Ticket-1002: Programmatically load the `allenai/WildChat` dataset
  - Status: Completed (HTTP-only path)
  - Actions:
    - Implemented HTTP-only loader using Hugging Face datasets-server (`rows` endpoint)
    - Refactored tests to HTTP-only; removed token-based skips
    - Test run: 5 passed (via `python -m pytest -q`)
    - Saved sample to `data/01_raw/wildchat_sample_http_100.jsonl`
  - Notes:
    - No HF tokens required for this project. Endpoints documented in Troubleshooting.
  - Next: Ticket-1003 (DataFrame conversion and EDA)
