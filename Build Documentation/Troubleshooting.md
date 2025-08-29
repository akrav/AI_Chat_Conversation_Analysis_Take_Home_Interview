Issue: WildChat-nontoxic is a gated dataset on Hugging Face

- Symptom: `DatasetNotFoundError` or RuntimeError indicating gated dataset when calling `load_dataset("allenai/WildChat-nontoxic")`.
- Root cause: Access requires accepting terms and authenticating with a Hugging Face token.
- Resolution:
  1. Visit the dataset page and request/accept access.
  2. Set an environment variable before running code/tests, e.g. `export HUGGINGFACE_HUB_TOKEN=YOUR_TOKEN`.
  3. Re-run with `python -m pytest -q` or your script; tests marked `internet`/`slow` will run when token is present, otherwise skip.
- Status: Skipped in CI/local runs without token; loader supports token via env.

Issue: `pytest` from global path did not see venv-installed packages

- Symptom: Running `pytest` invoked Homebrew's pytest, causing `ModuleNotFoundError: datasets`.
- Root cause: Global pytest binary was used instead of venv Python.
- Resolution: Use `python -m pytest -q` to ensure the venv interpreter runs tests.
- Status: Resolved. Tests pass using `python -m pytest -q`.

Issue: Need non-gated access to sample rows for development

- Symptom: Could not load with `datasets` library without token, blocking local development.
- Root cause: Gated dataset access via library; however, datasets-server and parquet endpoints are public for WildChat.
- Resolution: Implemented HTTP fallback using `https://datasets-server.huggingface.co/rows` to pull small batches without token. Added tests to exercise fallback.
- Status: Resolved. Use `prefer_http=True` in loader functions to fetch samples without a token.

Issue: HTTP-only data access design

- Decision: For this project, only public HTTP endpoints will be used. No HF tokens.
- Endpoints used:
  - Rows API: `https://datasets-server.huggingface.co/rows?dataset=allenai/WildChat&config=default&split=train&offset=0&length=K`
  - Splits API: `https://datasets-server.huggingface.co/splits?dataset=allenai/WildChat`
  - Parquet listing: `https://huggingface.co/api/datasets/allenai/WildChat/parquet/default/train`
- Implementation: `src/wildchat_loader.py` fetches batches of rows via Rows API and writes JSONL.
- Status: Working. Tests pass using HTTP-only path.

Issue: 422/429 from datasets-server when fetching large samples

- Symptom: `HTTPError: 422 Unprocessable Entity` for length>100 and `429 Too Many Requests` during large pulls.
- Root cause: API enforces max `length=100` and rate limits repeated calls.
- Resolution:
  - Clamp per-request length to 100.
  - Add exponential backoff and respect `Retry-After`.
  - Throttle between requests (`throttle_seconds`).
- Status: Implemented in `wildchat_loader.py` (HTTP-only). For very large pulls, increase throttle to reduce 429s.

Issue: Text cleaning removes URLs and special characters

- Symptom: URLs remain or excessive whitespace persists.
- Root cause: Insufficient regex coverage.
- Resolution: Added URL removal, special char filtering, and whitespace collapsing in `clean_text`.
- Status: Covered by `tests/test_text_cleaning.py`.

Issue: Preprocessing line count mismatch

- Symptom: Saved JSONL does not match requested sample size.
- Root cause: Underlying HTTP pulls may return fewer rows if the server throttles.
- Resolution: Tests use small sizes; production runs should check counts and loop or adjust throttle if needed.
- Status: Documented; loader supports throttling and backoff.

Issue: BERTopic runtime and sklearn warnings

- Symptom: Long runtime on first run; FutureWarning from sklearn about `force_all_finite`.
- Root cause: Model downloads and numeric checks in newer sklearn.
- Resolution: Acceptable for analysis; warning harmless. Consider caching embeddings or reducing sample size for quicker iterations.
- Status: Noted; tests pass with warnings.
