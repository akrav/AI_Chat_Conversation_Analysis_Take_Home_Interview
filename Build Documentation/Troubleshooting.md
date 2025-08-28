Issue: `pytest` from global path did not see venv-installed packages

- Symptom: Running `pytest` invoked Homebrew's pytest, causing `ModuleNotFoundError: datasets`.
- Root cause: Global pytest binary was used instead of venv Python.
- Resolution: Use `python -m pytest -q` to ensure the venv interpreter runs tests.
- Status: Resolved. Tests pass using `python -m pytest -q`.
