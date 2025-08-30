import os

def test_project_structure_exists():
    required_paths = [
        "data/01_raw",
        "data/02_interim",
        "data/03_processed",
        "notebooks",
        "src",
        "tests",
        "reports/images",
    ]
    for path in required_paths:
        assert os.path.isdir(path), f"Missing required directory: {path}"


def test_imports_work():
    import pandas as pd  # noqa: F401
    import numpy as np  # noqa: F401
    import datasets  # noqa: F401
    assert True 