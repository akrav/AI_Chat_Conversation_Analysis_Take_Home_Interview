import os

import pytest

from src.wildchat_loader import (
    load_wildchat_subset,
    save_dataset_as_jsonl,
    load_and_save_raw,
    DATASET_DEFAULT,
)


@pytest.mark.internet
def test_load_wildchat_subset_http_small_sample():
    rows = load_wildchat_subset(sample_size=5, prefer_http=True, dataset_name=DATASET_DEFAULT)
    assert isinstance(rows, list)
    assert len(rows) == 5
    assert isinstance(rows[0], dict)


def test_save_dataset_as_jsonl(tmp_path):
    sample = [{"id": i, "text": f"hello {i}"} for i in range(3)]
    out = tmp_path / "sample.jsonl"
    path = save_dataset_as_jsonl(sample, str(out))
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3


@pytest.mark.internet
def test_load_and_save_raw_http_creates_file(tmp_path):
    out_dir = tmp_path / "raw"
    path = load_and_save_raw(sample_size=5, output_dir=str(out_dir), prefer_http=True)
    assert os.path.exists(path)
    assert path.endswith(".jsonl") 