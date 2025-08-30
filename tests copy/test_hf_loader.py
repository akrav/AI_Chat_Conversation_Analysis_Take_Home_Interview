import os
import pytest

from src.hf_loader import load_wildchat_ds, get_hf_token


@pytest.mark.skipif(not get_hf_token(), reason="HF token not provided; skipping HF loader test")
def test_hf_loader_train_split():
    ds = load_wildchat_ds(split="train")
    assert len(ds) > 0 