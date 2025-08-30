from __future__ import annotations

import os
import json
from typing import Optional

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

# Load .env at import time
load_dotenv()


def get_hf_token() -> Optional[str]:
    return os.getenv("HUGGINGFACE_HUB_TOKEN")


def load_wildchat_ds(split: str = "train", token: Optional[str] = None):
    token = token or get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    ds = load_dataset("allenai/WildChat", split=split, token=token)
    return ds


def load_filter_to_dataframe(
    language: str = "English",
    toxic: bool = False,
    limit: Optional[int] = None,
    split: str = "train",
    token: Optional[str] = None,
) -> pd.DataFrame:
    token = token or get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    ds = load_dataset("allenai/WildChat", split=split, token=token)
    df = pd.DataFrame(ds)
    mask = (df["language"] == language) & (df["toxic"] == False)
    filtered = df.loc[mask]
    if limit is not None:
        filtered = filtered.head(limit)
    return filtered.reset_index(drop=True)


def save_dataframe_jsonl(df: pd.DataFrame, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return output_path 