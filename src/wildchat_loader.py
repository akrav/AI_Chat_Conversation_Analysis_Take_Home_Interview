from __future__ import annotations

import json
import os
from typing import Optional, List, Dict

import requests

DATASET_DEFAULT = "allenai/WildChat"
CONFIG_DEFAULT = "default"
SPLIT_DEFAULT = "train"


def _http_fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> List[Dict]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={requests.utils.quote(dataset, safe='')}"
        f"&config={requests.utils.quote(config, safe='')}"
        f"&split={requests.utils.quote(split, safe='')}"
        f"&offset={offset}&length={length}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    rows = [r["row"] for r in data.get("rows", [])]
    return rows


def load_wildchat_subset(
    split: str = SPLIT_DEFAULT,
    sample_size: int = 1000,
    dataset_name: str = DATASET_DEFAULT,
    config: str = CONFIG_DEFAULT,
    prefer_http: bool = True,
):
    # HTTP-only implementation via datasets-server
    rows: List[Dict] = []
    offset = 0
    while len(rows) < sample_size:
        need = min(100, sample_size - len(rows))
        batch = _http_fetch_rows(dataset_name, config, split, offset, need)
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows


def save_dataset_as_jsonl(dataset, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def load_and_save_raw(
    sample_size: int = 1000,
    output_dir: Optional[str] = None,
    dataset_name: str = DATASET_DEFAULT,
    config: str = CONFIG_DEFAULT,
    prefer_http: bool = True,
) -> str:
    if output_dir is None:
        output_dir = os.path.join("data", "01_raw")
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_wildchat_subset(
        sample_size=sample_size,
        dataset_name=dataset_name,
        config=config,
        prefer_http=prefer_http,
    )
    output_file = os.path.join(
        output_dir,
        f"wildchat_sample_http_{sample_size}.jsonl",
    )
    return save_dataset_as_jsonl(dataset, output_file) 