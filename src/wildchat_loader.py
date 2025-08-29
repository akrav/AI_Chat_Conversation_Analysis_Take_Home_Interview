from __future__ import annotations

import json
import os
import time
import random
from typing import Optional, List, Dict

import requests

DATASET_DEFAULT = "allenai/WildChat"
CONFIG_DEFAULT = "default"
SPLIT_DEFAULT = "train"
MAX_ROWS_PER_QUERY = 100


def _http_fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
    max_retries: int = 5,
    backoff_base: float = 0.5,
    backoff_max: float = 10.0,
) -> List[Dict]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={requests.utils.quote(dataset, safe='')}"
        f"&config={requests.utils.quote(config, safe='')}"
        f"&split={requests.utils.quote(split, safe='')}"
        f"&offset={offset}&length={min(length, MAX_ROWS_PER_QUERY)}"
    )
    sess = session or requests.Session()
    attempt = 0
    while True:
        resp = sess.get(url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return [r["row"] for r in data.get("rows", [])]
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            attempt += 1
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_s = float(retry_after)
                except ValueError:
                    sleep_s = None
            else:
                jitter = random.uniform(0, 0.5)
                sleep_s = min(backoff_max, backoff_base * (2 ** (attempt - 1)) + jitter)
            time.sleep(sleep_s or 1.0)
            continue
        resp.raise_for_status()


def load_wildchat_subset(
    split: str = SPLIT_DEFAULT,
    sample_size: int = 1000,
    dataset_name: str = DATASET_DEFAULT,
    config: str = CONFIG_DEFAULT,
    prefer_http: bool = True,
    batch_size: int = 100,
    throttle_seconds: float = 0.15,
) -> List[Dict]:
    rows: List[Dict] = []
    offset = 0
    with requests.Session() as sess:
        while len(rows) < sample_size:
            need = min(batch_size, MAX_ROWS_PER_QUERY, sample_size - len(rows))
            batch = _http_fetch_rows(
                dataset_name,
                config,
                split,
                offset,
                need,
                session=sess,
            )
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
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
    batch_size: int = 100,
    throttle_seconds: float = 0.15,
) -> str:
    if output_dir is None:
        output_dir = os.path.join("data", "01_raw")
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_wildchat_subset(
        sample_size=sample_size,
        dataset_name=dataset_name,
        config=config,
        prefer_http=prefer_http,
        batch_size=batch_size,
        throttle_seconds=throttle_seconds,
    )
    output_file = os.path.join(
        output_dir,
        f"wildchat_sample_http_{sample_size}.jsonl",
    )
    return save_dataset_as_jsonl(dataset, output_file) 