from __future__ import annotations

import os
import json
from typing import Optional

from src.wildchat_loader import load_wildchat_subset
from src.text_cleaning import clean_conversation_list


def preprocess_and_save(sample_size: int = 1000, output_dir: Optional[str] = None) -> str:
    if output_dir is None:
        output_dir = os.path.join("data", "02_interim")
    os.makedirs(output_dir, exist_ok=True)

    rows = load_wildchat_subset(sample_size=sample_size, prefer_http=True)
    cleaned_rows = []
    for r in rows:
        r2 = dict(r)
        if "conversation" in r2:
            r2["conversation"] = clean_conversation_list(r2["conversation"])
        cleaned_rows.append(r2)

    out_path = os.path.join(output_dir, f"wildchat_cleaned_{sample_size}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path 