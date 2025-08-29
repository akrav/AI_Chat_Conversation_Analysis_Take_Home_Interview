from __future__ import annotations

import os
import json
from typing import Optional

from src.text_cleaning import clean_conversation_list


def preprocess_and_save_from_raw(raw_path: str, output_dir: Optional[str] = None) -> str:
    if output_dir is None:
        output_dir = os.path.join("data", "02_interim")
    os.makedirs(output_dir, exist_ok=True)

    cleaned_rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            r2 = dict(row)
            if "conversation" in r2:
                r2["conversation"] = clean_conversation_list(r2["conversation"])
            cleaned_rows.append(r2)

    base = os.path.splitext(os.path.basename(raw_path))[0]
    out_path = os.path.join(output_dir, f"{base}_cleaned.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path 