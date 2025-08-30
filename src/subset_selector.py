from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd


def load_cleaned_index(cleaned_jsonl_path: str) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    with open(cleaned_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cid = str(row.get("conversation_id", ""))
            if cid:
                index[cid] = row
    return index


def select_subset(
    topics_csv_path: str,
    doc_topics_csv_path: str,
    cleaned_jsonl_path: str,
    output_path: str,
    top_n_topics: int = 10,
    max_docs_per_topic: int = 50,
    random_state: int = 42,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    topics_df = pd.read_csv(topics_csv_path)
    # topics_df columns typically include: Topic, Count, Name
    topics_df = topics_df[topics_df["Topic"] != -1]
    top_topics = (
        topics_df.sort_values("Count", ascending=False)
        .head(top_n_topics)["Topic"]
        .tolist()
    )

    doc_topics = pd.read_csv(doc_topics_csv_path)
    doc_topics = doc_topics[doc_topics["topic"].isin(top_topics)]

    sampled_rows: List[dict] = []
    cleaned_idx = load_cleaned_index(cleaned_jsonl_path)

    for topic_id in top_topics:
        topic_docs = doc_topics[doc_topics["topic"] == topic_id]
        if len(topic_docs) > max_docs_per_topic:
            topic_docs = topic_docs.sample(n=max_docs_per_topic, random_state=random_state)
        for _, row in topic_docs.iterrows():
            cid = str(row["conversation_id"]) if "conversation_id" in row else None
            if cid and cid in cleaned_idx:
                sampled_rows.append(cleaned_idx[cid])

    with open(output_path, "w", encoding="utf-8") as f:
        for r in sampled_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return output_path 