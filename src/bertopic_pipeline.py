from __future__ import annotations

import json
import os
from typing import List, Tuple, Dict, Optional

import pandas as pd
from bertopic import BERTopic
from umap import UMAP


def _concat_conversation(turns: List[Dict]) -> str:
    parts: List[str] = []
    for t in turns or []:
        content = t.get("content")
        if content:
            parts.append(str(content))
    return " \n ".join(parts).strip()


def load_docs_from_jsonl(jsonl_path: str) -> Tuple[List[str], List[str]]:
    docs: List[str] = []
    conv_ids: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = _concat_conversation(row.get("conversation", []))
            if text:
                docs.append(text)
                conv_ids.append(str(row.get("conversation_id", "")))
    return docs, conv_ids


def run_bertopic(
    docs: List[str],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size: int = 5,
) -> Tuple[BERTopic, List[int], Optional[List[List[float]]]]:
    n_docs = max(2, len(docs))
    n_neighbors = max(2, min(10, n_docs - 1))
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, metric="cosine", random_state=42, init="random")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        umap_model=umap_model,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


def save_outputs(
    topic_model: BERTopic,
    docs: List[str],
    topics: List[int],
    conv_ids: List[str],
    output_dir: str = "reports",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    # Topic info
    topic_info = topic_model.get_topic_info()
    topics_csv = os.path.join(output_dir, "bertopic_topics.csv")
    topic_info.to_csv(topics_csv, index=False)

    # Document-topic mapping
    df = pd.DataFrame({"conversation_id": conv_ids, "topic": topics, "doc": docs})
    doc_topics_csv = os.path.join(output_dir, "bertopic_doc_topics.csv")
    df.to_csv(doc_topics_csv, index=False)

    return {"topics_csv": topics_csv, "doc_topics_csv": doc_topics_csv} 