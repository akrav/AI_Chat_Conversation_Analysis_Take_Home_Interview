from __future__ import annotations

import json
import os
import csv
import re
from typing import List, Tuple, Dict, Optional

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sklearn_text
import hdbscan


_NONALNUM_RX = re.compile(r"[A-Za-z0-9]")
_WORD_RX = re.compile(r"[A-Za-z]{2,}")
_STOPWORDS = set(sklearn_text.ENGLISH_STOP_WORDS)


def _concat_conversation(turns: List[Dict]) -> str:
    parts: List[str] = []
    for t in turns or []:
        content = t.get("content")
        if content:
            parts.append(str(content))
    return " \n ".join(parts).strip()


def _has_enough_content(text: str, min_nonstop_tokens: int = 2) -> bool:
    if not text or not text.strip():
        return False
    if _NONALNUM_RX.search(text) is None:
        return False
    tokens = [tok for tok in _WORD_RX.findall(text.lower()) if tok not in _STOPWORDS]
    return len(tokens) >= min_nonstop_tokens


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


def load_docs_from_pairs_jsonl(jsonl_path: str, role_filter: Optional[str] = None) -> Tuple[List[str], List[Dict[str, str]]]:
    docs: List[str] = []
    meta: List[Dict[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            role = r.get("role")
            if role_filter and role != role_filter:
                continue
            text = r.get("text", "")
            if not _has_enough_content(text):
                continue
            cid = str(r.get("conversation_id", ""))
            turn = str(r.get("turn_index", ""))
            doc_id = f"{cid}:{turn}:{role}"
            docs.append(text)
            meta.append({"doc_id": doc_id, "conversation_id": cid, "role": role, "turn_index": turn})
    return docs, meta


def run_bertopic(
    docs: List[str],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size: int = 5,
) -> Tuple[BERTopic, List[int], Optional[List[List[float]]]]:
    n_docs = max(2, len(docs))
    # Relax UMAP neighborhood to reduce outliers
    n_neighbors = max(10, min(30, n_docs - 1))
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, metric="cosine", random_state=42, init="random")

    # Custom vectorizer to avoid stopword topics
    domain_stop = {
        "the","and","or","to","of","in","on","for","with","a","an","is","are","was","were","be","been","it","its","this","that","these","those","as","at","by","from","into","out","up","down","over","under","you","your","i","we","our","they","their","he","she","his","her","them","do","does","did","can","could","would","should","may","might","will","just","also","like","get","got","make","made","use","used","using","one","two","three","etc"
    }
    stop = list(sklearn_text.ENGLISH_STOP_WORDS.union(domain_stop))
    vectorizer_model = CountVectorizer(stop_words=stop, ngram_range=(1, 3), min_df=5, max_df=0.9)

    # Relax HDBSCAN clustering to assign more points to clusters
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=max(5, min_topic_size),
        min_samples=1,
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    rep = KeyBERTInspired()
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=rep,
        calculate_probabilities=True,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(docs)

    # Reassign outliers to closest topics
    try:
        new_topics = topic_model.reduce_outliers(docs, topics, probabilities=probs)
        topic_model.update_topics(docs, topics=new_topics)
        topics = new_topics
    except Exception:
        # If reduce_outliers fails for any reason, keep original topics
        pass

    return topic_model, topics, probs


def save_outputs(
    topic_model: BERTopic,
    docs: List[str],
    topics: List[int],
    identities: List[Dict[str, str]],
    output_dir: str = "reports",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    # Topic info normalized
    ti = topic_model.get_topic_info().rename(columns={"Topic": "topic_id", "Count": "count", "Name": "topic_name"})
    # Sanitize topic_name (remove newlines)
    if "topic_name" in ti.columns:
        ti["topic_name"] = ti["topic_name"].astype(str).str.replace("\r", " ", regex=False).str.replace("\n", " ", regex=False)
    # Build top_words as JSON array of strings (top-10)
    top_rows = []
    for _, row in ti.iterrows():
        try:
            tid = int(row["topic_id"]) if pd.notnull(row["topic_id"]) else -1
        except Exception:
            tid = -1
        if tid == -1:
            top_words = []
        else:
            tuples = topic_model.get_topic(tid) or []
            top_words = [w for w, score in tuples[:10]]
        top_rows.append(top_words)
    ti["top_words"] = [json.dumps(x, ensure_ascii=False) for x in top_rows]
    # Enforce dtypes and column order
    ti["count"] = pd.to_numeric(ti["count"], errors="coerce").fillna(0).astype(int)
    ti["topic_id"] = pd.to_numeric(ti["topic_id"], errors="coerce").fillna(-1).astype(int)
    ti = ti[["topic_id", "topic_name", "count", "top_words"]]
    topics_csv = os.path.join(output_dir, "bertopic_topics.csv")
    ti.to_csv(topics_csv, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")

    # Document-topic mapping with stable schema
    df = pd.DataFrame(identities)
    df["topic"] = topics
    df["doc"] = docs
    df = df[["doc_id", "conversation_id", "role", "turn_index", "topic", "doc"]]
    doc_topics_csv = os.path.join(output_dir, "bertopic_doc_topics.csv")
    df.to_csv(doc_topics_csv, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")

    return {"topics_csv": topics_csv, "doc_topics_csv": doc_topics_csv} 