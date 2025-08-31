from __future__ import annotations

import json
import os
from typing import List, Dict, Optional

import pandas as pd


def _load_cleaned_df(cleaned_jsonl_path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    with open(cleaned_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "conversation_id" not in df.columns:
        df["conversation_id"] = None
    if "conversation" in df.columns:
        df["full_text"] = df["conversation"].apply(
            lambda turns: " ".join([t.get("content", "") for t in (turns or [])]) if isinstance(turns, list) else None
        )
    return df


def _load_llm_df(llm_jsonl_path: str) -> pd.DataFrame:
    # Expect a FLAT JSONL from llm_analysis.py with one row per entity and columns:
    # conversation_id, llm_intent, llm_intent_confidence, llm_sentiment, llm_sent_confidence,
    # llm_entity, llm_entity_category, llm_entity_sentiment, llm_entity_sent_confidence
    rows: List[Dict] = []
    with open(llm_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Ensure expected columns exist even if empty
    for col in [
        "conversation_id",
        "llm_intent", "llm_intent_confidence",
        "llm_sentiment", "llm_sent_confidence",
        "llm_entity", "llm_entity_category", "llm_entity_sentiment", "llm_entity_sent_confidence",
    ]:
        if col not in df.columns:
            df[col] = None
    return df[[
        "conversation_id",
        "llm_intent", "llm_intent_confidence",
        "llm_sentiment", "llm_sent_confidence",
        "llm_entity", "llm_entity_category", "llm_entity_sentiment", "llm_entity_sent_confidence",
    ]]


def _load_pairs_df(pairs_jsonl_path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    with open(pairs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # reshape to one row per pair with user_text and assistant_text
    u = df[(df["role"] == "user") & df["pair_id"].notna()][["conversation_id", "pair_id", "text"]].rename(columns={"text": "user_text"})
    a = df[(df["role"] == "assistant") & df["pair_id"].notna()][["conversation_id", "pair_id", "text"]].rename(columns={"text": "assistant_text"})
    if u.empty and a.empty:
        return pd.DataFrame(columns=["conversation_id", "pair_id", "user_text", "assistant_text"])
    merged = pd.merge(u, a, on=["conversation_id", "pair_id"], how="outer")
    return merged


def build_unified_table(
    cleaned_jsonl_path: str,
    doc_topics_csv_path: str,
    topics_csv_path: str,
    rule_based_csv_path: str,
    llm_jsonl_path: str,
    output_csv_path: str,
    pairs_jsonl_path: str | None = None,
) -> str:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    cleaned_df = _load_cleaned_df(cleaned_jsonl_path)

    doc_topics = pd.read_csv(doc_topics_csv_path)
    topics_df = pd.read_csv(topics_csv_path)
    topics_df = topics_df.rename(columns={"Topic": "topic_id", "Name": "topic_name"})
    doc_topics = doc_topics.rename(columns={"topic": "topic_id"})
    doc_topics = doc_topics.merge(topics_df[["topic_id", "topic_name"]], on="topic_id", how="left")

    rule_df = pd.read_csv(rule_based_csv_path) if os.path.exists(rule_based_csv_path) else pd.DataFrame()
    # Support both new and legacy LLM filenames
    if not os.path.exists(llm_jsonl_path):
        alt = os.path.join(os.path.dirname(llm_jsonl_path), 'llm_analysis_results.jsonl')
        if os.path.exists(alt):
            llm_jsonl_path = alt
    llm_df = _load_llm_df(llm_jsonl_path) if os.path.exists(llm_jsonl_path) else pd.DataFrame()
    pairs_df = _load_pairs_df(pairs_jsonl_path) if pairs_jsonl_path and os.path.exists(pairs_jsonl_path) else pd.DataFrame()

    base = cleaned_df.merge(doc_topics[["conversation_id", "topic_id", "topic_name"]], on="conversation_id", how="left")

    if not rule_df.empty:
        base = base.merge(rule_df, on="conversation_id", how="left")

    if not llm_df.empty:
        base = base.merge(llm_df, on="conversation_id", how="left")

    if not pairs_df.empty:
        base = base.merge(pairs_df, on="conversation_id", how="left")

    desired_cols = [
        "conversation_id",
        "topic_id",
        "topic_name",
        "vader_pos", "vader_neg", "vader_neu", "vader_compound",
        "tb_polarity", "tb_subjectivity",
        "llm_intent", "llm_intent_confidence", "llm_sentiment", "llm_sent_confidence",
        "llm_entity", "llm_entity_category", "llm_entity_sentiment", "llm_entity_sent_confidence",
        "pair_id", "user_text", "assistant_text",
        "full_text",
    ]
    cols = [c for c in desired_cols if c in base.columns]

    out_df = base[cols].copy()
    # Write CSV
    out_df.to_csv(output_csv_path, index=False)

    # Also write Excel next to CSV for easier viewing
    try:
        xlsx_path = os.path.splitext(output_csv_path)[0] + ".xlsx"
        # Use openpyxl engine if available; pandas will fallback if installed
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="unified_table")
    except Exception:
        # Non-fatal if engine not installed; CSV remains primary artifact
        pass

    return output_csv_path 


def build_unified_pairs_table(pairs_jsonl_path: str, output_csv_path: str) -> str:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    pairs_df = _load_pairs_df(pairs_jsonl_path)
    pairs_df.to_csv(output_csv_path, index=False)
    return output_csv_path


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Build unified analysis table")
    subparsers = parser.add_subparsers(dest="cmd")

    p_entity = subparsers.add_parser("entity")
    p_entity.add_argument("cleaned_jsonl", help="Path to cleaned JSONL (from preprocess)")
    p_entity.add_argument("doc_topics_csv", help="Path to BERTopic doc_topics CSV")
    p_entity.add_argument("topics_csv", help="Path to BERTopic topics CSV")
    p_entity.add_argument("rule_based_csv", help="Path to rule-based sentiment CSV")
    p_entity.add_argument("llm_jsonl", help="Path to LLM analysis results JSONL")
    p_entity.add_argument("--out", default="data/04_analysis/unified_table.csv", help="Output CSV path")

    p_pairs = subparsers.add_parser("pairs")
    p_pairs.add_argument("pairs_jsonl", help="Path to pairs JSONL")
    p_pairs.add_argument("--out", default="data/04_analysis/unified_pairs.csv", help="Output CSV path")

    args = parser.parse_args()
    if args.cmd == "pairs":
        print(build_unified_pairs_table(args.pairs_jsonl, args.out))
    else:
        print(build_unified_table(args.cleaned_jsonl, args.doc_topics_csv, args.topics_csv, args.rule_based_csv, args.llm_jsonl, args.out)) 