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
    with open(llm_jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    recs: List[Dict] = []
    for r in rows:
        cid = r.get("conversation_id", None)
        la = r.get("llm_analysis", {})
        intent_obj = la.get("intent", {}) if isinstance(la.get("intent", {}), dict) else {}
        sent_obj = la.get("sentiment", {}) if isinstance(la.get("sentiment", {}), dict) else {}
        ent_obj = la.get("entities", {}) if isinstance(la.get("entities", {}), dict) else {}
        ent_list = ent_obj.get("entities", []) if isinstance(ent_obj.get("entities", []), list) else []
        # Ensure at least one entity row; rely on LLM v4 to provide it
        if not ent_list:
            ent_list = [{"text": None, "category": "other", "entity_sentiment": "neutral", "entity_sent_confidence": 0.5}]
        for item in ent_list:
            if isinstance(item, dict):
                e_text = item.get("text")
                e_cat = item.get("category")
                e_sent = item.get("entity_sentiment")
                e_conf = item.get("entity_sent_confidence")
            else:
                e_text = str(item)
                e_cat = "other"
                e_sent = "neutral"
                e_conf = 0.5
            recs.append({
                "conversation_id": cid,
                "llm_intent": intent_obj.get("intent"),
                "llm_intent_confidence": intent_obj.get("confidence"),
                "llm_sentiment": sent_obj.get("sentiment"),
                "llm_sent_confidence": sent_obj.get("confidence"),
                "llm_entity": e_text,
                "llm_entity_category": e_cat,
                "llm_entity_sentiment": e_sent,
                "llm_entity_sent_confidence": e_conf,
            })
    return pd.DataFrame(recs)


def build_unified_table(
    cleaned_jsonl_path: str,
    doc_topics_csv_path: str,
    topics_csv_path: str,
    rule_based_csv_path: str,
    llm_jsonl_path: str,
    output_csv_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    cleaned_df = _load_cleaned_df(cleaned_jsonl_path)

    doc_topics = pd.read_csv(doc_topics_csv_path)
    topics_df = pd.read_csv(topics_csv_path)
    topics_df = topics_df.rename(columns={"Topic": "topic_id", "Name": "topic_name"})
    doc_topics = doc_topics.rename(columns={"topic": "topic_id"})
    doc_topics = doc_topics.merge(topics_df[["topic_id", "topic_name"]], on="topic_id", how="left")

    rule_df = pd.read_csv(rule_based_csv_path) if os.path.exists(rule_based_csv_path) else pd.DataFrame()
    llm_df = _load_llm_df(llm_jsonl_path) if os.path.exists(llm_jsonl_path) else pd.DataFrame()

    base = cleaned_df.merge(doc_topics[["conversation_id", "topic_id", "topic_name"]], on="conversation_id", how="left")

    if not rule_df.empty:
        base = base.merge(rule_df, on="conversation_id", how="left")

    if not llm_df.empty:
        base = base.merge(llm_df, on="conversation_id", how="left")

    desired_cols = [
        "conversation_id",
        "topic_id",
        "topic_name",
        "vader_pos", "vader_neg", "vader_neu", "vader_compound",
        "tb_polarity", "tb_subjectivity",
        "llm_intent", "llm_intent_confidence", "llm_sentiment", "llm_sent_confidence",
        "llm_entity", "llm_entity_category", "llm_entity_sentiment", "llm_entity_sent_confidence",
        "full_text",
    ]
    cols = [c for c in desired_cols if c in base.columns]

    out_df = base[cols].copy()
    out_df.to_csv(output_csv_path, index=False)
    return output_csv_path


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Build unified analysis table")
    parser.add_argument("cleaned_jsonl", help="Path to cleaned JSONL (from preprocess)")
    parser.add_argument("doc_topics_csv", help="Path to BERTopic doc_topics CSV")
    parser.add_argument("topics_csv", help="Path to BERTopic topics CSV")
    parser.add_argument("rule_based_csv", help="Path to rule-based sentiment CSV")
    parser.add_argument("llm_jsonl", help="Path to LLM analysis results JSONL")
    parser.add_argument("--out", default="data/04_analysis/unified_table.csv", help="Output CSV path")
    args = parser.parse_args()
    print(build_unified_table(args.cleaned_jsonl, args.doc_topics_csv, args.topics_csv, args.rule_based_csv, args.llm_jsonl, args.out)) 