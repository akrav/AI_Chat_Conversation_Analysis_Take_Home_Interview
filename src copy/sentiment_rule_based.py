from __future__ import annotations

import json
import os
from typing import List, Dict, Optional

import pandas as pd

# NLTK VADER
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    nltk = None  # type: ignore
    SentimentIntensityAnalyzer = None  # type: ignore

# TextBlob
try:
    from textblob import TextBlob
except Exception:  # pragma: no cover
    TextBlob = None  # type: ignore


def ensure_vader_downloaded():  # pragma: no cover
    if nltk is None:
        raise RuntimeError("nltk not available")
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')


def vader_scores(text: str) -> Dict[str, float]:
    ensure_vader_downloaded()
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text or "")


def textblob_scores(text: str) -> Dict[str, float]:
    if TextBlob is None:
        raise RuntimeError("textblob not available")
    tb = TextBlob(text or "")
    polarity, subjectivity = tb.sentiment.polarity, tb.sentiment.subjectivity
    return {"polarity": polarity, "subjectivity": subjectivity}


def _extract_text_from_row(row: dict) -> str:
    conv = row.get("conversation", [])
    parts = [t.get("content", "") for t in conv if t.get("content")]
    return "\n".join(parts)


def batch_process_subset(
    subset_jsonl_path: str,
    output_csv_path: str,
    max_rows: Optional[int] = None,
) -> str:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    records: List[Dict] = []
    count = 0
    with open(subset_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cid = row.get("conversation_id", "")
            text = _extract_text_from_row(row)
            vs = vader_scores(text)
            tb = textblob_scores(text)
            rec = {
                "conversation_id": cid,
                "vader_pos": vs.get("pos", 0.0),
                "vader_neg": vs.get("neg", 0.0),
                "vader_neu": vs.get("neu", 0.0),
                "vader_compound": vs.get("compound", 0.0),
                "tb_polarity": tb.get("polarity", 0.0),
                "tb_subjectivity": tb.get("subjectivity", 0.0),
            }
            records.append(rec)
            count += 1
            if max_rows is not None and count >= max_rows:
                break
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path 