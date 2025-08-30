import os, json
import pandas as pd

from src.subset_selector import select_subset


def test_select_subset_small(tmp_path):
    # Create tiny topics CSV
    topics = pd.DataFrame({"Topic": [0, 1, -1], "Count": [10, 5, 100], "Name": ["t0", "t1", "-1"]})
    topics_csv = tmp_path / "topics.csv"
    topics.to_csv(topics_csv, index=False)

    # Create doc_topics CSV
    doc_topics = pd.DataFrame({
        "conversation_id": ["a", "b", "c", "d"],
        "topic": [0, 0, 1, -1],
        "doc": ["da", "db", "dc", "dd"],
    })
    doc_topics_csv = tmp_path / "doc_topics.csv"
    doc_topics.to_csv(doc_topics_csv, index=False)

    # Create cleaned JSONL
    cleaned = tmp_path / "cleaned.jsonl"
    rows = [
        {"conversation_id": "a", "conversation": [{"content": "hello"}]},
        {"conversation_id": "b", "conversation": [{"content": "world"}]},
        {"conversation_id": "c", "conversation": [{"content": "other"}]},
    ]
    with open(cleaned, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out_path = tmp_path / "subset.jsonl"
    out = select_subset(str(topics_csv), str(doc_topics_csv), str(cleaned), str(out_path), top_n_topics=1, max_docs_per_topic=2)
    assert os.path.exists(out)
    with open(out, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Only topic 0 selected, conversation ids a and b expected
    assert len(lines) == 2 