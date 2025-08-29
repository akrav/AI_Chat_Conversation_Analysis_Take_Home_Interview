import os, json

from src.bertopic_pipeline import load_docs_from_jsonl, run_bertopic, save_outputs
from src.preprocess import preprocess_and_save_from_raw


def test_bertopic_pipeline_small(tmp_path):
    # Build a tiny raw file
    raw = tmp_path / "raw.jsonl"
    rows = [
        {"conversation_id": "1", "conversation": [{"content": "I love pizza"}]},
        {"conversation_id": "2", "conversation": [{"content": "Python coding"}]},
        {"conversation_id": "3", "conversation": [{"content": "Machine learning"}]},
        {"conversation_id": "4", "conversation": [{"content": "Deep learning"}]},
    ]
    with open(raw, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cleaned = preprocess_and_save_from_raw(str(raw), output_dir=str(tmp_path / "interim"))
    docs, conv_ids = load_docs_from_jsonl(cleaned)
    model, topics, probs = run_bertopic(docs, min_topic_size=2)
    assert len(topics) == len(docs)
    out = save_outputs(model, docs, topics, conv_ids, output_dir=str(tmp_path / "reports"))
    assert os.path.exists(out["topics_csv"]) and os.path.exists(out["doc_topics_csv"]) 