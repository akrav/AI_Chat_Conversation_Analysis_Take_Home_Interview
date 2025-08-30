from src.text_cleaning import clean_text, clean_conversation_list
from src.preprocess import preprocess_and_save_from_raw
import os, json


def test_clean_text_basic():
    s = "Hello!!!   Visit https://example.com NOW!!!"
    out = clean_text(s)
    assert "http" not in out
    assert out == out.lower()
    assert "hello" in out


def test_clean_conversation_list():
    conv = [{"content": "Hi :)"}, {"content": None}]
    cleaned = clean_conversation_list(conv)
    assert len(cleaned) == 2
    assert cleaned[0]["content"] == "hi"
    assert cleaned[1]["content"] == ""


def test_preprocess_and_save_from_raw(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {"conversation": [{"content": "Hello URL https://x.y", "role": "user"}]},
        {"conversation": [{"content": "World!", "role": "assistant"}]},
    ]
    with open(raw, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = preprocess_and_save_from_raw(str(raw), output_dir=str(tmp_path / "interim"))
    assert os.path.exists(out)
    with open(out, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2 