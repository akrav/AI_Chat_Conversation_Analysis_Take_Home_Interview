from src.text_cleaning import clean_text, clean_conversation_list
from src.preprocess import preprocess_and_save
import os


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


def test_preprocess_and_save(tmp_path):
    out_dir = tmp_path / "interim"
    path = preprocess_and_save(sample_size=5, output_dir=str(out_dir))
    assert os.path.exists(path)
    # file has 5 lines
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 5 