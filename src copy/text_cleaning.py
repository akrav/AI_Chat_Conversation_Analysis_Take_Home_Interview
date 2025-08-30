from __future__ import annotations

import re
from typing import Dict, List

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Keep word chars, whitespace, and common punctuation; remove other special chars
_KEEP_CHARS_RE = re.compile(r"[^\w\s\.,!\?\'\"\-]", re.UNICODE)


def clean_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = _URL_RE.sub(" ", s)
    s = _KEEP_CHARS_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    s = s.lower()
    return s


def clean_conversation_list(conversation: List[Dict]) -> List[Dict]:
    cleaned: List[Dict] = []
    for turn in conversation or []:
        new_turn = dict(turn)
        new_turn["content"] = clean_text(turn.get("content", ""))
        cleaned.append(new_turn)
    return cleaned 