from __future__ import annotations

import json
import os
from typing import List, Dict, Optional


def _is_question(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return t.endswith("?") or t.lower().startswith(("how ", "what ", "why ", "when ", "where ", "can ", "could ", "would ", "should "))


def normalize_conversation(conversation_id: str, turns: List[Dict]) -> List[Dict]:
    records: List[Dict] = []
    pair_counter = 0
    pending_user_idx: Optional[int] = None
    for idx, t in enumerate(turns or []):
        role = t.get("role")
        text = t.get("content", "")
        rec: Dict = {
            "conversation_id": conversation_id,
            "turn_index": idx,
            "role": role,
            "text": text,
            "pair_id": None,
            "is_question": False,
            "is_answer": False,
        }
        if role == "user":
            rec["is_question"] = _is_question(text)
            pending_user_idx = idx
        elif role == "assistant":
            rec["is_answer"] = True
            if pending_user_idx is not None:
                pair_id = f"{conversation_id}_pair_{pair_counter}"
                rec["pair_id"] = pair_id
                # also backfill the user's record after append
                if records and records[-1]["turn_index"] == pending_user_idx:
                    records[-1]["pair_id"] = pair_id
                pair_counter += 1
                pending_user_idx = None
        records.append(rec)
    return records


def build_pairs_jsonl(cleaned_jsonl_path: str, out_jsonl_path: str) -> str:
    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    with open(cleaned_jsonl_path, "r", encoding="utf-8") as f_in, open(out_jsonl_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            cid = str(row.get("conversation_id", ""))
            convo = row.get("conversation", [])
            recs = normalize_conversation(cid, convo)
            for r in recs:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_jsonl_path


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Build per-turn pairs JSONL from cleaned conversations")
    parser.add_argument("cleaned_jsonl", help="Path to cleaned conversations JSONL")
    parser.add_argument("out_jsonl", help="Path to output pairs JSONL")
    args = parser.parse_args()
    print(build_pairs_jsonl(args.cleaned_jsonl, args.out_jsonl)) 