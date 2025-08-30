### Ticket-3001: Preprocessing 2.0 — split roles and link Q→A

**Description**
- Parse each conversation into explicit turns with `turn_index`, split into `user_text` and `assistant_text`, and create `pair_id` that links a user turn to its subsequent assistant reply (if present). Persist a normalized JSONL for downstream analysis and update the pipeline to optionally use the new format.

**Tasks**
- Add `src/preprocess_pairs.py` with functions:
  - `normalize_conversation(conversation: list[dict]) -> list[dict]` producing records with fields: `conversation_id`, `turn_index`, `role`, `text`, `pair_id` (same for the user turn and its answer), `is_question` (heuristic), `is_answer`.
  - `build_pairs_jsonl(cleaned_jsonl_path, out_jsonl_path)` reading cleaned data and writing `data/02_interim/pairs_1000.jsonl` (1 record per turn).
- Update tests to cover role-splitting and pair linkage for a small synthetic example.

**Recommended Tools**
- `pandas`, `re`

**How to Test**
- Unit test: Given a 2-turn conversation (user→assistant), assert both turns exist with matching `pair_id` and correct roles.
- Integration: Run builder against a tiny JSONL and validate output schema and counts.
- If a test fails: fix and document in `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md`.

**Reminder to AI Coder**
- Update Sprint-Progress and structure docs after completion. 