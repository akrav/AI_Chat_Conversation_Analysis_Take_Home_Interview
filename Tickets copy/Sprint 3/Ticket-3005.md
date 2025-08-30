### Ticket-3005: Unified table v2 (per-role columns)

**Description**
- Extend the unified table with per-role fields and pair linkage: `user_text`, `assistant_text`, `pair_id`, and optionally per-role LLM features (intent/sentiment/entities per role) if feasible.

**Tasks**
- Update `src/synthesis.py` to merge with the new pairs JSONL (`data/02_interim/pairs_*.jsonl`).
- Ensure the final CSV continues to include topics, rule-based sentiment, LLM fields, entity clustering, plus per-role text and IDs.

**Recommended Tools**
- `pandas`

**How to Test**
- Build unified table using small synthetic conversations; assert role fields and `pair_id` are present and consistent.

**Reminder to AI Coder**
- Update Sprint-Progress and structure. 