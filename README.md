# AI_Chat_Conversation_Analysis_Take_Home_Interview

## Design note: Q/A pairing tradeoff
- The source dataset (WildChat) stores entire conversations per row (a list of turns). A single conversation can contain many user questions and assistant answers.
- We explored producing a per-pair table (one row per user→assistant pair). While useful for per-role analysis, merging that structure with entity-level analytics significantly increased table size (Cartesian-style expansion at the conversation level), e.g., >1 GB for a 1K-conversation slice.
- To keep the repo lightweight and within storage constraints for this take‑home, we avoid shipping the paired table by default and run all visuals/analysis from the entity-level unified table (`data/04_analysis/unified_table.csv`).
- If needed, a per-pair table can still be generated locally using the CLI (`python -m src.synthesis pairs <pairs_jsonl> --out data/04_analysis/unified_pairs.csv`) without committing the large file.