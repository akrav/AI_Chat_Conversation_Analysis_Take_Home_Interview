# AI_Chat_Conversation_Analysis_Take_Home_Interview

## Note on Q→A Pairing

Separating each conversation into explicit question→answer pairs was explored and prototyped (see `src/preprocess_pairs.py`). However, joining per-pair records with entity-level rows causes large row-multiplication and ballooned storage (e.g., a v2 table reached ~1.4 GB for a 1k sample). To keep the repository lightweight and focused for this take-home, we avoid persisting a unified per-pair table. All analyses and visualizations operate on the entity-level unified table (`data/04_analysis/unified_table.csv`).