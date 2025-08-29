### Ticket-1006: Evaluate Top2Vec for topic modeling alongside BERTopic

**Description**
- Implement and evaluate a Top2Vec pipeline to generate topics on the cleaned dataset, comparable to the BERTopic pipeline outputs.

**Tasks**
- Install and configure Top2Vec and its dependencies.
- Implement a `top2vec_pipeline.py` to:
  - Load cleaned documents (JSONL) and prepare text like in BERTopic.
  - Fit a Top2Vec model and extract topics and representative documents.
  - Save outputs to `reports/` (topics CSV and doc-topic mapping CSV) with filenames distinct from BERTopic.
- Add a small smoke test (similar scope to BERTopic test) to ensure end-to-end execution on a tiny subset.

**Recommended Tools**
- `top2vec`, `pandas`, existing data loaders in `src/`.

**How to Test**
- Test: Run a small subset (e.g., 20 docs) and confirm outputs are created in a temp `reports/` folder.
- Test: Verify the topics CSV has topic identifiers and words, and that doc-topic CSV aligns in length with documents.
- If a test fails: Analyze and loop to fix; update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md`.

**Reminder to AI Coder**
- After completing this ticket, update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md` and `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md` as needed. 