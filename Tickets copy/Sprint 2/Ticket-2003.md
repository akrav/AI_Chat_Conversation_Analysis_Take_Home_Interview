### Ticket-2003: Sentiment analysis with NLTK VADER and TextBlob

**Description**
- Implement rule-based sentiment analysis using NLTK VADER and TextBlob on the selected subset. Persist polarity and subjectivity (TextBlob) and VADER scores per conversation (or per first user turn) for comparison with LLM sentiment.

**Tasks**
- Install dependencies: `nltk`, `textblob`.
- In `src/sentiment_rule_based.py`:
  - Provide functions to compute VADER scores and TextBlob sentiment for a given text.
  - Batch process `data/03_processed/subset_llm_english_notoxic.jsonl` and write `data/04_analysis/sentiment_rule_based.csv` with columns: `conversation_id`, `vader_pos`, `vader_neg`, `vader_neu`, `vader_compound`, `tb_polarity`, `tb_subjectivity`.
- Optionally expose a CLI entry for ad-hoc runs.

**Recommended Tools**
- `nltk` (VADER lexicon), `textblob`, `pandas`.

**How to Test**
- Test: Unit test that verifies deterministic scores for known strings (e.g., "great" positive, "terrible" negative) using both VADER and TextBlob.
- Test: Run on a tiny synthetic JSONL with 2-3 examples and verify CSV rows and columns exist.
- If a test fails: Analyze and loop to fix; update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md`.

**Reminder to AI Coder**
- After completing this ticket, update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md`. 