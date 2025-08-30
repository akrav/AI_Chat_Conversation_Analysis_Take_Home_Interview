### Ticket-3006: Synthesize findings and create visualizations (final pass)

**Description**
- Aggregate findings from BERTopic, LLM (v4), rule-based sentiment, and entity clustering to produce final visuals for the insights document.

**Tasks**
- Combine results from unified table `data/04_analysis/unified_table_1000.csv`.
- Generate visuals:
  - Topic frequency (top N)
  - Intent distribution (unique conversations)
  - Sentiment by topic (stacked)
  - Entity category counts and top entity clusters
  - Sentiment overlap (LLM vs VADER/TextBlob)
- Save all images in `reports/images` with clear filenames.

**Recommended Tools**
- Python, `Pandas`, `Matplotlib`, `Seaborn`.

**How to Test**
- Test: Verify images exist under `reports/images`.
- Test: Visual inspection for clarity and accuracy.
- If a test fails: fix and update Troubleshooting. 