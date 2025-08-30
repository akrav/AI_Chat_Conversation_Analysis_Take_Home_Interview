### Ticket-2003: Synthesize findings and create visualizations

**Description**
- Aggregate findings from both the BERTopic and LLM analysis and create visualizations for the final insights document.

**Tasks**
- Combine results from `data/04_analysis/bertopic_results.csv` and `data/04_analysis/llm_results.csv` into a unified data structure.
- Generate a visualization showing the frequency of topics over time to identify trends.
- Create a bar chart to show the distribution of LLM-classified user intents.
- Generate visualizations showing the relationship between LLM-categorized sentiment and specific topics.
- Save all generated visualizations in the `reports/images` folder.

**Recommended Tools**
- Python, `Pandas`, `Matplotlib`, `Seaborn`.

**How to Test**
- Test: Verify that visualizations are generated and saved to `reports/images` with appropriate filenames.
- Test: Visually inspect charts to ensure clarity and accurate representation.
- If a test fails: Analyze the error and loop to fix. Update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md` if needed. 