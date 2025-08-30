### Ticket-2002: Perform fine-grained analysis using LLMs

**Description**
- Implement a script to perform fine-grained analysis on the selected subset of conversations using a Large Language Model (LLM).

**Tasks**
- Implement a Python script to interact with a Large Language Model API (e.g., OpenAI API).
- Develop prompts to categorize, summarize, or extract specific information (e.g., user intent, sentiment) from the conversations in the subset.
- Store the results of the LLM analysis in a structured format (e.g., DataFrame) with extracted features.

**Recommended Tools**
- Python, `openai` library, `requests` library.

**How to Test**
- Test: Run on a small sample to confirm successful API interaction and expected output.
- Test: Check that the output is saved to the correct location and contains the new analytical features.
- If a test fails: Analyze the error and loop to fix. Update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md` if needed. 