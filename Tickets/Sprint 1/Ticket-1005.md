### Ticket-1005: Execute BERTopic analysis on the preprocessed dataset

**Description**
- Execute BERTopic analysis on the preprocessed dataset to identify core themes and topics.

**Tasks**
- Use the preprocessed data from `data/02_interim/`.
- Implement the BERTopic analysis pipeline, including using `huggingface/transformers` to create contextual embeddings.
- Extract the top topics and representative conversations.
- Save the results to the `reports/` folder.

**Recommended Tools**
- `BERTopic`, `huggingface/transformers`.

**How to Test**
- Test: Manually inspect the output to ensure topics are generated and appear coherent.
- Test: Verify that the output is saved to the correct location (`reports/`).
- If a test fails: Analyze the error and loop to fix. This might require updating the `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md` file.

**Reminder to AI Coder**
- After completing this ticket, please update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md`. 