### Ticket-1004: Create and apply the data cleaning function

**Description**
- Create and apply a reusable data cleaning function for conversation text.

**Tasks**
- Develop a reusable Python function in the `src/` directory to clean conversation text. The function should handle tasks like removing special characters and normalizing text.
- Apply the newly created cleaning function to the dataset.
- Save the fully preprocessed dataset to `data/02_interim/`.

**Recommended Tools**
- Python, `Pandas`, `re` (regular expressions).

**How to Test**
- Test: Create a unit test in the `tests/` directory for the text cleaning function to ensure it correctly removes special characters and normalizes a sample input string.
- Test: Write a simple assertion to check that the output file is created in the correct location (`data/02_interim/`).
- Test: Manually inspect a few rows of the saved data to ensure the text cleaning was applied correctly.
- If a test fails: Analyze the error and loop to fix. This might require updating the `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md` file.

**Reminder to AI Coder**
- After completing this ticket, please update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md`. 