### Ticket-1003: Convert to DataFrame and perform initial EDA

**Description**
- Convert the loaded `huggingface/datasets` object into a Pandas DataFrame and perform initial EDA.

**Tasks**
- Convert the loaded `huggingface/datasets` object into a Pandas DataFrame.
- Perform Exploratory Data Analysis (EDA) to check for missing values, data types, and basic statistics.
- Document the findings of the EDA in a Jupyter Notebook, which will serve as the `Data Exploration File`.

**Recommended Tools**
- `Pandas`, `Matplotlib` or `Seaborn` (for basic visualizations).

**How to Test**
- Test: Assert that the output is an instance of a Pandas DataFrame.
- Test: Manually inspect the notebook to ensure the checks for missing values and data types are present and the findings are clearly documented.
- If a test fails: Analyze the error and loop to fix. This might require updating the `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md` file.

**Reminder to AI Coder**
- After completing this ticket, please update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md`. 