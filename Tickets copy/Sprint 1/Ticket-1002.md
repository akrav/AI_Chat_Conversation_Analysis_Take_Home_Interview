### Ticket-1002: Programmatically load the `allenai/WildChat` dataset

**Description**
- Programmatically load the `allenai/WildChat` dataset.

**Tasks**
- Write a Python script or notebook to programmatically load the `allenai/WildChat` dataset using the Hugging Face `datasets` library.
- Save the raw dataset to `data/01_raw/` for future reference.

**Recommended Tools**
- Python, `huggingface/datasets`.

**How to Test**
- Test: Write a simple assertion to check if the loaded dataset object is not empty.
- Test: Verify that the file is saved to the correct directory (`data/01_raw/`).
- If a test fails: Analyze the error and loop to fix. This might require updating the `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md` file with the encountered issue and solution.

**Reminder to AI Coder**
- After completing this ticket, please update `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md` and `/Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md` if necessary. 