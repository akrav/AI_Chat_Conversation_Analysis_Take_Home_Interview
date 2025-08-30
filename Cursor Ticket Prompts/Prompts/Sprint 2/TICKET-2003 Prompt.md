### Cursor Prompt: Execute Ticket TICKET-2003

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 2003
- TICKET_NAME: Ticket-2003.md
- TICKET_FILE: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Tickets/Sprint 2/Ticket-2003.md

Permanent references (always follow):
- Architecture: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Architecture Document.md
- Implementation Plan: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Implementation Plan Document.md
- Product Requirements: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Product Requirements Document.md
- Project Structure: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env (never commit secrets). For new keys, add placeholders to env.example and document in README.

Objective:
- Implement rule-based sentiment analysis using NLTK VADER and TextBlob, and write a CSV to `data/04_analysis/sentiment_rule_based.csv`.

Constraints and style:
- Follow Engineering Best Practices.
- Keep changes scoped. No secrets in code or logs.

Required steps:
1) Read the ticket file.
2) Add `src/sentiment_rule_based.py` with VADER and TextBlob functions and a batch processor for the subset JSONL.
3) Write outputs to `data/04_analysis/sentiment_rule_based.csv` with required columns.
4) Add tests: known strings and tiny JSONL batch.
5) Documentation updates (Sprint-Progress, Troubleshooting if needed).
6) Testing:
   - Run tests; fix failures until green.

Output:
- Concise summary of outputs created and how to run the sentiment batch.

Success criteria:
- CSV saved with expected columns; tests added and passing; docs updated. 