### Cursor Prompt: Execute Ticket TICKET-3001

Please execute this ticket:

- TICKET_ID: 3001
- TICKET_NAME: Ticket-3001.md
- TICKET_FILE: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Tickets/Sprint 3/Ticket-3001.md

Permanent references (always follow):
- Architecture: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Architecture Document.md
- Implementation Plan: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Implementation Plan Document.md
- Product Requirements: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Product Requirements Document.md
- Project Structure: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md

Objective:
- Implement role-splitting and Q→A pair linkage; write pairs JSONL under data/02_interim/; add tests.

How to Test:
- Run: `python -m pytest -q`
- Verify: pairs JSONL exists and contains expected schema.

Notes:
- Do not commit secrets. Update .env.example if new vars are needed. 