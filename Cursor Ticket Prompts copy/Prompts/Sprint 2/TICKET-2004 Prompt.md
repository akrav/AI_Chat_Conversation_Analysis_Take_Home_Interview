### Cursor Prompt: Execute Ticket TICKET-2004

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 2004
- TICKET_NAME: Ticket-2004.md
- TICKET_FILE: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Tickets/Sprint 2/Ticket-2004.md

Permanent references (always follow):
- Architecture: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Architecture Document.md
- Implementation Plan: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Implementation Plan Document.md
- Product Requirements: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Documentation/Product Requirements Document.md
- Project Structure: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/AI_Chat_Conversation_Analysis_Take_Home_Interview/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env (never commit secrets). If new keys are needed, add placeholders to .env.example and document in README.

Objective:
- Implement TICKET-2004 fully: synthesize BERTopic + LLM outputs and generate required visualizations saved to `reports/images`.

Constraints and style:
- Follow Engineering Best Practices.
- Keep changes scoped. No secrets in code or logs.

Required steps:
1) Read the ticket file.
2) Merge analysis outputs into a unified structure.
3) Create time series of topic frequency, intents distribution bar chart, and sentiment-topic visuals.
4) Save all images to `reports/images` with clear naming.
5) Documentation updates (Sprint-Progress, Troubleshooting, structure).
6) Testing:
   - Add tests that check image files exist and basic data assumptions. Run tests; fix failures.

Output:
- Summary of visualizations created, file paths, and test commands.

Success criteria:
- All required visualizations created and saved.
- Tests added and passing.
- Docs updated (Sprint-Progress, Troubleshooting, structure). 