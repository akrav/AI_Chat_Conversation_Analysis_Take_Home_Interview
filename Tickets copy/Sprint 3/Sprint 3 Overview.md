### **Sprint 3 Overview: Deeper Structure, Better Topics, Richer Insights**

This sprint focuses on improving data structure, topic quality, and insights. We will split user vs assistant content, link Q→A pairs, upgrade BERTopic to avoid stopword topics, expand visual analytics (including sentiment overlaps and entity category counts), and refine entity clustering.

-----

### **Major Functionality Delivered**
- Per-role preprocessing: explicit `user_text`, `assistant_text`, turn indices, and `pair_id` for Q→A linkage
- BERTopic improvements: stronger vectorizer/stopwords, cleaner topic names
- Advanced visuals: sentiment overlap (LLM vs rule-based), entity category counts, top entity clusters
- Entity normalization + clustering refinements with cluster IDs (and optional labels)
- Unified table v2 including new per-role fields

-----

### **Tickets**
- Ticket-3001: Preprocessing 2.0 — split roles and link Q→A
- Ticket-3002: BERTopic quality improvements
- Ticket-3003: Advanced visualizations (category counts, overlap)
- Ticket-3004: Entity normalization and clustering improvements
- Ticket-3005: Unified table v2 (per-role columns) 