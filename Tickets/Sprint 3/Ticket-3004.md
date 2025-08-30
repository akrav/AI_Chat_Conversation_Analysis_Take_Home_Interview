### Ticket-3004: Entity normalization and clustering refinement

**Description**
- Normalize entities (lowercasing, punctuation/whitespace stripping, simple aliasing), then recluster with embeddings + HDBSCAN. Produce `entity_cluster_id` and a simple `entity_cluster_label` (prototype).

**Tasks**
- Enhance `src/entity_clustering.py`:
  - Add normalization util
  - Recompute embeddings and HDBSCAN labels
  - Assign `entity_cluster_label` by taking the medoid entity text per cluster
- Write results back into unified table.

**Recommended Tools**
- `sentence-transformers`, `hdbscan`, `numpy`, `pandas`

**How to Test**
- Test normalization on a small alias set (e.g., "MSFT"→"microsoft")
- Assert clusters are assigned (≥1 non-noise when duplicates exist)

**Reminder to AI Coder**
- Document aliasing decisions in Troubleshooting.md 