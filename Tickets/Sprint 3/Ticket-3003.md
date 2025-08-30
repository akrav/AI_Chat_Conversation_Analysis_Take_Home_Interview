### Ticket-3003: Advanced visualizations (categories & overlap)

**Description**
- Create richer visuals: entity category counts, sentiment overlap matrices (LLM vs VADER/TextBlob), entity sentiment by category, top entity clusters. Save to `reports/images`.

**Tasks**
- Extend `src/visualizations.py` with:
  - `plot_entity_category_counts`
  - `plot_sentiment_overlap_matrix` (LLM vs VADER/TextBlob)
  - `plot_entity_sentiment_by_category`
  - Ensure existing plots remain
- Add a CLI in the module to generate all images for a given CSV.

**Recommended Tools**
- `pandas`, `matplotlib`, `seaborn`

**How to Test**
- Tests check that PNGs are created and non-empty.
- Validate basic shape of overlap matrix and non-negativity.

**Reminder to AI Coder**
- Update Sprint-Progress and Troubleshooting with any plotting caveats. 