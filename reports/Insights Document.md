### Executive Summary

This analysis transforms 1,000 English, non-toxic WildChat conversations into brand-relevant insights using a hybrid pipeline: BERTopic (topics), rule-based sentiment (VADER/TextBlob), LLM entity/intent extraction, and lightweight entity clustering. Key findings:

- Top themes: assistance/task execution, creative writing, tooling/tech usage, travel/info-seeking. Sentiment is predominantly neutral→positive across top topics; negative clusters are limited and actionable.
- Entities: strong volume around products, organizations, and software; clusters reveal coherent groups (e.g., creative tools, development stacks). Category-level sentiment is favorable overall, with pockets of negativity for specific product/organization pairs.
- Cross-method sentiment alignment: Overlaps show broad agreement (LLM with VADER/TextBlob); divergence often stems from nuanced context that rule-based methods miss.

### Visual Highlights

- Intent distribution: `reports/images/intent_distribution.png`
- Sentiment by topic (stacked): `reports/images/sentiment_by_topic.png`
- Entity category counts: `reports/images/entity_category_counts.png`
- Entity clusters (labeled): `reports/images/entity_clusters_top_labeled.png`
- Entity clusters by category (unique conversations): `reports/images/entity_clusters_by_category.png`
- Sentiment overlap LLM vs VADER/TB: `reports/images/sentiment_overlap_llm_vs_vader.png`, `reports/images/sentiment_overlap_llm_vs_textblob.png`
- Entity sentiment by category (LLM/VADER/TB + combined): files prefixed with `entity_sentiment_by_category*`
- Brand/store/org → product heatmaps (avg sentiment color, counts annotated): `reports/images/brand_product_heatmap.png`, `reports/images/store_product_heatmap.png`, `reports/images/organization_product_heatmap.png`

### Illustrative Snippets (abridged)

- Creative writing assistance (positive): users request long-form scripts; assistant outputs structured drafts with styled scenes.
- How-to/Travel guidance (neutral→positive): Q&A around local transit; assistant provides step-by-step guides.
- Tooling/Stack queries (neutral): explorations of frameworks, libraries, and setup paths.

These examples reflect frequent intents with generally helpful assistant responses, reinforcing a product positioning around assistance quality and breadth.

### Insights and Opportunities

- Product experience
  - Prioritize templates/snippets for the most requested creative and instructional tasks to shorten time-to-value.
  - Monitor negative sentiment pockets in brand→product pairs to identify friction (documentation gaps, onboarding pain).

- Marketing
  - Highlight high-sentiment product categories and clusters where the assistant shines (e.g., structured guidance) in collateral.
  - Use entity clusters to segment outreach (e.g., creative tools vs developer stacks) with tailored narratives.

- Support/Content
  - Publish short, high-utility playbooks addressing common tasks (seen in top topics) and link them in-product.
  - Expand help content for products/orgs with mixed sentiment; use heatmaps to prioritize topics with high counts and lower average sentiment.

### Methodology (brief)

- Data: 1,000 English, non-toxic conversations (WildChat); one conversation per row, many turns.
- Pipeline: cleaning → BERTopic → rule-based sentiment (VADER/TextBlob) → LLM (intent/entities) → entity clustering with labels → visuals.
- Canonical table: `data/04_analysis/unified_table.csv` (entity-level). Pairing was explored but omitted for storage efficiency.

### Caveats

- Sampling and aggregation: entity-level expansion increases row counts; analyses use unique-conversation guards where appropriate.
- Rule-based sentiment is context-light; overlaps with LLM sentiment are reported to show method agreement.

### Next Steps

- Add lightweight per-role analyses using on-demand `unified_pairs.csv` (not persisted) to study Q→A quality without bloating storage.
- Iteratively label clusters and topics for improved interpretability. 