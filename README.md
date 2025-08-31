# AI Conversation Analytics — Design & Runbook

This repository implements an end‑to‑end analytics pipeline on AI chat conversations (WildChat). It transforms raw conversations into topic insights, entity‑level sentiment, and a rich set of visualizations suitable for strategy and product analysis.

## What this system does
- Loads a subset of the WildChat dataset
- Cleans and normalizes text
- Runs BERTopic for topic discovery
- Computes rule‑based sentiment (VADER + TextBlob)
- Uses an LLM to extract: intent, overall sentiment, and a nested list of entities with per‑entity sentiment and categories
- Flattens the LLM results (one row per entity), merges everything into a unified table
- Clusters entities (to normalize spelling/variants and surface competitor clusters)
- Produces visualizations of intents, topics, entity categories, multi‑sentiments, and brand/store/org vs product heatmaps

Outputs are written to:
- `data/01_raw/`, `data/02_interim/`, `data/03_processed/`, `data/04_analysis/`
- `reports/` (including `reports/images/`)

---

## Design decisions (why nested + flat, multiple sentiments, clustering)

- Nested vs Flat LLM results
  - The LLM naturally emits a nested JSON object per conversation: `{ intent, sentiment, entities: [ {text, category, sentiment} ] }`.
  - For analytics we need row‑wise tabular data. We therefore flatten to one row per entity with conversation‑level fields repeated.
  - We retain the nested file for audits/debugging; the pipeline consumes the flat file.

- Entity sentiment and category (why per‑entity sentiment matters)
  - A conversation’s overall sentiment can diverge from sentiment toward a particular product/store/brand/organization.
  - Entity categories (brand, store, product, organization, possible_*) let us filter down to the most actionable slices.
  - Per‑entity sentiment enables: brand→product heatmaps, store→product heatmaps, and per‑category multi‑sentiment bars.

- Multiple sentiments (LLM + VADER + TextBlob)
  - Redundant measurements increase confidence. Rule‑based methods are fast, consistent baselines; LLM sentiment can capture nuance.
  - Visuals include overlaps and side‑by‑side comparisons.

- Entity clustering
  - We normalize entity text and cluster to group spelling variations and semantically similar mentions. This reveals families of products and competitors.
  - If HDBSCAN is unavailable or degenerate, we fall back to KMeans; labels summarize top normalized names per cluster.

- Topics (BERTopic) and stopwords
  - Stopwords can become trivial topic artifacts. We harden vectorization/stopword handling so topics are meaningful.
  - Topic −1 (“Other”) is used for low‑content docs; we still keep them in the table.

- Metrics
  - “Mentions” = row counts (entity‑level rows in unified table)
  - “Unique entities” = distinct normalized `entity_norm`
  - “Conversations” = count of unique `conversation_id`

---

## Repository structure

```
project_root/
  data/
    01_raw/        # raw pulls/saves from the source
    02_interim/    # cleaned/normalized JSONL
    03_processed/  # model outputs (LLM nested+flat, CSV)
    04_analysis/   # unified table (CSV/XLSX), rule-based CSV
  reports/
    images/        # generated PNG figures
    Insights Document.md
  src/
    hf_loader.py               # load + filter WildChat
    preprocess.py              # text cleaning to interim JSONL
    bertopic_pipeline.py       # BERTopic, topics/doc_topics CSVs
    sentiment_rule_based.py    # VADER + TextBlob CSV
    llm_analysis.py            # LLM extraction (nested + flat outputs)
    synthesis.py               # build unified_table from flat LLM
    entity_clustering.py       # normalize + cluster entities
    visualizations.py          # generate figures
    run_pipeline.py            # end-to-end orchestrator (CLI)
```

Why the folders?
- `01_raw` preserves the original pull (reproducibility)
- `02_interim` separates cleaning from modeling
- `03_processed` stores model artifacts (LLM/others) that can be re‑used without recomputation
- `04_analysis` is the canonical analytics table used for visuals

---

## How to run (end‑to‑end)

Prereqs:
- Python 3.13 venv: `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Set a valid `OPENAI_API_KEY` in your shell (recommended) or a simple `.env` (see Troubleshooting for `.env` rules)

Quick start — full pipeline on 1000 rows (with LLM):
```bash
source .venv/bin/activate && OPENAI_API_KEY="$OPENAI_API_KEY" \
python -m src.run_pipeline --limit 1000 --llm_model gpt-4.1-nano --max_llm_rows all --concurrency 1
```
- Adjust `--concurrency` to your rate limits; if you see 429s, lower to 1.

Small run (smoke test, faster):
```bash
source .venv/bin/activate && OPENAI_API_KEY="$OPENAI_API_KEY" \
python -m src.run_pipeline --limit 5 --llm_model gpt-4.1-nano --max_llm_rows 5 --concurrency 1
```

What this does:
1) Load + filter → `data/01_raw/wildchat_english_notoxic_XXXX.jsonl`
2) Preprocess → `data/02_interim/..._cleaned.jsonl`
3) BERTopic → `reports/bertopic_topics.csv` + `reports/bertopic_doc_topics.csv`
4) Rule‑based sentiment → `data/04_analysis/sentiment_rule_based_1000.csv`
5) LLM analysis → nested JSONL and flat JSONL/CSV in `data/03_processed/`
6) Unified table → `data/04_analysis/unified_table.csv` and `.xlsx`
7) Entity clustering → cluster columns added to unified table
8) Visuals → PNGs in `reports/images/`

---

## How to run in steps (fine‑grained control)

1) Load + preprocess only
```bash
python -m src.hf_loader  # see functions inside if you need custom pulls
python -m src.preprocess
```

2) BERTopic only
```bash
python -m src.bertopic_pipeline
```

3) Rule‑based sentiment only
```bash
python - <<'PY'
from src.sentiment_rule_based import batch_process_subset
batch_process_subset('data/02_interim/wildchat_english_notoxic_1000_cleaned.jsonl',
                     'data/04_analysis/sentiment_rule_based_1000.csv')
PY
```

4) LLM only (produces nested + flat)
```bash
OPENAI_API_KEY="$OPENAI_API_KEY" \
python -m src.llm_analysis \
  data/02_interim/wildchat_english_notoxic_1000_cleaned.jsonl \
  data/03_processed/llm_analysis_results.jsonl \
  --model gpt-4.1-nano --max_rows all --concurrency 1
```
- Outputs:
  - Flat primary: `data/03_processed/llm_analysis_results.jsonl` and `.csv`
  - Nested debug: `data/03_processed/llm_analysis_results_nested.jsonl`

5) Build unified table + visuals using existing LLM results
```bash
python -m src.run_pipeline --limit 1000 --llm_model gpt-4.1-nano --max_llm_rows all --concurrency 1 --skip_llm
```

---

## Script internals (high level)

- `src/hf_loader.py`: filters by `language` and non‑toxic flag; writes JSONL to `01_raw/`.
- `src/preprocess.py`: standardizes and cleans content (URL removal, whitespace collapse, punctuation filtering) → `02_interim/`.
- `src/bertopic_pipeline.py`: fits BERTopic on sufficient‑context documents, writes topic name/id tables. Robust vectorizer fallback prevents “no terms remain”. Ensures topic −1 rows exist.
- `src/sentiment_rule_based.py`: VADER and TextBlob per‑conversation; writes CSV to `04_analysis/`.
- `src/llm_analysis.py`:
  - Builds a strict JSON prompt; requests `response_format={"type":"json_object"}` when supported.
  - Async with bounded concurrency, ETA logging, retries.
  - Writes both nested JSONL and FLAT JSONL/CSV (one row per entity). The flat file is the canonical input to synthesis.
- `src/synthesis.py`: merges cleaned + topics + rule‑based + flat LLM file → `unified_table.(csv|xlsx)`.
- `src/entity_clustering.py`: normalizes entity text, embeds and clusters; adds `entity_cluster_id` and `entity_cluster_label`.
- `src/visualizations.py`: generates all figures; bar charts sorted left→right descending, sensible fallbacks when data is missing.
- `src/run_pipeline.py`: the orchestrator with CLI flags; logs step durations and validates LLM row counts.

---

## Parameters and recommended values

- End‑to‑end (`src/run_pipeline.py`)
  - `--limit` (int): number of conversations to pull and process (e.g., 1000)
  - `--language` (str): language filter, default English
  - `--llm_model` (str): e.g., `gpt-4.1-nano` (fast) or `gpt-4o-mini`
  - `--max_llm_rows` (int|all): cap for LLM calls; `all` = no cap
  - `--concurrency` (int): parallel LLM calls; start with 3; lower if you hit 429s
  - `--skip_llm` (flag): reuse existing LLM outputs

- LLM‑only (`src.llm_analysis`)
  - `subset`, `output` (paths)
  - `--model`, `--max_rows`, `--concurrency`, `--log_every`

---

## Visualizations and what they tell us

- Intent distribution (`intent_distribution.png`)
  - Overview, what categories are talked about and how many Conversation Chains are there about that category.

- Entity category counts (`entity_category_counts.png`)
  - Which entity categories had the highest number of unique entities. Helps scope brand/product analytics. (Example Category: Software, Example entities: Nike, nike, and adidas would be 2 unique entities)

- Entity clusters (labeled) (`entity_clusters_top_labeled.png`)
  - Top clusters by number of unique entities; labels show top normalized names per cluster. Identifies major themes/competitors. (Cluster by entity, similarity, and how many unique entities are in those clusters.)


- Entity sentiment by category — LLM (`entity_sentiment_by_category_llm.png`)
  - Frequency chart. For each category, bars show the number of entity mentions in Negative/Neutral/Positive per LLM sentiment. Mixed sentiment is not averaged; an entity with both positive and negative mentions contributes to both buckets according to its mentions.

- Entity sentiment by category — VADER (`entity_sentiment_by_category_vader.png`)
  - Frequency chart. Mentions per category split into Negative/Neutral/Positive using VADER labels.

- Entity sentiment by category — TextBlob (`entity_sentiment_by_category_textblob.png`)
  - Frequency chart. Mentions per category split into Negative/Neutral/Positive using TextBlob labels.

- Entity sentiment by category — Combined (`entity_sentiment_by_category_combined.png`)
  - Side‑by‑side frequency panels (LLM/VADER/TextBlob) showing mentions per category by sentiment. No averaging.

Average sentiment by category — LLM/VADER/TextBlob (`entity_avg_sentiment_by_category_[method].png`)
  - Average sentiment scored as −1 (negative), 0 (neutral), +1 (positive), averaged per entity, then averaged across unique entities in each category (entity‑weighted). Shows a single bar per category per method.

Average sentiment by category — Combined (`entity_avg_sentiment_by_category_combined.png`)
  - Three stacked panels (LLM/VADER/TextBlob) showing entity‑weighted average sentiment by category for quick comparison.

- Entities — Brand multi‑sentiment (`entities_brand_multi.png`)
  - Multi‑panel frequency chart for the Brand category. Each entity’s stacked bar shows how many of its mentions are Negative/Neutral/Positive for LLM, VADER, and TextBlob.

- Entities — Organization multi‑sentiment (`entities_organization_multi.png`)
  - Multi‑panel frequency chart for the Organization category; stacked bars count mentions per sentiment.

- Entities — Possible brand/store/product multi‑sentiment (`entities_possible_brand_multi.png`, `entities_possible_store_multi.png`, `entities_possible_product_multi.png`)
  - Tracks ambiguous mentions likely to be brands/stores/products; useful for discovery.
  - Multi‑panel frequency charts; stacked bars count mentions per sentiment.

- Store→Product mentions heatmap (`store_product_mentions_heatmap.png`)
  - Rows: top stores; columns: top products; numbers: mention counts; color: average product sentiment (−1..1). Mentions-only version retained.

- Brand→Product mentions heatmap (`brand_product_mentions_heatmap.png`)
  - Mentions-only version retained.

- Organization→Product mentions heatmap (`organization_product_mentions_heatmap.png`)
  - Mentions-only version retained.

- Sentiment overlap — LLM vs VADER (`sentiment_overlap_llm_vs_vader.png`)
  - Agreement/mismatch matrix across overall sentiment labels.

- Sentiment overlap — LLM vs TextBlob (`sentiment_overlap_llm_vs_textblob.png`)
  - Agreement/mismatch matrix with TextBlob.

Notes:
- `sentiment_by_topic.png` has been deprecated and is no longer generated.
- “Mentions” are pure row counts from the unified table; mixed sentiment is not averaged. An entity may appear in multiple sentiment buckets proportional to its mention counts.
- “Average sentiment by category” uses entity‑weighted averages: map sentiment to −1/0/+1, average per entity (to avoid over‑counting repeated mentions of the same entity), then average across unique entities in the category.

---

## Visuals overview (what each chart shows)

- Intent distribution: `reports/images/intent_distribution.png`
  - Top conversation intents by unique conversations.

- Conversation sentiment by intent: `reports/images/entity_sentiment_by_category.png`
  - Stacked Negative/Neutral/Positive counts of unique conversations per intent.

- Entity category counts: `reports/images/entity_category_counts.png`
  - Unique entities per category (e.g., brand, organization, product).

- Entities — multi-sentiment (mentions-only):
  - Brand: `reports/images/entities_brand_multi.png`
  - Organization: `reports/images/entities_organization_multi.png`
  - Possible brand/store/product: 
    - `reports/images/entities_possible_brand_multi.png`
    - `reports/images/entities_possible_store_multi.png`
    - `reports/images/entities_possible_product_multi.png`
  - Each stacked bar shows mention counts in Negative/Neutral/Positive for LLM/VADER/TextBlob.

- Mentions heatmaps (with average sentiment color):
  - Store→Product: `reports/images/store_product_mentions_heatmap.png`
  - Brand→Product: `reports/images/brand_product_mentions_heatmap.png`
  - Organization→Product: `reports/images/organization_product_mentions_heatmap.png`
  - Numbers are mention counts; color encodes mean product sentiment (−1..1).

- Average sentiment by category (entity-weighted):
  - LLM: `reports/images/entity_avg_sentiment_by_category_llm.png`
  - VADER: `reports/images/entity_avg_sentiment_by_category_vader.png`
  - TextBlob: `reports/images/entity_avg_sentiment_by_category_textblob.png`
  - Combined panels: `reports/images/entity_avg_sentiment_by_category_combined.png`

Notes:
- Mentions are pure frequency; mixed sentiment is not averaged.
- Average sentiment uses entity-weighting: per-mention −1/0/+1 → mean per entity → mean across entities in the category.

## Insights

See the full analysis and actionable recommendations in the Insights Document: `Insights Document.md`.
