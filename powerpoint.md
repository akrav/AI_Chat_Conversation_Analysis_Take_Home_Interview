# Title: AI Conversation Analytics – Technical Deep Dive

## Slide 0: Table of Contents (Order of Operations)
- Load & filter → Preprocess → Topics (BERTopic)
- Rule-based sentiment (VADER/TextBlob)
- LLM extraction (intent, entities, per-entity sentiment)
- Synthesis/merge to unified table
- (Optional) Entity clustering
- Visualizations and insights

## Slide 1: Problem & Objectives
- Analyze real-world AI conversations to extract actionable insights for brands
- Provide reproducible visuals and queries
- Build a robust, performant pipeline with modular steps

## Slide 2: Data Sources
- WildChat subset (English, non-toxic)
- Outputs: nested LLM JSONL, flat entity-level CSV/JSONL
- Analysis table: `data/04_analysis/unified_table.csv`

## Slide 3: High-Level Pipeline
- Load & filter → Preprocess → Topics (BERTopic)
- Rule-based sentiment (VADER/TextBlob)
- LLM extraction (intent, entities, per-entity sentiment)
- Synthesis/merge to unified table
- (Optional) Entity clustering
- Visualizations and insights

## Slide 4: Order of Operations (Why this order)
- Preprocess before topics to improve topic quality
- Topics before synthesis for context fields
- Rule-based sentiment before LLM synthesis for complete sentiment features
- LLM extraction before visualizations to enable per-entity analytics
- Synthesis validates and aligns all fields for charts

## Slide 5: Tools & Libraries
- Python 3.13
- pandas, matplotlib, seaborn (optional), numpy
- VADER, TextBlob for rule-based sentiment
- BERTopic (UMAP/HDBSCAN/KMeans fallback via scikit-learn)
- sentence-transformers (optional) for embeddings
- OpenAI LLM for extraction (intent/entities/sentiment)

## Slide 6: Load & Filter (English, non-toxic, sample size)
- Rationale
  - Focus on English, non-toxic content (for US brands/products) to reduce noise
  - Limit to N=1000 on local machine to balance coverage and speed
- Code
  - File: `src/hf_loader.py` (language/toxic filter and limit)
  - Lines: 27–44
  - Snippet:
    ```python
    ds = load_dataset("allenai/WildChat", split=split, token=token)
    df = pd.DataFrame(ds)
    mask = (df["language"] == language) & (df["toxic"] == False)
    filtered = df.loc[mask]
    if limit is not None:
        filtered = filtered.head(limit)
    ```

## Slide 7: Preprocess (Cleaning & Normalization)
- What we do
  - Remove URLs, collapse whitespace, keep common punctuation, lowercase
  - Normalize each conversation turn consistently
- Why
  - Improves vectorization and topic stability; reduces junk tokens
- Code
  - File: `src/text_cleaning.py` (clean_text, clean_conversation_list)
  - Lines: 12–20, 23–29
  - Snippet:
    ```python
    s = _URL_RE.sub(" ", s)
    s = _KEEP_CHARS_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    s = s.lower()
    ```
  - Pipeline call: `src/preprocess.py` 10–29

## Slide 8: Topics (BERTopic) – Stopwords & Outputs
- What we do
  - Remove stopwords so topics capture meaningful content words rather than function words (the, and, of)
  - Build dense representations with UMAP (reduces dimensionality while preserving structure), then cluster with HDBSCAN (density-based)
  - Use a robust vectorizer fallback (min_df/ngram adjustments) to avoid failures on sparse corpora
  - Keep Topic -1 (Other) to preserve low-content/uncategorizable docs instead of dropping them
- Why we reduce dimensionality then cluster
  - Text turns into very high‑dimensional vectors (one dimension per token); distances become noisy and slow
  - UMAP compresses these vectors into a small number of dimensions that still preserve neighborhood structure
  - HDBSCAN then clusters the lower‑dimensional points; this is faster and more accurate than clustering raw sparse vectors
- Vectorizer fallback (what it is and why it helps)
  - The vectorizer picks which words/ngrams become features. With tiny/sparse corpora, strict settings can yield “no features.”
  - We start stricter (min_df=5, allow bigrams) to reduce noise; if it fails, we relax (min_df=1, unigrams) so a vocabulary definitely exists.
  - This guarantees BERTopic can fit even when data is small or uneven.
- Code (links/paths)
  - Stopwords/content sufficiency: `src/bertopic_pipeline.py` 18–21, 32–38
  - UMAP/HDBSCAN: `src/bertopic_pipeline.py` 85–103
  - Vectorizer fallback: `src/bertopic_pipeline.py` 106–117, 125–131
  - Topic -1 & artifacts: `src/bertopic_pipeline.py` 162–168, 190–221

## Slide 9: Rule-based Sentiment (VADER/TextBlob)
- Why VADER & TextBlob
  - Surveyed common rule-based approaches; these are fast, stable, widely used
  - Dual-method redundancy provides additional confidence via agreement
- What we compute
  - VADER: pos/neg/neu/compound; TextBlob: polarity/subjectivity
- Code
  - File: `src/sentiment_rule_based.py` 24–31, 33–37, 39–45, 53–83
  - Snippet:
    ```python
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    tb = TextBlob(text)
    polarity, subjectivity = tb.sentiment
    ```

## Slide 10: LLM Extraction (Intent, Entities, Per-Entity Sentiment)
- Goals
  - Overall conversation intent & sentiment
  - Nested entities with category + per-entity sentiment
  - Note: entity sentiment/category may diverge from overall chat sentiment/intent (e.g., negative chat but positive entity)
- Flattening detail
  - Multiple entities per conversation → duplicate conversation_id across rows so each entity gets one row
- Concurrency for speed
  - Semaphore-bounded async calls; retry/backoff; progress logging
  - In plain English: a semaphore is a simple counter that limits how many requests run in parallel at the same time. This prevents overwhelming the API and reduces throttling.
- Code (links/paths)
  - Prompt & schema: `src/llm_analysis.py` 66–81
  - Async + retry/backoff: `src/llm_analysis.py` 158–196
  - Concurrency gate + progress logs: `src/llm_analysis.py` 198–233
  - Flatten to flat JSONL/CSV: `src/llm_analysis.py` 267–306
- Snippet (concurrency & retry):
```python
# src/llm_analysis.py 198–233, 158–196
sem = asyncio.Semaphore(concurrency)
async def bound_analyze(r):
    async with sem:
        return await _analyze_single_async(client, r, model)
# retry/backoff
attempt = 0
while True:
    try:
        resp = await client.chat.completions.create(...)
        return enriched
    except Exception:
        attempt += 1
        if attempt > retries:
            return fallback_enriched
        await asyncio.sleep(min(2 ** attempt + random.random(), 10))
```

## Slide 11: Synthesis/Merge to Unified Table
- What we do
  - Load cleaned conversation text and derive full_text
    - Cleaned conversation: each message’s text normalized (lowercased, links removed, punctuation filtered, whitespace collapsed)
    - Derived full_text: all normalized messages stitched together per conversation to create a single analysis string
  - Merge doc_topics with topic names
    - There is a topics table mapping topic_id → topic_name
    - We merge doc_topics (document→topic_id assignments) with topics to attach the readable topic_name to each conversation
  - Merge rule-based sentiment by conversation_id
  - Merge LLM flat outputs by conversation_id (duplicates rows for per-entity analytics)
  - Optionally merge pairs (user/assistant text pairs) if available
  - Write unified CSV and also XLSX for easy viewing
- Code (links/paths)
  - Load cleaned + full_text: `src/synthesis.py` 10–22
  - Topics merge (doc_topics + topics names): `src/synthesis.py` 79–84
  - Rule-based & LLM merges: `src/synthesis.py` 96–104
  - Column selection & outputs: `src/synthesis.py` 105–132
- Snippet (merge logic):
```python
# src/synthesis.py 94–104, 105–118
base = cleaned_df.merge(doc_topics[["conversation_id","topic_id","topic_name"]],
                        on="conversation_id", how="left")
if not rule_df.empty:
    base = base.merge(rule_df, on="conversation_id", how="left")
if not llm_df.empty:
    base = base.merge(llm_df, on="conversation_id", how="left")
cols = [c for c in desired_cols if c in base.columns]
out_df = base[cols].copy()
```

## Slide 12: Entity Clustering
- What we do
  - Normalize entities → embed (MiniLM) → HDBSCAN (fallback KMeans → buckets); relabel and human-readable labels
- Why MiniLM embeddings
  - MiniLM gives compact, semantic vectors quickly; ideal for short entity strings where speed matters
- Why HDBSCAN first
  - Density-based clustering auto-discovers cluster count, handles variable densities, and labels noise points (no forced assignment)
- Why fallback KMeans (then buckets)
  - If HDBSCAN is unavailable or produces one large cluster, KMeans provides a stable alternative with a heuristic k; buckets ensure the pipeline never fails
- Code & snippets
  - Normalization: `src/entity_clustering.py` 10–17
    ```python
    s = text.lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = " ".join(s.split())
    ```
  - Embedding: `src/entity_clustering.py` 20–29
    ```python
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    ```
  - TF‑IDF fallback embedding: `src/entity_clustering.py` 30–34
    ```python
    vec = TfidfVectorizer(min_df=1, max_features=4096)
    X = vec.fit_transform(texts)
    return normalize(X).toarray().astype(np.float32)
    ```
  - Clustering (HDBSCAN → KMeans → buckets): `src/entity_clustering.py` 37–63
    ```python
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embeddings).astype(int)
    # Fallback KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings).astype(int)
    # Fallback buckets
    labels = np.arange(n, dtype=int) // bucket
    ```
  - Relabel + human-readable labels: `src/entity_clustering.py` 90–109
    ```python
    unique_labels = sorted(non_noise.unique().tolist())
    mapping = {old: new for new, old in enumerate(unique_labels)}
    label_map[int(cid)] = ", ".join(top_entities[:3])
    ```

## Slide 13: Visualizations and Insights
- See README “Visualizations and what they tell us” and “Visuals overview” for per-graph descriptions
- Mentions-only distributions; entity-weighted averages; co-mention heatmaps with average sentiment color; conversation sentiment by intent

## Slide 14: Key Design Choices (Why)
- Mentions distributions vs entity-weighted averages (breadth vs macro orientation)
- Conversation dedup for intent charts (unique conversations)
- Robust fallbacks: vectorizer, clustering, seaborn optional imports, Topic -1 kept

## Slide 15: Performance – LLM Calls
- Async with bounded concurrency; retry/backoff
- `--max_llm_rows` for iteration; `--skip_llm` to reuse
- Code: `src/llm_analysis.py` 198–233, 158–196, 235–309

## Slide 16: Performance – Data Ops & Viz
- Vectorized crosstab/groupby; minimal joins
- Conditional imports (seaborn) to reduce startup overhead
- References: `src/visualizations.py`

## Slide 17: Performance – Clustering Safeguards
- HDBSCAN → KMeans → buckets fallbacks
  - Rationale: always produce cluster labels; degrade gracefully when structure is weak or dependencies are missing
- TF‑IDF max_features; normalized embeddings
  - TF‑IDF (plain-English): score words higher if they’re frequent in one entity but rare overall; this highlights discriminative terms. We cap max_features to keep memory and noise in check.
  - Normalized embeddings (plain-English): rescale vectors to a common length so distances mean the same thing across items. This stabilizes clustering boundaries.
- Code (links/paths)
  - TF‑IDF fallback embedding: `src/entity_clustering.py` 30–34
  - Normalized sentence-transformers embeddings: `src/entity_clustering.py` 26–28

## Slide 18: Reproducibility
- Deterministic parameters (random_state)
- Clear data lifecycle: raw → interim → processed → analysis
- CLI entry points for modular reruns

## Slide 19: Commands (End-to-End)
```bash
python -m src.run_pipeline --limit 1000 --llm_model gpt-4.1-nano --max_llm_rows all --concurrency 1
python -m src.visualizations data/04_analysis/unified_table.csv --out_dir reports/images
```

## Slide 20: Commands (Targeted)
```bash
python - <<'PY'
from src.sentiment_rule_based import batch_process_subset
batch_process_subset('data/02_interim/wildchat_english_notoxic_1000_cleaned.jsonl',
                     'data/04_analysis/sentiment_rule_based_1000.csv')
PY
python -m src.entity_clustering data/04_analysis/unified_table.csv --min_cluster_size 4
```

## Slide 21: Visual Reading Guide
- Multi-sentiment bars: compare LLM/VADER/TextBlob agreement
- Heatmaps: high-count pairs + lower average sentiment = prioritize content/support
- Intent sentiment: target high-volume intents with neutral/negative load

## Slide 22: QA & Extensions
- Add notebooks for interactive slicing
- Expand cluster label curation & topic naming
- Add retrieval-friendly product/brand schemas for better LLM answers

## Slide 23: Appendix – Code Snippets (Representative)
- (See also README “Key code references”)
- Rule-based Sentiment:
```python
# src/sentiment_rule_based.py 33–37, 39–45
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)
tb = TextBlob(text)
polarity, subjectivity = tb.sentiment
```
- BERTopic Fallbacks:
```python
# src/bertopic_pipeline.py 106–117, 125–131
try:
    topics_eff, probs_eff = topic_model.fit_transform(eff_docs)
except ValueError:
    topic_model = build_model(min_df=1, max_df=1.0, ngram=(1, 1))
    topics_eff, probs_eff = topic_model.fit_transform(eff_docs)
```
- Embeddings & Clustering:
```python
# src/entity_clustering.py 20–34, 37–63
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
labels = clusterer.fit_predict(embeddings).astype(int)
```
- LLM Concurrency:
```python
# src/llm_analysis.py 198–233, 158–196
sem = asyncio.Semaphore(concurrency)
async def bound_analyze(r):
    async with sem:
        return await _analyze_single_async(client, r, model)
``` 