### Ticket-3002: BERTopic quality improvements

**Description**
- Improve BERTopic topic quality to reduce stopword-dominant topics (e.g., "the and of to"). Configure a custom vectorizer and stopwords, add n-grams, and improve topic naming.

**Tasks**
- In `src/bertopic_pipeline.py`:
  - Plug in a `CountVectorizer` with:
    - stopwords = sklearn english + custom domain list
    - `ngram_range=(1,3)`, `min_df=5`, `max_df=0.9`
  - Pass `vectorizer_model` to BERTopic; consider `representation_model="KeyBERTInspired"` or `c_tf_idf` tuning.
  - Ensure small-corpus safety remains (UMAP neighbors logic).
- Re-run on the 1,000-doc sample and overwrite `reports/bertopic_topics.csv` and `reports/bertopic_doc_topics.csv`.
- Add a small test asserting that the top-10 words of top topics do not include common stopwords.

**Recommended Tools**
- `sklearn.feature_extraction.text.CountVectorizer`, `BERTopic` representation options

**How to Test**
- Unit test: topic words filter
- Integration: run pipeline on tiny synthetic docs; assert outputs exist

**Reminder to AI Coder**
- Log domain stopwords chosen in Troubleshooting.md 