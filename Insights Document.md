# AI Conversation Insights

This document synthesizes insights from the unified conversation table and the generated visuals. It is intended to help brands understand how they appear in AI-mediated conversations and where to prioritize action.

## Executive takeaways

- Creative and exploratory usage dominates: A large share of conversations land in creative writing and general chat; these intents skew neutral/positive overall. 
- Product inquiries are common.
- Technical support questions persist: There is sustained support demand with a noticeable neutral baseline. Clear troubleshooting guides, decision-trees, and error FAQs designed for LLMs can convert neutral into positive experiences.

## Company- and product-specific observations

- OpenAI (organization)
  - Sentiment distribution is mixed-to-positive across methods (LLM: neutral 2, positive 4; VADER: positive 5, neutral 1).
  - Graph: see `reports/images/entities_organization_multi.png` (find the OpenAI bar across LLM/VADER/TextBlob panels).
  - Query (reproduce counts):
    ```python
    import pandas as pd
    df = pd.read_csv('data/04_analysis/unified_table.csv')
    e = 'openai'
    sub = df[df.llm_entity.str.lower()==e]
    def norm(s):
        s=str(s).lower().strip()
        return 'positive' if s in {'positive','pos','+'} else ('negative' if s in {'negative','neg','-'} else 'neutral')
    # LLM per-entity sentiment
    sub.llm_entity_sentiment.map(norm).value_counts()
    # VADER/TextBlob labels
    def vlabel(x):
        try: x=float(x)
        except: return 'neutral'
        return 'positive' if x>=0.05 else ('negative' if x<=-0.05 else 'neutral')
    def tlabel(x):
        try: x=float(x)
        except: return 'neutral'
        return 'positive' if x>0 else ('negative' if x<0 else 'neutral')
    sub.vader_compound.map(vlabel).value_counts(), sub.tb_polarity.map(tlabel).value_counts()
    ```

- TF1 (brand)
  - Strong positive skew across methods (LLM: 7 pos / 1 neutral; VADER/TB: majority positive).
  - Graph: see `reports/images/entities_brand_multi.png` (locate TF1 among brands).
  - Query (reproduce counts):
    ```python
    import pandas as pd
    df = pd.read_csv('data/04_analysis/unified_table.csv')
    e = 'tf1'
    sub = df[df.llm_entity.str.lower()==e]
    def norm(s):
        s=str(s).lower().strip()
        return 'positive' if s in {'positive','pos','+'} else ('negative' if s in {'negative','neg','-'} else 'neutral')
    sub.llm_entity_sentiment.map(norm).value_counts()
    ```

- Products (generic/product-like mentions)
  - Frequent generic mentions (e.g., “video”, “light bulb”, “dynamic microphone”) trend neutral by LLM and positive by rule-based.
  - Graphs: `reports/images/entities_possible_product_multi.png` (ambiguous product-like mentions); sentiment by intent (context) `reports/images/entity_sentiment_by_category.png`.
  - Query (top product-like mentions + sentiment):
    ```python
    import pandas as pd
    df = pd.read_csv('data/04_analysis/unified_table.csv')
    top_prods = df[df.llm_entity_category=='product'].llm_entity.value_counts().head(10)
    top_prods
    # Example sentiment for one product term
    term = top_prods.index[0]
    df[df.llm_entity==term].llm_entity_sentiment.value_counts()
    ```

- Regions/places surfacing as possible brands/stores
  - Examples like “Morocco” show highly positive sentiment (LLM/VADER/TB overwhelmingly positive), indicating travel/culture contexts that are inspirational.
  - Graph: `reports/images/entities_possible_brand_multi.png` (locate “Morocco”).
  - Query (reproduce counts):
    ```python
    import pandas as pd
    df = pd.read_csv('data/04_analysis/unified_table.csv')
    for name in ['morocco','Morocco']:
        sub = df[df.llm_entity==name]
        def norm(s):
            s=str(s).lower().strip()
            return 'positive' if s in {'positive','pos','+'} else ('negative' if s in {'negative','neg','-'} else 'neutral')
        print(name, sub.llm_entity_sentiment.map(norm).value_counts().to_dict())
    ```

## How industries show up in AI interfaces

- Media/Entertainment: Positive bias when users ask for summaries, character info, or content-related creative tasks (e.g., TF1, “Game of Thrones” related entities). Rich metadata (synopses, cast, scenes) aids AI responses and keeps brand top-of-mind.
  - Graphs to inspect:
    - Brands/Orgs multi-sentiment: `reports/images/entities_brand_multi.png`, `reports/images/entities_organization_multi.png`
    - Conversation sentiment by intent: `reports/images/entity_sentiment_by_category.png` (Creative Writing)
  - Queries:
    ```python
    import pandas as pd
    df = pd.read_csv('data/04_analysis/unified_table.csv')
    # Entertainment-related entities (examples)
    ents = ['tf1','game of thrones']
    def norm(s):
        s=str(s).lower().strip()
        return 'positive' if s in {'positive','pos','+'} else ('negative' if s in {'negative','neg','-'} else 'neutral')
    for e in ents:
        sub = df[df.llm_entity.str.lower()==e]
        print(e, sub.llm_entity_sentiment.map(norm).value_counts().to_dict())
    # Conversation-level Creative Writing intent sentiment
    conv = df[['conversation_id','llm_intent','llm_sentiment']].drop_duplicates().copy()
    conv['label'] = conv['llm_sentiment'].map(norm)
    conv[conv.llm_intent=='Creative Writing'].label.value_counts()
    ```

- Consumer Electronics/Hardware: Neutral-to-positive tone with practical “which device” and “how to” prompts. Publishing compatibility matrices, step-by-step guides, and troubleshooting flows can shift neutral to positive.
  - Graphs to inspect:
    - Possible/Product multi-sentiment: `reports/images/entities_possible_product_multi.png`
    - Mentions heatmaps for brand→product: `reports/images/brand_product_mentions_heatmap.png`
  - Queries:
    ```python
    import pandas as pd
    df = pd.read_csv('data/04_analysis/unified_table.csv')
    # Top product-like terms and their sentiment
    prods = df[df.llm_entity_category=='product'].llm_entity.value_counts().head(15)
    print(prods)
    term = prods.index[0]
    print(term, df[df.llm_entity==term].llm_entity_sentiment.value_counts().to_dict())
    # Find co-mentioned brand→product pairs
    brands = df[df.llm_entity_category.isin(['brand','possible_brand'])][['conversation_id','llm_entity']].rename(columns={'llm_entity':'brand_like'})
    prod = df[df.llm_entity_category.isin(['product','possible_product'])][['conversation_id','llm_entity']].rename(columns={'llm_entity':'product_like'})
    merged = brands.merge(prod, on='conversation_id', how='inner')
    merged.groupby(['brand_like','product_like']).size().sort_values(ascending=False).head(20)
    ```

- AI/Software Platforms: Mixed but broadly constructive; users ask operational questions (capabilities, usage patterns). Clear developer docs, API examples, and security/limits pages should be optimized for LLM ingestion (concise, canonical, up-to-date).
  - Graphs to inspect:
    - Organizations multi-sentiment: `reports/images/entities_organization_multi.png`
    - Conversation sentiment by intent (General Chat, Technical Support): `reports/images/entity_sentiment_by_category.png`
  - Queries:
    ```python
    import pandas as pd
    df = pd.read_csv('data/04_analysis/unified_table.csv')
    # Example orgs
    for e in ['openai','microsoft','google']:
        sub = df[df.llm_entity.str.lower()==e]
        print(e, sub.llm_entity_sentiment.str.lower().value_counts().to_dict())
    # Support/General Chat sentiment distribution
    def norm(s):
        s=str(s).lower().strip()
        return 'positive' if s in {'positive','pos','+'} else ('negative' if s in {'negative','neg','-'} else 'neutral')
    conv = df[['conversation_id','llm_intent','llm_sentiment']].drop_duplicates().copy()
    conv['label'] = conv['llm_sentiment'].map(norm)
    conv.groupby('llm_intent').label.value_counts().unstack(fill_value=0).loc[['Technical Support','General Chat']]
    ```

- Commerce/Stores: Sparse but present; users want availability and where-to-buy. Structured inventory/availability and store locators that LLMs can parse will convert vague intent into directed actions.
  - Graphs to inspect:
    - Store multi-sentiment: `reports/images/entities_possible_store_multi.png`
    - Store→Product mentions heatmap: `reports/images/store_product_mentions_heatmap.png`
  - Queries:
    ```python
    import pandas as pd
    df = pd.read_csv('data/04_analysis/unified_table.csv')
    # Top store-like entities
    stores = df[df.llm_entity_category=='possible_store'].llm_entity.value_counts().head(10)
    print(stores)
    # Store→product pairs (mentions)
    stores_df = df[df.llm_entity_category.isin(['store','possible_store'])][['conversation_id','llm_entity']].rename(columns={'llm_entity':'store_like'})
    prods_df = df[df.llm_entity_category.isin(['product','possible_product'])][['conversation_id','llm_entity']].rename(columns={'llm_entity':'product_like'})
    merged = stores_df.merge(prods_df, on='conversation_id', how='inner')
    merged.groupby(['store_like','product_like']).size().sort_values(ascending=False).head(20)
    ```

## Evidence (selected visuals)

- Conversation sentiment by intent: `reports/images/entity_sentiment_by_category.png`
- Entities — Brand multi-sentiment: `reports/images/entities_brand_multi.png`
- Entities — Organization multi-sentiment: `reports/images/entities_organization_multi.png`
- Entities — Possible brand/store/product multi-sentiment: 
  - `reports/images/entities_possible_brand_multi.png`
  - `reports/images/entities_possible_store_multi.png`
  - `reports/images/entities_possible_product_multi.png`
- Mentions heatmaps (co-mentions with average sentiment coloring):
  - Store→Product: `reports/images/store_product_mentions_heatmap.png`
  - Brand→Product: `reports/images/brand_product_mentions_heatmap.png`
  - Organization→Product: `reports/images/organization_product_mentions_heatmap.png`

## Notes on methodology

- Mentions charts use pure frequency (row counts). An entity can appear in multiple sentiment buckets based on mentions.
- Average sentiment charts compute entity-weighted averages (−1/0/+1 per mention → mean per entity → mean across entities in category).
- Conversation sentiment by intent is deduplicated by conversation to reflect user-level distribution rather than per-entity counts. 