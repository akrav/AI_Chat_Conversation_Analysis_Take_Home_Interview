from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_placeholder(out_path: str, title: str, subtitle: str = "") -> str:
    plt.figure(figsize=(8, 3))
    plt.axis("off")
    plt.title(title)
    if subtitle:
        plt.text(0.5, 0.4, subtitle, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


# --- Existing plots (unchanged) ---

def plot_topic_frequency_overall(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    df = pd.read_csv(unified_csv)
    if "topic_name" not in df.columns:
        return _save_placeholder(os.path.join(out_dir, "topic_frequency_overall.png"), "Missing column: topic_name")
    agg = df.groupby("topic_name", dropna=False)["conversation_id"].nunique().sort_values(ascending=False).head(25)
    if agg.empty:
        return _save_placeholder(os.path.join(out_dir, "topic_frequency_overall.png"), "No topics to display")
    plt.figure(figsize=(12, 6))
    agg.plot(kind="bar")
    plt.title("Top Topic Frequency (by unique conversation)")
    plt.ylabel("Conversations")
    plt.tight_layout()
    out = os.path.join(out_dir, "topic_frequency_overall.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_intents_distribution(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    df = pd.read_csv(unified_csv)
    if "llm_intent" not in df.columns:
        return _save_placeholder(os.path.join(out_dir, "intent_distribution.png"), "Missing column: llm_intent")
    counts = df.groupby("llm_intent")["conversation_id"].nunique().sort_values(ascending=False)
    if counts.empty:
        return _save_placeholder(os.path.join(out_dir, "intent_distribution.png"), "No intents to display")
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Intent Distribution (unique conversations)")
    plt.ylabel("Conversations")
    plt.tight_layout()
    out = os.path.join(out_dir, "intent_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_sentiment_by_topic(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    df = pd.read_csv(unified_csv)
    req = {"topic_name", "llm_sentiment", "conversation_id"}
    if not req.issubset(df.columns):
        return _save_placeholder(os.path.join(out_dir, "sentiment_by_topic.png"), "Missing columns", str(req))
    pivot = (
        df.pivot_table(index="topic_name", columns="llm_sentiment", values="conversation_id", aggfunc=lambda x: len(set(x)))
        .fillna(0)
        .astype(int)
    )
    if pivot.empty:
        return _save_placeholder(os.path.join(out_dir, "sentiment_by_topic.png"), "No sentiment/topic data")
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(20).index]
    pivot.plot(kind="bar", stacked=True, figsize=(14, 7))
    plt.title("Sentiment by Topic (unique conversations)")
    plt.ylabel("Conversations")
    plt.tight_layout()
    out = os.path.join(out_dir, "sentiment_by_topic.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_top_entity_clusters(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    df = pd.read_csv(unified_csv)
    out = os.path.join(out_dir, "entity_clusters_top.png")
    # Fallback: if clustering not available, show top entities overall
    if "llm_entity" not in df.columns:
        return _save_placeholder(out, "Missing column: llm_entity")
    if "entity_cluster_id" not in df.columns:
        counts = df["llm_entity"].value_counts().head(20)
        if counts.empty:
            return _save_placeholder(out, "No entities to display")
        plt.figure(figsize=(12, 6))
        counts.plot(kind="bar")
        plt.title("Top Entities (no clustering available)")
        plt.ylabel("Mentions")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out
    agg = (
        df[df["entity_cluster_id"].ge(0)].groupby("entity_cluster_id")["llm_entity"].nunique().sort_values(ascending=False).head(20)
    )
    if agg.empty:
        return _save_placeholder(out, "No clusters to display")
    plt.figure(figsize=(12, 6))
    agg.plot(kind="bar")
    plt.title("Top Entity Clusters (unique entities)")
    plt.ylabel("Unique Entities")

    plt.savefig(out, dpi=150)
    plt.close()
    return out


# --- New plots ---

def plot_top_entity_clusters_labeled(unified_csv: str, out_dir: str, top_n: int = 12) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_clusters_top_labeled.png")
    df = pd.read_csv(unified_csv)
    # Fallback when clustering is unavailable
    if "entity_cluster_id" not in df.columns or "llm_entity" not in df.columns:
        if "llm_entity" not in df.columns:
            return _save_placeholder(out, "Missing columns for labeled clusters")
        counts = df["llm_entity"].value_counts().head(top_n)
        if counts.empty:
            return _save_placeholder(out, "No entities to label")
        plt.figure(figsize=(14, 6))
        counts.plot(kind="bar", color="#6baed6")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mentions")
        plt.title("Top Entities (no clustering available)")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out
    # Build labels: top-3 entities per cluster
    labels: List[Tuple[int, str]] = []
    for cid, grp in df[df["entity_cluster_id"].ge(0)].groupby("entity_cluster_id"):
        top_entities = grp["llm_entity"].value_counts().head(3).index.tolist()
        labels.append((cid, ", ".join(map(str, top_entities))))
    lab_df = pd.DataFrame(labels, columns=["entity_cluster_id", "label"])
    agg = df[df["entity_cluster_id"].ge(0)].groupby("entity_cluster_id")["llm_entity"].nunique().sort_values(ascending=False).head(top_n)
    if agg.empty:
        return _save_placeholder(out, "No clusters to label")
    plot_df = agg.reset_index().merge(lab_df, on="entity_cluster_id", how="left")
    # Ensure left-to-right decreasing order
    plot_df = plot_df.sort_values("llm_entity", ascending=False)
    plt.figure(figsize=(14, 6))
    plt.bar(plot_df["label"], plot_df["llm_entity"], color="#6baed6")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Unique Entities")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entity_clusters_by_category(unified_csv: str, out_dir: str, top_k: int = 15) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_clusters_by_category.png")
    df = pd.read_csv(unified_csv)
    req = {"entity_cluster_id", "llm_entity_category", "conversation_id"}
    if not req.issubset(df.columns):
        # Fallback: count unique conversations per category
        if not {"llm_entity_category", "conversation_id"}.issubset(df.columns):
            return _save_placeholder(out, "Missing columns for clusters by category")
        counts = df.groupby("llm_entity_category")["conversation_id"].nunique().sort_values(ascending=False).head(top_k)
        if counts.empty:
            return _save_placeholder(out, "No cluster/category data")
        counts.plot(kind="bar", figsize=(12, 6))
        plt.title("Conversations by Category (unique conversations)")
        plt.ylabel("Conversations")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out
    # Unique conversations per category with at least one cluster id
    tmp = df[df["entity_cluster_id"].ge(0)].groupby("llm_entity_category")["conversation_id"].nunique()
    counts = tmp.sort_values(ascending=False).head(top_k)
    if counts.empty:
        return _save_placeholder(out, "No cluster/category data")
    counts.plot(kind="bar", figsize=(12, 6))
    plt.title("Entity Clusters by Category (unique conversations)")
    plt.ylabel("Conversations")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _product_heatmap_with_avg(df: pd.DataFrame, out: str, row_label: str, top_rows: int, top_cols: int) -> str:
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None  # type: ignore
    import numpy as np  # ensure available for fallback and sizing
    if df.empty:
        return _save_placeholder(out, "No data for heatmap")
    # Normalize sentiments and map to scores
    df = df.copy()
    df["sent_norm"] = df["product_sentiment"].apply(_normalize_sentiment_label)
    score_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    df["sent_score"] = df["sent_norm"].map(score_map).fillna(0.0)
    # Aggregate by pair
    agg = df.groupby([row_label, "product_like"]).agg(
        conv_count=("conversation_id", lambda x: len(set(x))),
        avg_score=("sent_score", "mean"),
    ).reset_index()
    if agg.empty:
        return _save_placeholder(out, "No data for heatmap")
    counts_pivot = agg.pivot(index=row_label, columns="product_like", values="conv_count").fillna(0)
    avg_pivot = agg.pivot(index=row_label, columns="product_like", values="avg_score").fillna(0)
    # Select top rows/cols by counts to ensure a non-trivial grid
    row_order = counts_pivot.sum(axis=1).sort_values(ascending=False).head(top_rows).index
    col_order = counts_pivot.sum(axis=0).sort_values(ascending=False).head(top_cols).index
    # If selection collapses to <2 in either dimension, expand to all (capped at 30)
    if len(row_order) < 2:
        row_order = counts_pivot.sum(axis=1).sort_values(ascending=False).head(min(30, counts_pivot.shape[0])).index
    if len(col_order) < 2:
        col_order = counts_pivot.sum(axis=0).sort_values(ascending=False).head(min(30, counts_pivot.shape[1])).index
    counts_pivot = counts_pivot.loc[row_order, col_order]
    avg_pivot = avg_pivot.loc[row_order, col_order]
    if counts_pivot.size == 0:
        return _save_placeholder(out, "No data after limiting top rows/cols")
    plt.figure(figsize=(max(10, len(col_order)*0.6), max(6, len(row_order)*0.5)))
    if sns is not None:
        ax = sns.heatmap(avg_pivot, annot=counts_pivot, fmt=".0f", linewidths=.5, cmap="RdYlGn", vmin=-1, vmax=1, center=0, cbar=True)
    else:
        plt.imshow(avg_pivot.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax = plt.gca()
        for (i, j), val in np.ndenumerate(counts_pivot.values):
            ax.text(j + 0.5, i + 0.5, int(val), ha='center', va='center', color='black')
        ax.set_xticks(np.arange(len(col_order)) + 0.5)
        ax.set_yticks(np.arange(len(row_order)) + 0.5)
        ax.set_xticklabels(col_order)
        ax.set_yticklabels(row_order)
    plt.title(f"Avg product sentiment (color) with counts (numbers) by {row_label.replace('_',' ').title()}")
    plt.xlabel("Product")
    plt.ylabel(row_label.replace('_',' ').title())
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_store_product_heatmap(unified_csv: str, out_dir: str, top_rows: int = 30, top_cols: int = 30) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "store_product_heatmap.png")
    df = pd.read_csv(unified_csv)
    req = {"conversation_id", "llm_entity", "llm_entity_category", "llm_entity_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for store/product heatmap")
    stores = {"store", "possible_store"}
    products = {"product", "possible_product"}
    sdf = df[df["llm_entity_category"].isin(stores)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "store_like", "llm_entity_sentiment": "store_sentiment"})
    pdf = df[df["llm_entity_category"].isin(products)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "product_like", "llm_entity_sentiment": "product_sentiment"})
    merged = sdf.merge(pdf, on="conversation_id", how="inner")
    if merged.empty:
        return _save_placeholder(out, "No store-product co-mentions")
    return _product_heatmap_with_avg(merged[["conversation_id", "store_like", "product_like", "product_sentiment"]], out, row_label="store_like", top_rows=top_rows, top_cols=top_cols)


def plot_brand_product_heatmap(unified_csv: str, out_dir: str, top_rows: int = 20, top_cols: int = 20) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "brand_product_heatmap.png")
    df = pd.read_csv(unified_csv)
    req = {"conversation_id", "llm_entity", "llm_entity_category", "llm_entity_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for brand/product heatmap")
    brands = {"brand", "possible_brand"}
    products = {"product", "possible_product"}
    bdf = df[df["llm_entity_category"].isin(brands)][["conversation_id", "llm_entity"]].rename(columns={"llm_entity": "brand_like"})
    pdf = df[df["llm_entity_category"].isin(products)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "product_like", "llm_entity_sentiment": "product_sentiment"})
    merged = bdf.merge(pdf, on="conversation_id", how="inner")
    if merged.empty:
        return _save_placeholder(out, "No brand-product co-mentions")
    return _product_heatmap_with_avg(merged[["conversation_id", "brand_like", "product_like", "product_sentiment"]].rename(columns={"brand_like":"brand_like"}), out, row_label="brand_like", top_rows=top_rows, top_cols=top_cols)


def plot_organization_product_heatmap(unified_csv: str, out_dir: str, top_rows: int = 20, top_cols: int = 20) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "organization_product_heatmap.png")
    df = pd.read_csv(unified_csv)
    req = {"conversation_id", "llm_entity", "llm_entity_category", "llm_entity_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for organization/product heatmap")
    orgs = {"organization"}
    products = {"product", "possible_product"}
    odf = df[df["llm_entity_category"].isin(orgs)][["conversation_id", "llm_entity"]].rename(columns={"llm_entity": "org_like"})
    pdf = df[df["llm_entity_category"].isin(products)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "product_like", "llm_entity_sentiment": "product_sentiment"})
    merged = odf.merge(pdf, on="conversation_id", how="inner")
    if merged.empty:
        return _save_placeholder(out, "No organization-product co-mentions")
    return _product_heatmap_with_avg(merged[["conversation_id", "org_like", "product_like", "product_sentiment"]].rename(columns={"org_like":"org_like"}), out, row_label="org_like", top_rows=top_rows, top_cols=top_cols)


def plot_entity_category_counts(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    df = pd.read_csv(unified_csv)
    out = os.path.join(out_dir, "entity_category_counts.png")
    if "llm_entity_category" not in df.columns:
        return _save_placeholder(out, "Missing column: llm_entity_category")
    # Count unique entities per category (fallback to row count if llm_entity missing)
    if "llm_entity" in df.columns:
        counts = df.groupby("llm_entity_category")["llm_entity"].nunique().sort_values(ascending=False)
    else:
        counts = df["llm_entity_category"].value_counts()
    if counts.empty:
        return _save_placeholder(out, "No entity categories to display")
    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title("Entity Category Counts (unique entities)")
    plt.ylabel("Unique Entities")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _normalize_sentiment_label(s: str | None) -> str:
    if not isinstance(s, str):
        return "neutral"
    s = s.strip().lower()
    if s in {"pos", "positive", "+"}:
        return "positive"
    if s in {"neg", "negative", "-"}:
        return "negative"
    return "neutral"


def _vader_label(row: pd.Series) -> str:
    c = row.get("vader_compound", None)
    try:
        c = float(c)
    except Exception:
        return "neutral"
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


def _textblob_label(row: pd.Series) -> str:
    try:
        p = float(row.get("tb_polarity", None))
    except Exception:
        return "neutral"
    if p > 0:
        return "positive"
    if p < 0:
        return "negative"
    return "neutral"


def _plot_overlap_heatmap(ct: pd.DataFrame, title: str, out: str) -> str:
    if ct.empty:
        return _save_placeholder(out, title, "No overlap data")
    try:
        import seaborn as sns  # type: ignore
        plt.figure(figsize=(6, 5))
        sns.heatmap(ct, annot=True, fmt=".0f", cmap="Blues")
        plt.title(title)
    except Exception:
        plt.figure(figsize=(6, 5))
        plt.imshow(ct.values, cmap="Blues")
        plt.xticks(range(len(ct.columns)), ct.columns, rotation=45, ha="right")
        plt.yticks(range(len(ct.index)), ct.index)
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_sentiment_overlap_vader(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "sentiment_overlap_llm_vs_vader.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_sentiment", "vader_compound"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for VADER overlap")
    df = df.copy()
    df["llm_sentiment_norm"] = df["llm_sentiment"].apply(_normalize_sentiment_label)
    df["vader_label"] = df.apply(_vader_label, axis=1)
    ct = pd.crosstab(df["llm_sentiment_norm"], df["vader_label"]).reindex(index=["negative","neutral","positive"], columns=["negative","neutral","positive"], fill_value=0)
    return _plot_overlap_heatmap(ct, "Sentiment Overlap: LLM vs VADER", out)


def plot_sentiment_overlap_textblob(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "sentiment_overlap_llm_vs_textblob.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_sentiment", "tb_polarity"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for TextBlob overlap")
    df = df.copy()
    df["llm_sentiment_norm"] = df["llm_sentiment"].apply(_normalize_sentiment_label)
    df["tb_label"] = df.apply(_textblob_label, axis=1)
    ct = pd.crosstab(df["llm_sentiment_norm"], df["tb_label"]).reindex(index=["negative","neutral","positive"], columns=["negative","neutral","positive"], fill_value=0)
    return _plot_overlap_heatmap(ct, "Sentiment Overlap: LLM vs TextBlob", out)


def plot_entity_sentiment_by_category(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_sentiment_by_category.png")
    df = pd.read_csv(unified_csv)
    # Use overall conversation sentiment by conversation category (intent) if available
    if {"llm_intent", "llm_sentiment"}.issubset(df.columns):
        dfx = df[["conversation_id", "llm_intent", "llm_sentiment"]].drop_duplicates()
        dfx["label"] = dfx["llm_sentiment"].apply(_normalize_sentiment_label)
        pivot = pd.crosstab(dfx["llm_intent"], dfx["label"]).fillna(0)
        if pivot.empty:
            return _save_placeholder(out, "No conversation sentiment by intent data")
        order_rows = pivot.sum(axis=1).sort_values(ascending=False).head(25).index
        pivot = pivot.loc[order_rows]
        for col in ["negative", "neutral", "positive"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["negative", "neutral", "positive"]]
        pivot.plot(kind="bar", stacked=True, figsize=(16, 8))
        plt.title("Conversation Sentiment by Intent (unique conversations)")
        plt.ylabel("Conversations")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out
    # Fallback to previous entity-level mentions pivot
    req = {"llm_entity_category", "llm_entity_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for sentiment/category")
    pivot = pd.crosstab(df["llm_entity_category"], df["llm_entity_sentiment"]).fillna(0)
    if pivot.empty:
        return _save_placeholder(out, "No entity sentiment/category data")
    order_rows = pivot.sum(axis=1).sort_values(ascending=False).head(20).index
    pivot = pivot.loc[order_rows]
    for col in ["negative", "neutral", "positive"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["negative", "neutral", "positive"]]
    pivot.plot(kind="bar", stacked=True, figsize=(14, 7))
    plt.title("Entity Sentiment by Category (mentions)")
    plt.ylabel("Mentions")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entity_sentiment_by_category_llm(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_sentiment_by_category_llm.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_entity_category", "llm_sentiment", "llm_entity"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for LLM entity/category")
    pivot = pd.crosstab(df["llm_entity_category"], df["llm_sentiment"]).fillna(0)
    if pivot.empty:
        return _save_placeholder(out, "No LLM entity/category data")
    order_rows = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order_rows]
    for col in ["negative", "neutral", "positive"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["negative", "neutral", "positive"]]
    pivot.plot(kind="bar", stacked=True, figsize=(14, 7))
    plt.title("Entity Mentions by Category split by LLM sentiment")
    plt.ylabel("Mentions")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entity_sentiment_by_category_vader(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_sentiment_by_category_vader.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_entity_category", "vader_compound", "llm_entity"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for VADER entity/category")
    df = df.copy()
    df["vader_label"] = df.apply(_vader_label, axis=1)
    pivot = pd.crosstab(df["llm_entity_category"], df["vader_label"]).fillna(0)
    if pivot.empty:
        return _save_placeholder(out, "No VADER entity/category data")
    order_rows = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order_rows]
    for col in ["negative", "neutral", "positive"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["negative", "neutral", "positive"]]
    pivot.plot(kind="bar", stacked=True, figsize=(14, 7))
    plt.title("Entity Mentions by Category split by VADER sentiment")
    plt.ylabel("Mentions")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entity_sentiment_by_category_textblob(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_sentiment_by_category_textblob.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_entity_category", "tb_polarity", "llm_entity"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for TextBlob entity/category")
    df = df.copy()
    df["tb_label"] = df.apply(_textblob_label, axis=1)
    pivot = pd.crosstab(df["llm_entity_category"], df["tb_label"]).fillna(0)
    if pivot.empty:
        return _save_placeholder(out, "No TextBlob entity/category data")
    order_rows = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order_rows]
    for col in ["negative", "neutral", "positive"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["negative", "neutral", "positive"]]
    pivot.plot(kind="bar", stacked=True, figsize=(14, 7))
    plt.title("Entity Mentions by Category split by TextBlob sentiment")
    plt.ylabel("Mentions")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_top_entities_overall(unified_csv: str, out_dir: str, top_n: int = 25) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "top_entities_overall.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_entity", "conversation_id"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing column: llm_entity")
    # Count unique conversations per entity
    counts = df.groupby("llm_entity")["conversation_id"].nunique().sort_values(ascending=False).head(top_n)
    if counts.empty:
        return _save_placeholder(out, "No entities to display")
    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title("Top Entities Overall (unique conversations)")
    plt.ylabel("Conversations")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_brand_store_org_product_sentiment(unified_csv: str, out_dir: str, top_pairs: int = 30) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "brand_store_org_product_sentiment.png")
    df = pd.read_csv(unified_csv)
    req = {"conversation_id", "llm_entity", "llm_entity_category", "llm_entity_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for brand/store/org product sentiment")
    brands = {"brand", "possible_brand", "store", "possible_store", "organization"}
    products = {"product", "possible_product"}
    bdf = df[df["llm_entity_category"].isin(brands)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "brand_like"})
    pdf = df[df["llm_entity_category"].isin(products)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "product_like", "llm_entity_sentiment": "product_sentiment"})
    if bdf.empty or pdf.empty:
        return _save_placeholder(out, "Insufficient brand/product co-mentions")
    merged = bdf.merge(pdf, on="conversation_id", how="inner")
    if merged.empty:
        return _save_placeholder(out, "No brand-product co-mentions")
    # Count by pair and brand sentiment
    grp = merged.groupby(["brand_like", "product_like", "llm_entity_sentiment"]).size().unstack(fill_value=0)
    totals = grp.sum(axis=1)
    grp = grp.loc[totals.sort_values(ascending=False).head(top_pairs).index]
    labels = [f"{b} -> {p}" for b, p in grp.index]

    import numpy as np
    plt.figure(figsize=(22, 10))
    bottom = np.zeros(len(grp))
    x = range(len(grp))
    for sentiment, color in zip(["negative", "neutral", "positive"], ["#ef3b2c", "#9e9ac8", "#2ca25f"]):
        vals = grp[sentiment].values if sentiment in grp.columns else np.zeros(len(grp))
        plt.bar(x, vals, bottom=bottom, color=color, label=sentiment)
        bottom += vals
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Mentions")
    plt.title("Brand/Store/Org -> Product co-mentions with sentiment (by brand sentiment)")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _entities_multi_sentiment_chart(df: pd.DataFrame, out: str, title: str, top_n: int = 30) -> str:
    if df.empty:
        return _save_placeholder(out, title, "No entities after filter")
    # Determine top entities by total mentions
    top_entities = df["llm_entity"].value_counts().head(top_n).index.tolist()

    def pivot_for(mapper: str) -> pd.DataFrame:
        dfx = df.copy()
        if mapper == "llm":
            dfx["label"] = dfx["llm_entity_sentiment"].apply(_normalize_sentiment_label)
        elif mapper == "vader":
            dfx["label"] = dfx.apply(_vader_label, axis=1)
        else:
            dfx["label"] = dfx.apply(_textblob_label, axis=1)
        # Frequency mentions per entity per sentiment
        p = pd.crosstab(dfx["llm_entity"], dfx["label"]).reindex(index=top_entities).fillna(0)
        for col in ["negative", "neutral", "positive"]:
            if col not in p.columns:
                p[col] = 0
        # Maintain left-to-right decreasing order using totals
        totals = p.sum(axis=1)
        p = p.loc[totals.sort_values(ascending=False).index]
        return p[["negative", "neutral", "positive"]]

    llm_p = pivot_for("llm")
    vader_p = pivot_for("vader")
    tb_p = pivot_for("tb")

    import numpy as np
    fig, axes = plt.subplots(3, 1, figsize=(22, 18), sharex=True)
    for ax, name, pvt in zip(axes, ["LLM", "VADER", "TextBlob"], [llm_p, vader_p, tb_p]):
        bottom = np.zeros(len(pvt))
        x = range(len(pvt.index))
        for sentiment, color in zip(["negative", "neutral", "positive"], ["#ef3b2c", "#9e9ac8", "#2ca25f"]):
            vals = pvt[sentiment].values
            ax.bar(x, vals, bottom=bottom, color=color, label=sentiment if ax is axes[0] else None)
            bottom += vals
        ax.set_title(name)
        ax.set_ylabel("Mentions")
    axes[-1].set_xticks(list(range(len(llm_p.index))))
    axes[-1].set_xticklabels(llm_p.index, rotation=45, ha="right")
    axes[0].legend(title="Sentiment")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entities_by_categories_multi_sentiment(unified_csv: str, out_dir: str, categories: set[str], filename: str, title: str, top_n: int = 30) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, filename)
    df = pd.read_csv(unified_csv)
    req = {"llm_entity", "llm_entity_category", "llm_entity_sentiment", "vader_compound", "tb_polarity"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing cols for multi-sentiment entities chart")
    dff = df[df["llm_entity_category"].isin(categories)].copy()
    return _entities_multi_sentiment_chart(dff, out, title, top_n=top_n)


def plot_entities_brand_multi(unified_csv: str, out_dir: str) -> str:
    return plot_entities_by_categories_multi_sentiment(unified_csv, out_dir, {"brand"}, "entities_brand_multi.png", "Entities: Brand (LLM/VADER/TextBlob)")


def plot_entities_org_multi(unified_csv: str, out_dir: str) -> str:
    return plot_entities_by_categories_multi_sentiment(unified_csv, out_dir, {"organization"}, "entities_organization_multi.png", "Entities: Organization (LLM/VADER/TextBlob)")


def plot_entities_possible_product_multi(unified_csv: str, out_dir: str) -> str:
    return plot_entities_by_categories_multi_sentiment(unified_csv, out_dir, {"possible_product"}, "entities_possible_product_multi.png", "Entities: Possible Product (LLM/VADER/TextBlob)")


def plot_entities_possible_store_multi(unified_csv: str, out_dir: str) -> str:
    return plot_entities_by_categories_multi_sentiment(unified_csv, out_dir, {"possible_store"}, "entities_possible_store_multi.png", "Entities: Possible Store (LLM/VADER/TextBlob)")


def plot_entities_possible_brand_multi(unified_csv: str, out_dir: str) -> str:
    return plot_entities_by_categories_multi_sentiment(unified_csv, out_dir, {"possible_brand"}, "entities_possible_brand_multi.png", "Entities: Possible Brand (LLM/VADER/TextBlob)")


def plot_entity_sentiment_by_category_combined(unified_csv: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_sentiment_by_category_combined.png")
    df = pd.read_csv(unified_csv)
    req_cols = {"llm_entity_category", "llm_entity", "llm_sentiment", "vader_compound", "tb_polarity"}
    if not req_cols.issubset(df.columns):
        missing = req_cols.difference(df.columns)
        return _save_placeholder(out, "Missing columns for combined chart", ", ".join(sorted(missing)))

    categories = df.groupby("llm_entity_category")["llm_entity"].nunique().sort_values(ascending=False).head(25).index.tolist()
    if not categories:
        return _save_placeholder(out, "No categories to plot")

    def counts_by_sentiment(mapper: str) -> pd.DataFrame:
        dfx = df.copy()
        if mapper == "llm":
            dfx["label"] = dfx["llm_sentiment"].apply(_normalize_sentiment_label)
        elif mapper == "vader":
            dfx["label"] = dfx.apply(_vader_label, axis=1)
        else:
            dfx["label"] = dfx.apply(_textblob_label, axis=1)
        pivot = pd.crosstab(dfx["llm_entity_category"], dfx["label"]).fillna(0)
        pivot = pivot.reindex(index=categories).fillna(0)
        for col in ["negative", "neutral", "positive"]:
            if col not in pivot.columns:
                pivot[col] = 0
        return pivot[["negative", "neutral", "positive"]]

    llm_ct = counts_by_sentiment("llm")
    vader_ct = counts_by_sentiment("vader")
    tb_ct = counts_by_sentiment("tb")

    import numpy as np
    fig, axes = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
    for ax, title, ct in zip(axes, ["LLM", "VADER", "TextBlob"], [llm_ct, vader_ct, tb_ct]):
        bottom = np.zeros(len(ct))
        x = range(len(ct.index))
        for sentiment, color in zip(["negative", "neutral", "positive"], ["#ef3b2c", "#9e9ac8", "#2ca25f"]):
            vals = ct[sentiment].values
            ax.bar(x, vals, bottom=bottom, label=sentiment if ax is axes[0] else None, color=color)
            bottom += vals
        ax.set_title(title)
        ax.set_ylabel("Unique Entities")
    axes[-1].set_xticks(list(range(len(llm_ct.index))))
    axes[-1].set_xticklabels(llm_ct.index, rotation=45, ha="right")
    axes[0].legend(title="Sentiment")
    fig.suptitle("Entity Mentions by Category (LLM, VADER, TextBlob)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_brand_store_org_product_multi(unified_csv: str, out_dir: str, top_pairs: int = 30) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "brand_store_org_product_multi.png")
    df = pd.read_csv(unified_csv)
    req = {"conversation_id", "llm_entity", "llm_entity_category", "llm_entity_sentiment", "vader_compound", "tb_polarity"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for brand/product multi chart")
    brands = {"brand", "possible_brand", "store", "possible_store", "organization"}
    products = {"product", "possible_product"}
    bdf = df[df["llm_entity_category"].isin(brands)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "brand_like"})
    pdf = df[df["llm_entity_category"].isin(products)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "product_like", "llm_entity_sentiment": "product_sentiment"})
    merged = bdf.merge(pdf, on="conversation_id", how="inner")
    if merged.empty:
        return _save_placeholder(out, "No brand-product co-mentions")

    # Choose top pairs by overall frequency
    pair_counts = merged.groupby(["brand_like", "product_like"]).size().sort_values(ascending=False)
    top_pairs_idx = pair_counts.head(top_pairs).index.tolist()

    def pivot_for(mapper: str) -> pd.DataFrame:
        dfx = merged.copy()
        if mapper == "llm":
            dfx["label"] = dfx["llm_entity_sentiment"].apply(_normalize_sentiment_label)
        elif mapper == "vader":
            conv = df[["conversation_id", "vader_compound"]].drop_duplicates()
            dfx = dfx.merge(conv, on="conversation_id", how="left")
            dfx["label"] = dfx.apply(_vader_label, axis=1)
        else:
            conv = df[["conversation_id", "tb_polarity"]].drop_duplicates()
            dfx = dfx.merge(conv, on="conversation_id", how="left")
            dfx["label"] = dfx.apply(_textblob_label, axis=1)
        grp = dfx.groupby(["brand_like", "product_like", "label"]).size().unstack(fill_value=0)
        # Reindex to top pairs and ensure sentiment columns
        if grp.empty:
            return pd.DataFrame(index=top_pairs_idx, columns=["negative", "neutral", "positive"]).fillna(0)
        for col in ["negative", "neutral", "positive"]:
            if col not in grp.columns:
                grp[col] = 0
        grp = grp.reindex(index=top_pairs_idx).fillna(0)
        return grp[["negative", "neutral", "positive"]]

    llm_p = pivot_for("llm")
    vader_p = pivot_for("vader")
    tb_p = pivot_for("tb")

    import numpy as np
    labels = [f"{b} -> {p}" for b, p in top_pairs_idx]
    fig, axes = plt.subplots(3, 1, figsize=(22, 18), sharex=True)
    for ax, name, pvt in zip(axes, ["LLM", "VADER", "TextBlob"], [llm_p, vader_p, tb_p]):
        bottom = np.zeros(len(pvt))
        x = range(len(pvt.index))
        for sentiment, color in zip(["negative", "neutral", "positive"], ["#ef3b2c", "#9e9ac8", "#2ca25f"]):
            vals = pvt[sentiment].values
            ax.bar(x, vals, bottom=bottom, color=color, label=sentiment if ax is axes[0] else None)
            bottom += vals
        ax.set_title(name)
        ax.set_ylabel("Mentions")
    axes[-1].set_xticks(list(range(len(labels))))
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].legend(title="Sentiment")
    fig.suptitle("Brand/Store/Org -> Product co-mentions by sentiment (LLM/VADER/TextBlob)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _product_heatmap_with_mentions(df: pd.DataFrame, out: str, row_label: str, top_rows: int, top_cols: int) -> str:
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None  # type: ignore
    import numpy as np
    if df.empty:
        return _save_placeholder(out, "No data for mentions heatmap")
    # Map product sentiment to scores -1/0/1 for coloring
    score_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    df = df.copy()
    df["sent_score"] = df["product_sentiment"].map(lambda s: score_map.get(str(s).strip().lower(), 0.0))
    # Mentions: simple counts of pair rows
    counts = df.groupby([row_label, "product_like"]).size().rename("mentions").reset_index()
    avg = df.groupby([row_label, "product_like"]).agg(avg_score=("sent_score", "mean")).reset_index()
    merged = counts.merge(avg, on=[row_label, "product_like"], how="left")
    if merged.empty:
        return _save_placeholder(out, "No pair mentions")
    # Select top rows/cols by total mentions
    row_order = (
        merged.groupby(row_label)["mentions"].sum().sort_values(ascending=False).head(top_rows).index.tolist()
    )
    col_order = (
        merged.groupby("product_like")["mentions"].sum().sort_values(ascending=False).head(top_cols).index.tolist()
    )
    piv_counts = (
        merged.pivot(index=row_label, columns="product_like", values="mentions").reindex(index=row_order, columns=col_order)
    )
    piv_avg = (
        merged.pivot(index=row_label, columns="product_like", values="avg_score").reindex(index=row_order, columns=col_order)
    )
    piv_counts = piv_counts.fillna(0)
    piv_avg = piv_avg.fillna(0)
    plt.figure(figsize=(max(10, len(col_order) * 0.6), max(6, len(row_order) * 0.5)))
    if sns is not None:
        sns.heatmap(piv_avg, cmap="RdYlGn", vmin=-1, vmax=1, cbar=True, annot=piv_counts, fmt=".0f", annot_kws={"size": 8})
    else:
        plt.imshow(piv_avg.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax = plt.gca()
        for (i, j), val in np.ndenumerate(piv_counts.values):
            ax.text(j + 0.5, i + 0.5, int(val), ha='center', va='center', color='black')
        ax.set_xticks(np.arange(len(col_order)) + 0.5)
        ax.set_yticks(np.arange(len(row_order)) + 0.5)
        ax.set_xticklabels(col_order)
        ax.set_yticklabels(row_order)
    plt.title(f"Mentions heatmap with average sentiment (color) by {row_label.replace('_',' ').title()}")
    plt.xlabel("Product")
    plt.ylabel(row_label.replace('_',' ').title())
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_store_product_heatmap_mentions(unified_csv: str, out_dir: str, top_rows: int = 30, top_cols: int = 30) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "store_product_mentions_heatmap.png")
    df = pd.read_csv(unified_csv)
    req = {"conversation_id", "llm_entity", "llm_entity_category", "llm_entity_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for store/product mentions heatmap")
    stores = {"store", "possible_store"}
    products = {"product", "possible_product"}
    sdf = df[df["llm_entity_category"].isin(stores)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "store_like", "llm_entity_sentiment": "store_sentiment"})
    pdf = df[df["llm_entity_category"].isin(products)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "product_like", "llm_entity_sentiment": "product_sentiment"})
    merged = sdf.merge(pdf, on="conversation_id", how="inner")
    if merged.empty:
        return _save_placeholder(out, "No store-product co-mentions")
    return _product_heatmap_with_mentions(merged[["conversation_id", "store_like", "product_like", "product_sentiment"]], out, row_label="store_like", top_rows=top_rows, top_cols=top_cols)


def plot_brand_product_heatmap_mentions(unified_csv: str, out_dir: str, top_rows: int = 20, top_cols: int = 20) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "brand_product_mentions_heatmap.png")
    df = pd.read_csv(unified_csv)
    req = {"conversation_id", "llm_entity", "llm_entity_category", "llm_entity_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for brand/product mentions heatmap")
    brands = {"brand", "possible_brand"}
    products = {"product", "possible_product"}
    bdf = df[df["llm_entity_category"].isin(brands)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "brand_like", "llm_entity_sentiment": "brand_sentiment"})
    pdf = df[df["llm_entity_category"].isin(products)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "product_like", "llm_entity_sentiment": "product_sentiment"})
    merged = bdf.merge(pdf, on="conversation_id", how="inner")
    if merged.empty:
        return _save_placeholder(out, "No brand-product co-mentions")
    return _product_heatmap_with_mentions(merged[["conversation_id", "brand_like", "product_like", "product_sentiment"]], out, row_label="brand_like", top_rows=top_rows, top_cols=top_cols)


def plot_organization_product_heatmap_mentions(unified_csv: str, out_dir: str, top_rows: int = 20, top_cols: int = 20) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "organization_product_mentions_heatmap.png")
    df = pd.read_csv(unified_csv)
    req = {"conversation_id", "llm_entity", "llm_entity_category", "llm_entity_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for organization/product mentions heatmap")
    orgs = {"organization"}
    products = {"product", "possible_product"}
    odf = df[df["llm_entity_category"].isin(orgs)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "org_like", "llm_entity_sentiment": "org_sentiment"})
    pdf = df[df["llm_entity_category"].isin(products)][["conversation_id", "llm_entity", "llm_entity_sentiment"]].rename(columns={"llm_entity": "product_like", "llm_entity_sentiment": "product_sentiment"})
    merged = odf.merge(pdf, on="conversation_id", how="inner")
    if merged.empty:
        return _save_placeholder(out, "No organization-product co-mentions")
    return _product_heatmap_with_mentions(merged[["conversation_id", "org_like", "product_like", "product_sentiment"]], out, row_label="org_like", top_rows=top_rows, top_cols=top_cols)


# Average sentiment charts (entity-weighted)

def plot_entity_avg_sentiment_by_category_llm(unified_csv: str, out_dir: str, top_k: int = 25) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_avg_sentiment_by_category_llm.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_entity_category", "llm_entity", "llm_sentiment"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for avg sentiment (LLM)")
    dfx = df.copy()
    dfx["label"] = dfx["llm_sentiment"].apply(_normalize_sentiment_label)
    score_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    dfx["score"] = dfx["label"].map(score_map).fillna(0.0)
    ent_means = dfx.groupby(["llm_entity_category", "llm_entity"])['score'].mean().reset_index()
    cat_means = ent_means.groupby("llm_entity_category")['score'].mean().sort_values(ascending=False)
    if cat_means.empty:
        return _save_placeholder(out, "No data for avg sentiment (LLM)")
    top_cats = dfx.groupby("llm_entity_category")["llm_entity"].nunique().sort_values(ascending=False).head(top_k).index
    cat_means = cat_means.reindex(top_cats).dropna()
    plt.figure(figsize=(14, 6))
    ax = cat_means.plot(kind="bar", color="#6baed6")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    plt.ylabel("Average Sentiment (-1..1)")
    plt.title("Average Entity Sentiment by Category (entity-weighted, LLM)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entity_avg_sentiment_by_category_vader(unified_csv: str, out_dir: str, top_k: int = 25) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_avg_sentiment_by_category_vader.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_entity_category", "llm_entity", "vader_compound"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for avg sentiment (VADER)")
    dfx = df.copy()
    dfx["label"] = dfx.apply(_vader_label, axis=1)
    score_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    dfx["score"] = dfx["label"].map(score_map).fillna(0.0)
    ent_means = dfx.groupby(["llm_entity_category", "llm_entity"])['score'].mean().reset_index()
    cat_means = ent_means.groupby("llm_entity_category")['score'].mean().sort_values(ascending=False)
    if cat_means.empty:
        return _save_placeholder(out, "No data for avg sentiment (VADER)")
    top_cats = dfx.groupby("llm_entity_category")["llm_entity"].nunique().sort_values(ascending=False).head(top_k).index
    cat_means = cat_means.reindex(top_cats).dropna()
    plt.figure(figsize=(14, 6))
    ax = cat_means.plot(kind="bar", color="#9e9ac8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    plt.ylabel("Average Sentiment (-1..1)")
    plt.title("Average Entity Sentiment by Category (entity-weighted, VADER)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entity_avg_sentiment_by_category_textblob(unified_csv: str, out_dir: str, top_k: int = 25) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_avg_sentiment_by_category_textblob.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_entity_category", "llm_entity", "tb_polarity"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for avg sentiment (TextBlob)")
    dfx = df.copy()
    dfx["label"] = dfx.apply(_textblob_label, axis=1)
    score_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    dfx["score"] = dfx["label"].map(score_map).fillna(0.0)
    ent_means = dfx.groupby(["llm_entity_category", "llm_entity"])['score'].mean().reset_index()
    cat_means = ent_means.groupby("llm_entity_category")['score'].mean().sort_values(ascending=False)
    if cat_means.empty:
        return _save_placeholder(out, "No data for avg sentiment (TextBlob)")
    top_cats = dfx.groupby("llm_entity_category")["llm_entity"].nunique().sort_values(ascending=False).head(top_k).index
    cat_means = cat_means.reindex(top_cats).dropna()
    plt.figure(figsize=(14, 6))
    ax = cat_means.plot(kind="bar", color="#2ca25f")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    plt.ylabel("Average Sentiment (-1..1)")
    plt.title("Average Entity Sentiment by Category (entity-weighted, TextBlob)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entity_avg_sentiment_by_category_combined(unified_csv: str, out_dir: str, top_k: int = 25) -> str:
    _ensure_dir(out_dir)
    out = os.path.join(out_dir, "entity_avg_sentiment_by_category_combined.png")
    df = pd.read_csv(unified_csv)
    req = {"llm_entity_category", "llm_entity", "llm_sentiment", "vader_compound", "tb_polarity"}
    if not req.issubset(df.columns):
        return _save_placeholder(out, "Missing columns for avg sentiment (combined)")
    def cat_means_for(mapper: str) -> pd.Series:
        dfx = df.copy()
        if mapper == "llm":
            dfx["label"] = dfx["llm_sentiment"].apply(_normalize_sentiment_label)
        elif mapper == "vader":
            dfx["label"] = dfx.apply(_vader_label, axis=1)
        else:
            dfx["label"] = dfx.apply(_textblob_label, axis=1)
        score_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        dfx["score"] = dfx["label"].map(score_map).fillna(0.0)
        ent_means = dfx.groupby(["llm_entity_category", "llm_entity"])['score'].mean().reset_index()
        return ent_means.groupby("llm_entity_category")['score'].mean().sort_values(ascending=False)
    llm_means = cat_means_for("llm")
    vader_means = cat_means_for("vader")
    tb_means = cat_means_for("tb")
    top_cats = df.groupby("llm_entity_category")["llm_entity"].nunique().sort_values(ascending=False).head(top_k).index
    llm_means = llm_means.reindex(top_cats).dropna()
    vader_means = vader_means.reindex(top_cats).dropna()
    tb_means = tb_means.reindex(top_cats).dropna()
    import numpy as np
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
    for ax, series, title, color in zip(axes, [llm_means, vader_means, tb_means], ["LLM", "VADER", "TextBlob"], ["#6baed6", "#9e9ac8", "#2ca25f"]):
        ax.bar(series.index, series.values, color=color)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Avg Sent (-1..1)")
        ax.set_title(title)
    axes[-1].set_xticklabels(top_cats, rotation=45, ha="right")
    fig.suptitle("Average Entity Sentiment by Category (entity-weighted)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# Utility outputs

def write_entity_cluster_frequency_table(unified_csv: str, out_csv: str = "reports/entity_cluster_frequency.csv") -> str:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.read_csv(unified_csv)
    if "entity_cluster_id" not in df.columns:
        # If clusters are missing, summarize by entity instead
        freq = df["llm_entity"].value_counts().rename_axis("entity").reset_index(name="mentions")
        freq.to_csv(out_csv, index=False)
        return out_csv
    # Keep non-noise clusters and compute mentions per cluster
    sub = df[df["entity_cluster_id"].ge(0)].copy()
    if sub.empty:
        pd.DataFrame(columns=["entity_cluster_id", "entity_cluster_label", "mentions"]).to_csv(out_csv, index=False)
        return out_csv
    freq = (
        sub.groupby(["entity_cluster_id", "entity_cluster_label"]).size().rename("mentions").reset_index()
    )
    freq = freq.sort_values("mentions", ascending=False)
    freq.to_csv(out_csv, index=False)
    return out_csv


# --- Generate all ---

def generate_all(unified_csv: str, out_dir: str = "reports/images") -> Dict[str, str]:
    return {
        "intent_distribution": plot_intents_distribution(unified_csv, out_dir),
        # "sentiment_by_topic": plot_sentiment_by_topic(unified_csv, out_dir),  # removed per request
        "entity_clusters_top_labeled": plot_top_entity_clusters_labeled(unified_csv, out_dir),
        "entity_category_counts": plot_entity_category_counts(unified_csv, out_dir),
        "sentiment_overlap_llm_vs_vader": plot_sentiment_overlap_vader(unified_csv, out_dir),
        "sentiment_overlap_llm_vs_textblob": plot_sentiment_overlap_textblob(unified_csv, out_dir),
        "entity_sentiment_by_category": plot_entity_sentiment_by_category(unified_csv, out_dir),
        "entity_sentiment_by_category_llm": plot_entity_sentiment_by_category_llm(unified_csv, out_dir),
        "entity_sentiment_by_category_vader": plot_entity_sentiment_by_category_vader(unified_csv, out_dir),
        "entity_sentiment_by_category_textblob": plot_entity_sentiment_by_category_textblob(unified_csv, out_dir),
        "entity_sentiment_by_category_combined": plot_entity_sentiment_by_category_combined(unified_csv, out_dir),
        "entity_avg_sentiment_by_category_llm": plot_entity_avg_sentiment_by_category_llm(unified_csv, out_dir),
        "entity_avg_sentiment_by_category_vader": plot_entity_avg_sentiment_by_category_vader(unified_csv, out_dir),
        "entity_avg_sentiment_by_category_textblob": plot_entity_avg_sentiment_by_category_textblob(unified_csv, out_dir),
        "entity_avg_sentiment_by_category_combined": plot_entity_avg_sentiment_by_category_combined(unified_csv, out_dir),
        "entities_brand_multi": plot_entities_brand_multi(unified_csv, out_dir),
        "entities_org_multi": plot_entities_org_multi(unified_csv, out_dir),
        "entities_possible_product_multi": plot_entities_possible_product_multi(unified_csv, out_dir),
        "entities_possible_store_multi": plot_entities_possible_store_multi(unified_csv, out_dir),
        "entities_possible_brand_multi": plot_entities_possible_brand_multi(unified_csv, out_dir),
        "store_product_mentions_heatmap": plot_store_product_heatmap_mentions(unified_csv, out_dir),
        "brand_product_mentions_heatmap": plot_brand_product_heatmap_mentions(unified_csv, out_dir),
        "organization_product_mentions_heatmap": plot_organization_product_mentions_heatmap(unified_csv, out_dir),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse, json
    parser = argparse.ArgumentParser(description="Generate all analytics images from a unified CSV")
    parser.add_argument("unified_csv", help="Path to unified table CSV")
    parser.add_argument("--out_dir", default="reports/images", help="Output directory for images")
    args = parser.parse_args()
    _ensure_dir(args.out_dir)
    results = generate_all(args.unified_csv, args.out_dir)
    print(json.dumps(results, indent=2)) 