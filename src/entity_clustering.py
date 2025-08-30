from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


def _normalize_entity(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    # Lowercase, remove non-alphanumeric except spaces, collapse whitespace
    s = text.lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = " ".join(s.split())
    return s


def _embed_texts(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # pragma: no cover
        SentenceTransformer = None  # type: ignore
    if SentenceTransformer is not None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)
    # Fallback: TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(min_df=1, max_features=4096)
    X = vec.fit_transform(texts)
    return normalize(X).toarray().astype(np.float32)


def _cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    try:
        import hdbscan
    except Exception:
        hdbscan = None  # type: ignore
    labels: np.ndarray | None = None
    if hdbscan is not None:
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
            labels = clusterer.fit_predict(embeddings).astype(int)
        except Exception:
            labels = None
    # Fallbacks if HDBSCAN unavailable or produced a degenerate result
    if labels is None or (np.unique(labels[labels >= 0]).size <= 1):
        try:
            from sklearn.cluster import KMeans
            # Heuristic cluster count: sqrt(n/2) bounded [2, 30]
            n = embeddings.shape[0]
            k = int(max(2, min(30, (n / 2) ** 0.5)))
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(embeddings).astype(int)
        except Exception:
            # Last resort: sequential small buckets to create variety
            n = embeddings.shape[0]
            bucket = max(2, min(50, min_cluster_size))
            labels = np.arange(n, dtype=int) // bucket
    return labels.astype(int)


def cluster_entities(
    unified_csv_path: str,
    output_csv_path: Optional[str] = None,
    min_cluster_size: int = 3,
) -> str:
    df = pd.read_csv(unified_csv_path)
    # Prepare normalized entities used for clustering
    df["entity_norm"] = df.get("llm_entity", pd.Series([None] * len(df))).apply(_normalize_entity)
    # Combine category + normalized entity for better separation
    mask = df["entity_norm"].astype(str).str.strip().ne("") & df["entity_norm"].notna()
    texts = (
        df.loc[mask, ["llm_entity_category", "entity_norm"]]
        .fillna("")
        .astype(str)
        .apply(lambda r: f"{r['llm_entity_category']} :: {r['entity_norm']}", axis=1)
        .tolist()
    )
    if not texts:
        df["entity_cluster_id"] = -1
        df["entity_cluster_label"] = None
    else:
        emb = _embed_texts(texts)
        labels = _cluster_embeddings(emb, min_cluster_size=min_cluster_size)
        # Map back to original index positions
        entity_cluster_id = pd.Series(index=df.index, data=-1, dtype=int)
        entity_cluster_id.loc[mask] = labels
        # Re-label clusters (excluding -1) to consecutive integers for readability
        non_noise = entity_cluster_id[entity_cluster_id.ge(0)]
        if not non_noise.empty:
            unique_labels = sorted(non_noise.unique().tolist())
            mapping = {old: new for new, old in enumerate(unique_labels)}
            entity_cluster_id = entity_cluster_id.apply(lambda x: mapping.get(x, -1))
        df["entity_cluster_id"] = entity_cluster_id
        # Build human-readable labels: top-3 normalized entities per cluster
        label_map: dict[int, str] = {}
        grp = df[df["entity_cluster_id"].ge(0)].groupby("entity_cluster_id")
        for cid, sub in grp:
            top_entities = (
                sub["entity_norm"].value_counts()
                .head(3)
                .index.tolist()
            )
            label_map[int(cid)] = ", ".join([t for t in top_entities if t]) if top_entities else ""
        df["entity_cluster_label"] = df["entity_cluster_id"].apply(lambda cid: label_map.get(int(cid), None) if cid >= 0 else None)

    out_path = output_csv_path or unified_csv_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Cluster LLM entities and assign entity_cluster_id")
    parser.add_argument("unified_csv", help="Path to unified table CSV")
    parser.add_argument("--out", default=None, help="Optional output CSV (defaults to overwrite input)")
    parser.add_argument("--min_cluster_size", type=int, default=3)
    args = parser.parse_args()

    print(cluster_entities(args.unified_csv, output_csv_path=args.out, min_cluster_size=args.min_cluster_size)) 