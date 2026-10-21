"""Clustering runs + metrics. UMAP(10-d) -> HDBSCAN, with k-means/agglomerative baselines."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix

from swm import db
from swm.represent import CACHE

ALGOS = ("hdbscan", "kmeans", "agglomerative")
SEED = 42


def unit(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)


def reduce(X: np.ndarray, model: str, n_components: int) -> np.ndarray:
    """UMAP, cached per embedding model: 10-d for clustering, 2-d for the scatter."""
    path, emb_path = CACHE / f"{model}.umap{n_components}.npy", CACHE / f"{model}.npy"
    fresh = path.exists() and (not emb_path.exists() or path.stat().st_mtime >= emb_path.stat().st_mtime)
    if fresh:
        Z = np.load(path)
        if len(Z) == len(X):
            return Z

    import umap

    Z = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.0 if n_components > 2 else 0.1,
        metric="cosine",
        random_state=SEED,
    ).fit_transform(X).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, Z)
    return Z


def centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cluster ids, unit-norm centroids) of non-noise clusters in the original embedding space."""
    ids = np.unique(labels[labels >= 0])
    return ids, unit(np.stack([X[labels == c].mean(axis=0) for c in ids]))


def assign_noise(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """HDBSCAN noise (-1) -> nearest centroid, so every doc has a cluster in the UI."""
    assigned = labels.copy()
    noise = labels < 0
    if noise.any() and (~noise).any():
        ids, C = centroids(X, labels)
        assigned[noise] = ids[np.argmax(unit(X[noise]) @ C.T, axis=1)]
    return assigned


def purity(y: np.ndarray, labels: np.ndarray) -> float:
    cm = contingency_matrix(y, labels)
    return float(cm.max(axis=0).sum() / cm.sum())


def evaluate(labels: np.ndarray, Z: np.ndarray, y: np.ndarray | None = None) -> dict:
    """NMI/ARI/purity on raw labels (noise counts as its own cluster); silhouette on non-noise."""
    core = labels >= 0
    m = {"n_clusters": int(len(np.unique(labels[core]))), "noise_rate": float(1 - core.mean())}
    if y is not None:
        m |= {
            "nmi": float(normalized_mutual_info_score(y, labels)),
            "ari": float(adjusted_rand_score(y, labels)),
            "purity": purity(y, labels),
        }
    if m["n_clusters"] > 1:
        m["silhouette"] = float(silhouette_score(
            Z[core], labels[core], sample_size=min(5000, int(core.sum())), random_state=SEED
        ))
    return m


def run(X: np.ndarray, Z: np.ndarray, algo: str, y: np.ndarray | None = None, **params) -> dict:
    """Cluster the reduced embedding Z. X (original space) is used for noise reassignment.

    Returns {labels, assigned, probs, outlier_scores, metrics}.
    """
    probs = outlier_scores = None
    if algo == "hdbscan":
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=params.get("min_cluster_size", 50),
            min_samples=params.get("min_samples"),
        ).fit(Z)
        labels, probs, outlier_scores = model.labels_, model.probabilities_, model.outlier_scores_
    elif algo == "kmeans":
        labels = KMeans(n_clusters=params.get("k", 20), n_init=10, random_state=SEED).fit_predict(Z)
    elif algo == "agglomerative":
        # ponytail: ward needs O(n^2) memory (~1.3 GB at 17k docs); subsample + kNN-assign if that bites.
        labels = AgglomerativeClustering(n_clusters=params.get("k", 20), linkage="ward").fit_predict(Z)
    else:
        raise ValueError(f"Unknown algo {algo!r}; choose from {ALGOS}")

    return {
        "labels": labels,
        "assigned": assign_noise(X, labels),
        "probs": probs,
        "outlier_scores": outlier_scores,
        "metrics": evaluate(labels, Z, y),
    }


def save_run(run_id: str, embedding: str, algo: str, params: dict, ids, result: dict,
             xy: np.ndarray, keywords: dict[int, list]) -> None:
    """Replace one run: runs row, doc_cluster rows (with 2-D coords), cluster_info (c-TF-IDF keywords)."""
    probs, outlier = result["probs"], result["outlier_scores"]
    sizes = Counter(result["assigned"].tolist())
    doc_rows = [
        (run_id, str(doc_id), int(a), bool(l < 0),
         None if probs is None else float(probs[i]),
         None if outlier is None else float(outlier[i]),
         float(xy[i, 0]), float(xy[i, 1]))
        for i, (doc_id, l, a) in enumerate(zip(ids, result["labels"], result["assigned"]))
    ]
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))  # cascades to dependent tables
        cur.execute(
            "INSERT INTO runs (run_id, embedding, algo, params, metrics) VALUES (%s, %s, %s, %s, %s)",
            (run_id, embedding, algo, db.Jsonb(params), db.Jsonb(result["metrics"])),
        )
        cur.executemany(
            "INSERT INTO doc_cluster (run_id, doc_id, cluster_id, is_noise, prob, outlier_score, x, y) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            doc_rows,
        )
        cur.executemany(
            "INSERT INTO cluster_info (run_id, cluster_id, size, keywords) VALUES (%s, %s, %s, %s)",
            [(run_id, c, sizes[c], db.Jsonb(kw)) for c, kw in keywords.items()],
        )
