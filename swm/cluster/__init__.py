"""Clustering runs + metrics. UMAP(10-d) -> HDBSCAN, with k-means/agglomerative baselines."""


def run(embeddings: "np.ndarray", algo: str, **params) -> dict:
    """Returns {'labels': ..., 'probs': ..., 'metrics': {nmi, ari, purity, silhouette}}."""
    raise NotImplementedError
