"""Document embeddings, cached to data/cache/*.npy so no stage recomputes upstream."""


def embed(docs: list[dict], model: str) -> "np.ndarray":
    """model in {'tfidf_lsa', 'all-mpnet-base-v2', 'bge-base-en-v1.5'}."""
    raise NotImplementedError
