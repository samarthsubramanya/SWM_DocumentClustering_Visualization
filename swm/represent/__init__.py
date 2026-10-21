"""Document embeddings, cached to data/cache/*.npy so no stage recomputes upstream."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"

SBERT_MODELS = {
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}
MODELS = ("tfidf_lsa", *SBERT_MODELS)


@lru_cache(maxsize=2)
def _sbert(model: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SBERT_MODELS[model])


def encode_texts(texts, model: str = "all-mpnet-base-v2", progress: bool = False) -> np.ndarray:
    """Unit-norm sentence embeddings (also used for KG entity alias merging)."""
    return _sbert(model).encode(
        list(texts), batch_size=64, show_progress_bar=progress, normalize_embeddings=True
    ).astype(np.float32)


def _tfidf_lsa(texts: list[str], dims: int = 300) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    pipe = make_pipeline(
        TfidfVectorizer(stop_words="english", max_df=0.5, min_df=5, sublinear_tf=True),
        TruncatedSVD(dims, random_state=42),
        Normalizer(copy=False),
    )
    return pipe.fit_transform(texts).astype(np.float32)


def embed(docs: list[dict], model: str, force: bool = False) -> np.ndarray:
    """model in MODELS. Returns the cached matrix when it was built for the same doc ids."""
    if model not in MODELS:
        raise ValueError(f"Unknown embedding model {model!r}; choose from {MODELS}")
    ids = np.array([d["id"] for d in docs])
    path, ids_path = CACHE / f"{model}.npy", CACHE / f"{model}.ids.npy"
    if not force and path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path)
        if np.array_equal(cached_ids, ids):
            return np.load(path)
        # SBERT vectors are per-document, so a subset of the cached docs can be reused
        # (TF-IDF+LSA is fit on the corpus and must be refit).
        pos = {doc_id: i for i, doc_id in enumerate(cached_ids)}
        if model in SBERT_MODELS and all(doc_id in pos for doc_id in ids):
            X = np.load(path)[[pos[doc_id] for doc_id in ids]]
            np.save(path, X)
            np.save(ids_path, ids)
            return X

    texts = [d["text"] for d in docs]
    X = _tfidf_lsa(texts) if model == "tfidf_lsa" else encode_texts(texts, model, progress=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    np.save(ids_path, ids)
    return X


def load(model: str) -> tuple[np.ndarray, np.ndarray]:
    """(doc ids, embeddings) straight from the cache."""
    path = CACHE / f"{model}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/02_embed.py --models {model}")
    return np.load(CACHE / f"{model}.ids.npy"), np.load(path)
