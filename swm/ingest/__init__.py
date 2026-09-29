"""Corpus loaders. Each loader returns a list[dict] with at least {id, text}."""


def load_20ng(subset: str = "all", min_tokens: int = 20) -> list[dict]:
    """Load 20news-bydate, strip headers/footers/quotes, drop short posts.

    ponytail: scikit-learn's fetch_20newsgroups does the header/footer/quote
    stripping for us (remove=('headers','footers','quotes')) — no custom
    parser needed.
    """
    raise NotImplementedError
