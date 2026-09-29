"""Per-document evidence and LLM rationale. Evidence-first: the LLM verbalizes
structured evidence it cannot invent, so every rationale is checkable."""


def evidence(doc_id: str, run_id: str) -> dict:
    """c-TF-IDF term hits, centroid margin, nearest exemplars, membership prob."""
    raise NotImplementedError


def rationale(doc_id: str, run_id: str) -> str:
    """LLM rationale that must quote >=2 evidence terms; regenerate on failure."""
    raise NotImplementedError
