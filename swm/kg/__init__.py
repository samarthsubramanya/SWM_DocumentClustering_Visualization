"""Ontology-anchored cluster knowledge graphs: extract -> normalize -> load -> label."""


def extract(cluster_id: str, ontology_path: str = "ontology/news.yaml") -> list[dict]:
    """LLM triple extraction over representative docs, constrained to the ontology."""
    raise NotImplementedError


def normalize(triples: list[dict]) -> list[dict]:
    """Lowercase + alias merge via embedding similarity > 0.9."""
    raise NotImplementedError


def load_neo4j(triples: list[dict]) -> None:
    """MERGE triples into Neo4j using the ontology-compiled schema."""
    raise NotImplementedError


def pagerank_label(cluster_id: str) -> str:
    """Top-3 PageRank entities + top c-TF-IDF term."""
    raise NotImplementedError
