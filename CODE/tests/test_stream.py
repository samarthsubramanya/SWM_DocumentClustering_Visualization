"""Dynamic ingestion: placing new posts in a run (synthetic vectors; no database or encoder)."""

import numpy as np

from pipeline import stream
from pipeline.cluster import unit
from src.agents.ingest_agent import article_text


def toy_assigner() -> stream.Assigner:
    rng = np.random.default_rng(0)
    axes = np.eye(4, dtype=np.float32)
    X = unit(np.vstack([axes[c] + 0.05 * rng.standard_normal((20, 4)) for c in (0, 1)]).astype(np.float32))
    clusters = np.repeat([0, 1], 20)
    centers = unit(np.stack([X[clusters == c].mean(axis=0) for c in (0, 1)]))
    return stream.Assigner(
        run_id="r", model="m", ids=np.array([f"train/g/{i}" for i in range(40)]), X=X, clusters=clusters,
        xy=np.array([[float(c), 0.0] for c in clusters], dtype=np.float32), cluster_ids=np.array([0, 1]),
        centers=centers, keywords={0: [["rocket", 0.3]], 1: [["engine", 0.3]]}, threshold=0.5,
    )


def test_new_post_joins_the_cluster_its_neighbors_vote_for():
    a = toy_assigner()
    [ev] = a.assign(["the rocket launched; rocket news"], vectors=np.array([[1.0, 0.1, 0, 0]]))
    assert ev["cluster_id"] == 0 and ev["runner_up_cluster"] == 1 and ev["margin"] > 0
    assert ev["term_hits"] == [{"term": "rocket", "count": 2, "cluster_weight": 0.3}]
    assert len(ev["exemplars"]) == 3 and all(e["doc_id"].startswith("train/") for e in ev["exemplars"])
    assert ev["x"] < 0.5 and not ev["emerging"]   # placed among cluster 0's posts on the map


def test_post_unlike_every_cluster_is_emerging():
    [ev] = toy_assigner().assign(["unrelated words here"], vectors=np.array([[0, 0, 1.0, 0]]))
    assert ev["emerging"]


def test_article_text_drops_the_request():
    assert article_text("add this article: NASA launched a probe.") == "NASA launched a probe."
    assert article_text("Where does this post belong? My car engine stalls") == "My car engine stalls"
    assert article_text("Plain pasted text") == "Plain pasted text"
