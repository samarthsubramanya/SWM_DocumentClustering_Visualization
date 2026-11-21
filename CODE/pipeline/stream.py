"""Dynamic ingestion: place a new post into an existing clustering run.

k-means and agglomerative runs cannot assign an unseen post by themselves. A new post joins the
cluster most of its 15 nearest member posts belong to (a similarity-weighted vote in embedding
space). On the 20NG test posts this agreed with batch re-clustering 0.829 of the time, against
0.767 for "nearest cluster center": Ward clusters are not round, so their centers mislead near
the borders (EVALUATIONS/dynamic/). The evidence is still measured against cluster centers, like
every stored evidence card. A post that is less similar to every center than almost every member
of the run is to its own (below the 5th percentile) is flagged as *emerging*: it may belong to a
topic the run has no cluster for.

Nothing is written to the databases; callers decide what to keep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from pipeline import db, represent
from pipeline.cluster import unit
from pipeline.explain import TOKEN_PATTERN

VOTERS = 15               # nearest member posts that vote on a new post's cluster
EMERGING_PERCENTILE = 5   # below what 95% of the run's members reach -> "emerging"
MAP_NEIGHBORS = 10        # map position = similarity-weighted mean of this many nearest members


@dataclass
class Assigner:
    run_id: str
    model: str
    ids: np.ndarray           # member post ids, aligned with X, clusters and xy
    X: np.ndarray             # unit-norm member embeddings
    clusters: np.ndarray      # each member's cluster id
    xy: np.ndarray            # members' 2-D map positions
    cluster_ids: np.ndarray   # sorted cluster ids, aligned with centers
    centers: np.ndarray       # unit-norm cluster centers
    keywords: dict[int, list] # c-TF-IDF terms per cluster
    threshold: float          # emerging below this similarity to the nearest center

    @classmethod
    def from_run(cls, run_id: str) -> "Assigner":
        model = db.query("SELECT embedding FROM runs WHERE run_id = %s", (run_id,))[0]["embedding"]
        rows = db.query(
            "SELECT doc_id, cluster_id, is_noise, x, y FROM doc_cluster WHERE run_id = %s ORDER BY doc_id",
            (run_id,),
        )
        cached_ids, cached_X = represent.load(model)
        pos = {str(d): i for i, d in enumerate(cached_ids)}
        rows = [r for r in rows if not r["is_noise"] and r["doc_id"] in pos]
        X = unit(cached_X[[pos[r["doc_id"]] for r in rows]].astype(np.float32))
        clusters = np.array([r["cluster_id"] for r in rows])
        cluster_ids = np.unique(clusters)
        centers = unit(np.stack([X[clusters == c].mean(axis=0) for c in cluster_ids]))
        own = (X * centers[np.searchsorted(cluster_ids, clusters)]).sum(axis=1)
        keywords = {r["cluster_id"]: r["keywords"] for r in db.query(
            "SELECT cluster_id, keywords FROM cluster_info WHERE run_id = %s", (run_id,))}
        return cls(
            run_id=run_id, model=model, ids=np.array([r["doc_id"] for r in rows]), X=X, clusters=clusters,
            xy=np.array([[r["x"], r["y"]] for r in rows], dtype=np.float32), cluster_ids=cluster_ids,
            centers=centers, keywords=keywords, threshold=float(np.percentile(own, EMERGING_PERCENTILE)),
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed new posts the way the run's member vectors were embedded, so they are comparable."""
        meta = represent.CACHE / f"{self.model}.meta.json"
        chunked = meta.exists() and json.loads(meta.read_text()).get("chunk", False)
        return (represent.encode_docs if chunked else represent.encode_texts)(texts, self.model)

    def assign(self, texts: list[str], vectors: np.ndarray | None = None) -> list[dict]:
        """Evidence for each new post, in the shape of the stored evidence cards, plus
        `emerging` and a map position. Pass `vectors` to skip encoding."""
        V = unit(np.asarray(self.encode(texts) if vectors is None else vectors, dtype=np.float32))
        sims = V @ self.centers.T
        vocab = sorted({t for kw in self.keywords.values() for t, _ in kw})
        counts = CountVectorizer(vocabulary=vocab, token_pattern=TOKEN_PATTERN).transform(texts).toarray()

        out = []
        for i in range(len(texts)):
            everyone = self.X @ V[i]
            voters = np.argpartition(-everyone, VOTERS)[:VOTERS]
            votes = np.zeros(len(self.cluster_ids))
            np.add.at(votes, np.searchsorted(self.cluster_ids, self.clusters[voters]), everyone[voters])
            best = int(votes.argmax())
            others = np.delete(np.arange(len(self.cluster_ids)), best)
            second = int(others[np.argmax(sims[i, others])]) if len(others) else None
            cid = int(self.cluster_ids[best])
            own_sim = float(sims[i, best])
            weights = {t: w for t, w in self.keywords.get(cid, [])}
            hits = sorted(
                ({"term": vocab[j], "count": int(counts[i, j]), "cluster_weight": weights[vocab[j]]}
                 for j in np.flatnonzero(counts[i]) if vocab[j] in weights),
                key=lambda h: -h["cluster_weight"],
            )

            members = np.flatnonzero(self.clusters == cid)
            member_sims = self.X[members] @ V[i]
            nearest = members[np.argsort(-member_sims)[:3]]

            near = np.argsort(-everyone)[:MAP_NEIGHBORS]
            w = np.clip(everyone[near], 1e-6, None)
            x, y = (self.xy[near] * w[:, None]).sum(axis=0) / w.sum()

            out.append({
                "cluster_id": cid,
                "is_noise": False,
                "membership_prob": None,
                "own_centroid_sim": round(own_sim, 4),
                "runner_up_cluster": None if second is None else int(self.cluster_ids[second]),
                "runner_up_sim": None if second is None else round(float(sims[i, second]), 4),
                "margin": None if second is None else round(own_sim - float(sims[i, second]), 4),
                "term_hits": hits,
                "cluster_keywords": [t for t, _ in self.keywords.get(cid, [])],
                "exemplars": [{"doc_id": str(self.ids[j]), "similarity": round(float(self.X[j] @ V[i]), 4)}
                              for j in nearest],
                "emerging": float(sims[i].max()) < self.threshold,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
            })
        return out


@lru_cache(maxsize=4)
def assigner(run_id: str) -> Assigner:
    """One Assigner per run, built on first use (loads the run's vectors and centers)."""
    return Assigner.from_run(run_id)
