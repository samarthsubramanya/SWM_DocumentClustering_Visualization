#!/usr/bin/env python3
"""Run the clustering grid (embedding x algorithm) and write runs/doc_cluster to Postgres."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from swm import cluster, explain, ingest, represent  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", nargs="+", default=list(represent.MODELS), choices=represent.MODELS)
    p.add_argument("--algos", nargs="+", default=list(cluster.ALGOS), choices=cluster.ALGOS)
    p.add_argument("--k", type=int, default=20, help="k for k-means / agglomerative")
    p.add_argument("--min-cluster-size", type=int, default=50, help="HDBSCAN min_cluster_size")
    args = p.parse_args()

    docs = ingest.load_documents()
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    y = np.array([d["label"] for d in docs])

    for emb in args.embeddings:
        X = represent.embed(docs, emb)  # cache hit after 02_embed
        Z10, Z2 = cluster.reduce(X, emb, 10), cluster.reduce(X, emb, 2)
        for algo in args.algos:
            params = {"min_cluster_size": args.min_cluster_size} if algo == "hdbscan" else {"k": args.k}
            result = cluster.run(X, Z10, algo, y=y, **params)
            keywords = explain.ctfidf(texts, result["labels"])
            run_id = f"{emb}__{algo}"
            cluster.save_run(run_id, emb, algo, params, ids, result, Z2, keywords)
            print(run_id, json.dumps({k: round(v, 4) for k, v in result["metrics"].items()}))
