#!/usr/bin/env python3
"""Run the clustering grid (embedding x algorithm) and write runs/doc_cluster to Postgres.

--seeds 42 43 44 repeats every run with other UMAP / k-means seeds. Seed 42 (the first) is
stored in Postgres as before; the others only add metrics to eval/reports/clustering-seeds-*.json
(mean and std per run), so the explained run never changes underneath 04/05.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
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
    p.add_argument("--seeds", type=int, nargs="+", default=[cluster.SEED],
                   help="first seed is stored; the rest only add mean/std metrics")
    args = p.parse_args()
    if args.seeds[0] != cluster.SEED:
        p.error(f"the first seed must be {cluster.SEED} (the stored run)")

    docs = ingest.load_documents()
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    y = np.array([d["label"] for d in docs])

    per_seed = defaultdict(list)
    for emb in args.embeddings:
        X = represent.embed(docs, emb)  # cache hit after 02_embed
        for seed in args.seeds:
            Z10 = cluster.reduce(X, emb, 10, seed=seed)
            Z2 = cluster.reduce(X, emb, 2) if seed == cluster.SEED else None
            for algo in args.algos:
                params = {"min_cluster_size": args.min_cluster_size} if algo == "hdbscan" else {"k": args.k}
                result = cluster.run(X, Z10, algo, y=y, seed=seed, **params)
                run_id = f"{emb}__{algo}"
                per_seed[run_id].append(result["metrics"])
                if seed == cluster.SEED:
                    keywords = explain.ctfidf(texts, result["labels"])
                    cluster.save_run(run_id, emb, algo, params, ids, result, Z2, keywords)
                print(f"{run_id} seed={seed}",
                      json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result["metrics"].items()}))

    if len(args.seeds) > 1:
        summary = {}
        for run_id, ms in per_seed.items():
            keys = sorted({k for m in ms for k, v in m.items() if isinstance(v, (int, float)) and v is not None})
            summary[run_id] = {k: {"mean": float(np.mean([m[k] for m in ms if m.get(k) is not None])),
                                   "std": float(np.std([m[k] for m in ms if m.get(k) is not None]))}
                               for k in keys}
        out = ROOT / "eval" / "reports" / f"clustering-seeds-{time.strftime('%Y%m%d-%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"seeds": args.seeds, "runs": summary}, indent=2))
        print("\nrun                                   NMI (mean ± std)    ARI (mean ± std)    DBCV")
        for run_id, m in summary.items():
            f = lambda k: f"{m[k]['mean']:.3f} ± {m[k]['std']:.3f}" if k in m else "n/a"
            print(f"{run_id:38s} {f('nmi'):19s} {f('ari'):19s} {f('dbcv')}")
        print(f"Seed report: {out}")
