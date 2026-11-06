#!/usr/bin/env python3
"""Choose k for k-means / agglomerative WITHOUT the newsgroup labels.

k = 20 equals the number of newsgroups, which gives the k-means / agglomerative baselines
an advantage HDBSCAN does not get. This sweeps k on the cached UMAP-10 space and reports
two label-free criteria: silhouette (max) and the k-means inertia elbow (proposal VI-D).
NMI is printed for information only; it must not be used to pick k.

  python scripts/03b_k_select.py --embedding bge-base-en-v1.5 --kmin 10 --kmax 40 --step 2
  python scripts/03_cluster.py --k <chosen k> ...    # re-run with the label-free k if it differs
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score, silhouette_score  # noqa: E402

from swm import cluster, ingest, represent  # noqa: E402


def elbow(ks: list[int], inertia: list[float]) -> int:
    """k at the point farthest from the line joining the first and last (k, inertia) points."""
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inertia) - inertia[-1]) / (inertia[0] - inertia[-1])
    return int(ks[int(np.argmax(np.abs(1 - x - y)))])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding", default="bge-base-en-v1.5", choices=represent.MODELS)
    p.add_argument("--kmin", type=int, default=10)
    p.add_argument("--kmax", type=int, default=40)
    p.add_argument("--step", type=int, default=2)
    args = p.parse_args()

    docs = ingest.load_documents()
    y = np.array([d["label"] for d in docs])
    X = represent.embed(docs, args.embedding)
    Z = cluster.reduce(X, args.embedding, 10)
    ks = list(range(args.kmin, args.kmax + 1, args.step))
    cache = ROOT / "data" / "cache" / "ward_tree"  # the Ward tree is built once and cut at every k

    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=cluster.SEED).fit(Z)
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward", memory=str(cache),
                                     compute_full_tree=True).fit_predict(Z)
        row = {"k": k, "kmeans_inertia": float(km.inertia_)}
        for name, lab in (("kmeans", km.labels_), ("agglomerative", ag)):
            row[f"{name}_silhouette"] = float(silhouette_score(Z, lab, sample_size=5000, random_state=cluster.SEED))
            row[f"{name}_nmi_info_only"] = float(normalized_mutual_info_score(y, lab))
        rows.append(row)
        print(f"k={k:3d}  sil kmeans {row['kmeans_silhouette']:.3f}  sil agg {row['agglomerative_silhouette']:.3f}"
              f"  (NMI, info only: {row['kmeans_nmi_info_only']:.3f} / {row['agglomerative_nmi_info_only']:.3f})")

    choice = {
        "kmeans_by_silhouette": max(rows, key=lambda r: r["kmeans_silhouette"])["k"],
        "agglomerative_by_silhouette": max(rows, key=lambda r: r["agglomerative_silhouette"])["k"],
        "kmeans_by_elbow": elbow(ks, [r["kmeans_inertia"] for r in rows]),
    }
    out = ROOT / "eval" / "reports" / f"k-select-{args.embedding}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"embedding": args.embedding, "choice": choice, "sweep": rows}, indent=2))
    print(f"\nLabel-free choice: {choice}\nReport: {out}")
