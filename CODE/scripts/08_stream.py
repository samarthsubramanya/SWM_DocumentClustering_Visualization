#!/usr/bin/env python3
"""Dynamic ingestion evaluation: stream the 20 Newsgroups test posts into a clustering run.

The run was clustered on the train posts; every test post is new to it. Each one is placed by
a vote of its 15 nearest train posts (pipeline/stream.py), and the result is scored against the
targets in the proposal:

  agreement   streamed cluster == batch cluster, where the batch clustering re-clusters train
              and test together (Ward, same k, on the cached 10-d UMAP) and its clusters are
              matched to the run's on the train posts                    target >= 0.85
  latency     one post at a time, encoding included                       target p95 < 3 s
  quality     NMI / ARI / purity of the streamed posts against their newsgroups, next to the
              run's own NMI on the train posts
  emerging    share of posts flagged as possibly belonging to no existing cluster

Report: EVALUATIONS/dynamic/stream-<run>-<time>.{json,md}. Nothing is written to Postgres or Neo4j
unless you pass --save, which stores the result as a separate run, <run>__streamed: the run's own
posts plus the streamed ones, with evidence for every post. The original run is left untouched, and
the streamed posts are stored with source '20ng-stream', so the pipeline stages, which load source
'20ng', never mix them into a clustering.

  python scripts/08_stream.py --run all-mpnet-base-v2__agglomerative [--save]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

from pipeline import cluster, db, ingest, represent, stream  # noqa: E402

REPORTS = ROOT.parent / "EVALUATIONS" / "dynamic"
ARCHIVE = ROOT.parent / "DATA" / "20newsgroups" / "20news-bydate.tar.gz"


def batch_agreement(a: stream.Assigner, test_ids: list[str], streamed: np.ndarray, k: int) -> dict:
    """Re-cluster train + test in one batch and compare it with streaming on the test posts."""
    ids, _ = represent.load(a.model)
    Z = np.load(represent.CACHE / f"{a.model}.umap10.npy")   # UMAP of every cached post (train + test)
    batch = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    pos = {str(d): i for i, d in enumerate(ids)}

    # Match batch clusters to the run's clusters on the train posts both of them cover.
    run_of = dict(zip(a.ids, a.clusters))
    train = [d for d in run_of if d in pos]
    run_ids = sorted(set(a.clusters))
    table = np.zeros((k, len(run_ids)))
    for d in train:
        table[batch[pos[d]], run_ids.index(run_of[d])] += 1
    rows, cols = linear_sum_assignment(-table)
    to_run = {int(r): run_ids[c] for r, c in zip(rows, cols)}

    mapped_train = np.array([to_run.get(int(batch[pos[d]]), -1) for d in train])
    mapped_test = np.array([to_run.get(int(batch[pos[d]]), -1) for d in test_ids])
    return {
        "agreement": float(np.mean(mapped_test == streamed)),
        # how well the batch re-clustering reproduces the run itself: a ceiling for agreement
        "batch_vs_run_on_train": float(np.mean(mapped_train == np.array([run_of[d] for d in train]))),
    }


def latency(a: stream.Assigner, posts: list[dict], n: int, seed: int = 42) -> dict:
    """Seconds per post when posts arrive one at a time (encoding + placement)."""
    a.assign([posts[0]["text"]])   # warm-up: loads the encoder once
    sample = random.Random(seed).sample(posts, min(n, len(posts)))
    times = []
    for p in sample:
        t = time.perf_counter()
        a.assign([p["text"]])
        times.append(time.perf_counter() - t)
    return {"posts": len(times), "p50_s": float(np.percentile(times, 50)),
            "p95_s": float(np.percentile(times, 95)), "max_s": float(max(times))}


def save_streamed_run(a: stream.Assigner, posts: list[dict], placed: list[dict], metrics: dict) -> str:
    """Store <run>__streamed: the base run's posts and evidence, copied, plus the streamed posts."""
    run_id = f"{a.run_id}__streamed"
    base = db.query("SELECT algo, params FROM runs WHERE run_id = %s", (a.run_id,))[0]
    params = {**base["params"], "base_run": a.run_id, "placement": f"{stream.VOTERS}-nearest-neighbor vote",
              "streamed_posts": len(posts)}
    docs = [(d["id"], d["text"], d["label"], d["label_name"], "test", "20ng-stream") for d in posts]
    rows = [(run_id, d["id"], e["cluster_id"], e["x"], e["y"]) for d, e in zip(posts, placed)]
    evidence = [(run_id, d["id"], db.Jsonb(e)) for d, e in zip(posts, placed)]

    with db.connect() as conn, conn.cursor() as cur:   # one transaction: all of it or none of it
        cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))   # rerunning replaces it (cascades)
        for start in range(0, len(docs), 500):                         # remote databases are slow
            cur.executemany(
                """INSERT INTO documents (id, text, label, label_name, split, source) VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO NOTHING""", docs[start:start + 500])
        cur.execute("INSERT INTO runs (run_id, embedding, algo, params, metrics) VALUES (%s, %s, %s, %s, %s)",
                    (run_id, a.model, base["algo"], db.Jsonb(params), db.Jsonb(metrics)))
        cur.execute("""INSERT INTO doc_cluster (run_id, doc_id, cluster_id, is_noise, prob, outlier_score, x, y)
                       SELECT %s, doc_id, cluster_id, is_noise, prob, outlier_score, x, y
                       FROM doc_cluster WHERE run_id = %s""", (run_id, a.run_id))
        for start in range(0, len(rows), 500):
            cur.executemany("INSERT INTO doc_cluster (run_id, doc_id, cluster_id, x, y) VALUES (%s, %s, %s, %s, %s)",
                            rows[start:start + 500])
        cur.execute("""INSERT INTO cluster_info (run_id, cluster_id, size, keywords, label)
                       SELECT %s, ci.cluster_id, (SELECT count(*) FROM doc_cluster dc
                                                  WHERE dc.run_id = %s AND dc.cluster_id = ci.cluster_id),
                              ci.keywords, ci.label
                       FROM cluster_info ci WHERE ci.run_id = %s""", (run_id, run_id, a.run_id))
        cur.execute("""INSERT INTO explanations (run_id, doc_id, evidence)
                       SELECT %s, doc_id, evidence FROM explanations WHERE run_id = %s""", (run_id, a.run_id))
        for start in range(0, len(evidence), 500):
            cur.executemany("INSERT INTO explanations (run_id, doc_id, evidence) VALUES (%s, %s, %s)",
                            evidence[start:start + 500])
    return run_id


def markdown(r: dict) -> str:
    q, ag, lat = r["quality"], r["batch"], r["latency"]
    lines = [
        f"# Dynamic ingestion: {r['run_id']}",
        "",
        f"{r['posts']} test posts streamed into a run clustered on {r['train_posts']} train posts "
        f"(15-nearest-neighbor vote, `{r['model']}` embeddings).",
        "",
        "| Metric | Result | Target |",
        "|---|---|---|",
        f"| Agreement with batch clustering | {ag['agreement']:.3f} | ≥ 0.85 |",
        f"| Latency per post, p95 (p50) | {lat['p95_s']:.3f} s ({lat['p50_s']:.3f} s) | < 3 s |",
        f"| NMI of streamed posts vs newsgroups | {q['nmi']:.3f} (run on train: {r['run_nmi']:.3f}) | ≥ 0.55 |",
        f"| ARI / purity of streamed posts | {q['ari']:.3f} / {q['purity']:.3f} | |",
        f"| Flagged as emerging | {r['emerging_share']:.1%} (threshold: similarity < {r['threshold']:.3f}) | |",
        "",
        f"The batch reference re-clusters train and test together with Ward (k = {r['k']}) on the cached "
        f"10-d UMAP. On the train posts it agrees with the stored run {ag['batch_vs_run_on_train']:.3f} of "
        "the time, which bounds how high streaming agreement can go.",
        "",
        f"Latency was measured one post at a time on {lat['posts']} random test posts (max {lat['max_s']:.3f} s). "
        f"The scores use the cached test-post vectors; fresh encodings of {r['encode_check']['posts']} posts "
        f"matched them with cosine ≥ {r['encode_check']['min_cosine']:.4f}.",
        "",
        "## Lowest-similarity posts (emerging candidates)",
        "",
        "| Post | Nearest cluster | Similarity | Newsgroup | Opening |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| `{e['post']}` | {e['cluster_id']} | {e['similarity']:.3f} | {e['newsgroup']} | {e['opening']} |"
              for e in r["emerging_examples"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the run to stream into, e.g. all-mpnet-base-v2__agglomerative")
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--latency-posts", type=int, default=200)
    p.add_argument("--save", action="store_true", help="store the result as the separate run <run>__streamed")
    args = p.parse_args()

    a = stream.Assigner.from_run(args.run)
    known = set(a.ids)   # the run's own posts; anything else is new to it
    posts = [d for d in ingest.load_20ng("test", archive=args.archive) if d["id"] not in known]
    print(f"{args.run}: {len(a.ids)} train posts, {len(posts)} new test posts")

    cached_ids, cached_X = represent.load(a.model)
    pos = {str(d): i for i, d in enumerate(cached_ids)}
    posts = [d for d in posts if d["id"] in pos]
    vectors = cached_X[[pos[d["id"]] for d in posts]]

    check = random.Random(0).sample(range(len(posts)), min(100, len(posts)))
    fresh = a.encode([posts[i]["text"] for i in check])
    cosines = (cluster.unit(fresh) * cluster.unit(vectors[check])).sum(axis=1)

    t = time.time()
    placed = a.assign([d["text"] for d in posts], vectors=vectors)
    print(f"placed {len(placed)} posts in {time.time() - t:.1f}s")
    streamed = np.array([e["cluster_id"] for e in placed])
    gold = np.array([d["label"] for d in posts])

    k = int(db.query("SELECT params->>'k' AS k FROM runs WHERE run_id = %s", (args.run,))[0]["k"] or len(a.cluster_ids))
    lowest = np.argsort([e["own_centroid_sim"] for e in placed])[:8]
    report = {
        "run_id": args.run,
        "model": a.model,
        "k": k,
        "train_posts": len(a.ids),
        "posts": len(posts),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_nmi": float(db.query("SELECT metrics->>'nmi' AS nmi FROM runs WHERE run_id = %s", (args.run,))[0]["nmi"]),
        "quality": {
            "nmi": float(normalized_mutual_info_score(gold, streamed)),
            "ari": float(adjusted_rand_score(gold, streamed)),
            "purity": cluster.purity(gold, streamed),
        },
        "batch": batch_agreement(a, [d["id"] for d in posts], streamed, k),
        "latency": latency(a, posts, args.latency_posts),
        "threshold": a.threshold,
        "emerging_share": float(np.mean([e["emerging"] for e in placed])),
        "encode_check": {"posts": len(check), "min_cosine": float(cosines.min())},
        "emerging_examples": [
            {"post": db.display_id(posts[i]["id"]), "cluster_id": placed[i]["cluster_id"],
             "similarity": placed[i]["own_centroid_sim"], "newsgroup": posts[i]["label_name"],
             "opening": " ".join(posts[i]["text"].split())[:90].replace("|", "/")}
            for i in lowest
        ],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"stream-{args.run}-{time.strftime('%Y%m%d-%H%M%S')}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
    stem.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))
    print(f"Report: {stem}.json / .md")

    if args.save:
        everyone = db.query("""SELECT dc.cluster_id, d.label FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id
                               WHERE dc.run_id = %s""", (args.run,))
        labels = np.array([r["cluster_id"] for r in everyone] + streamed.tolist())
        truth = np.array([r["label"] for r in everyone] + gold.tolist())
        metrics = {"nmi": float(normalized_mutual_info_score(truth, labels)),
                   "ari": float(adjusted_rand_score(truth, labels)),
                   "purity": cluster.purity(truth, labels),
                   "n_clusters": int(len(np.unique(labels))), "noise_rate": 0.0,
                   "streamed_share": len(posts) / len(labels)}
        print(f"Saved run {save_streamed_run(a, posts, placed, metrics)} "
              f"({len(labels)} posts, NMI {metrics['nmi']:.3f} over all of them)")
