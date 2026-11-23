#!/usr/bin/env python3
"""Give another clustering run its own knowledge graph, without touching Neo4j.

Neo4j holds one run's graph (05_kg.py replaces the whole graph when it loads a run). Entities
are extracted per post, so every run over the same posts shares them; only the grouping of
posts into clusters differs. This script reads the loaded graph (read-only), regroups its
posts by the clusters of --run, and writes:

  data/cache/graphml/<run>/<cluster>.graphml   one file per cluster (the dashboard reads these)
  cluster_info.label for --run                 PageRank labels, as 05_kg.py computes them

  python scripts/05b_kg_per_run.py --run all-mpnet-base-v2__kmeans
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import networkx as nx  # noqa: E402

from pipeline import db, kg  # noqa: E402

MENTIONS = "MATCH (d:Document)-[m]->(e) RETURN d.id AS doc, type(m) AS relation, labels(e)[0] AS type, e.name AS name"
RELATIONS = """MATCH (a)-[r]->(b) WHERE NOT a:Document
RETURN labels(a)[0] AS source_type, a.name AS source, type(r) AS relation,
       labels(b)[0] AS target_type, b.name AS target, coalesce(r.doc_ids, []) AS doc_ids"""


def read_graph():
    """Every post -> entity link and every entity -> entity relation in Neo4j (read-only)."""
    drv = kg.driver()
    mentions, _, _ = drv.execute_query(MENTIONS, routing_="r")
    relations, _, _ = drv.execute_query(RELATIONS, routing_="r")
    loaded, _, _ = drv.execute_query("MATCH (d:Document) RETURN DISTINCT d.run_id AS run", routing_="r")
    return [r.data() for r in mentions], [r.data() for r in relations], [r["run"] for r in loaded]


def cluster_graph(docs: set[str], mentions: list[dict], relations: list[dict]) -> nx.DiGraph:
    """Same nodes, ids and attributes as kg.cluster_graph and the GraphML that 05_kg.py writes."""
    G = nx.DiGraph()
    for m in mentions:
        if m["doc"] in docs:
            node, doc = f'{m["type"]}:{m["name"]}', f'Document:{m["doc"]}'
            G.add_node(node, kind="entity", type=m["type"], name=m["name"])
            G.add_node(doc, kind="document", type="Document", name=m["doc"])
            G.add_edge(doc, node, relation=m["relation"])
    for r in relations:
        if docs.intersection(r["doc_ids"]):   # a relation belongs to the clusters of the posts that stated it
            G.add_edge(f'{r["source_type"]}:{r["source"]}', f'{r["target_type"]}:{r["target"]}',
                       relation=r["relation"])
    return G


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the run to build graphs for")
    args = p.parse_args()

    mentions, relations, loaded = read_graph()
    if args.run in loaded:
        sys.exit(f"{args.run} is the run loaded in Neo4j; the dashboard already reads its graph there.")
    print(f"Neo4j holds {loaded}: {len(mentions)} post-entity links, {len(relations)} relations.")

    members = defaultdict(set)
    for r in db.query("SELECT doc_id, cluster_id FROM doc_cluster WHERE run_id = %s", (args.run,)):
        members[r["cluster_id"]].add(r["doc_id"])
    if not members:
        sys.exit(f"No clustering for {args.run} in Postgres.")
    covered = {m["doc"] for m in mentions}
    keywords = {r["cluster_id"]: r["keywords"] for r in db.query(
        "SELECT cluster_id, keywords FROM cluster_info WHERE run_id = %s", (args.run,))}

    for cid, docs in sorted(members.items()):
        G = cluster_graph(docs, mentions, relations)
        kg.export_graphml(G, args.run, cid)
        label = kg.pagerank_label(G, keywords[cid][0][0] if keywords.get(cid) else None)
        db.execute("UPDATE cluster_info SET label = %s WHERE run_id = %s AND cluster_id = %s",
                   (label, args.run, cid))
        print(f"cluster {cid}: {len(docs & covered)}/{len(docs)} posts in the graph, "
              f"{G.number_of_nodes()} nodes · {label}")
    print(f"GraphML: {kg.graphml_path(args.run, 0).parent}")
