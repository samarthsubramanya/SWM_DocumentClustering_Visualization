#!/usr/bin/env python3
"""Extract, normalize, and load per-cluster knowledge graphs into Neo4j; compute PageRank labels.

Offline extraction (e.g. on Sol, no Postgres/Neo4j there):
  local:  python scripts/05_kg.py --all-docs --export-docs data/kg_docs.jsonl
  Sol:    python scripts/05_kg.py --docs-file data/kg_docs.jsonl --workers 16   (fills the cache)
  local:  copy data/cache/kg/ back, then python scripts/05_kg.py --all-docs   (all cache hits)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from swm import db, kg  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="run_id (default: SWM_RUN_ID or best NMI)")
    size = p.add_mutually_exclusive_group()
    size.add_argument("--docs-per-cluster", type=int, default=30,
                      help="representative docs per cluster (closest to the centroid)")
    size.add_argument("--all-docs", action="store_true",
                      help="extract from every doc of the run (~13 s/doc locally; cached and resumable)")
    p.add_argument("--ontology", default=str(kg.DEFAULT_ONTOLOGY))
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent LLM requests; match OLLAMA_NUM_PARALLEL on the server")
    io = p.add_mutually_exclusive_group()
    io.add_argument("--export-docs", type=Path, metavar="JSONL",
                    help="write the selected docs to JSONL and exit (for offline extraction)")
    io.add_argument("--docs-file", type=Path, metavar="JSONL",
                    help="extract docs from JSONL into the cache only; no Postgres/Neo4j needed")
    args = p.parse_args()
    onto = kg.load_ontology(args.ontology)

    def report(stats: dict) -> None:
        kept = stats.get("entities", 0) + stats.get("relations", 0)
        total = kept + stats.get("rejected", 0)
        print(f"Extraction: {stats}; ontology conformance {kept / max(total, 1):.3f}")
        if stats.get("unreachable_docs"):
            print(f"WARNING: {stats['unreachable_docs']} docs skipped (Ollama unreachable); rerun to fill them in.")

    if args.docs_file:
        docs = [json.loads(line) for line in args.docs_file.read_text().splitlines() if line.strip()]
        print(f"Extracting {len(docs)} documents from {args.docs_file} into {kg.extraction_cache_dir()}.")
        report(kg.extract(docs, onto, args.workers)[2])
        raise SystemExit(0)

    run_id = args.run or db.default_run()
    reps = kg.representative_docs(run_id, None if args.all_docs else args.docs_per_cluster)
    if not reps:
        raise SystemExit(f"No evidence for {run_id}; run scripts/04_explain.py --run {run_id} first.")
    docs = [{"id": r["doc_id"], "text": r["text"]} for r in reps]

    if args.export_docs:
        args.export_docs.parent.mkdir(parents=True, exist_ok=True)
        args.export_docs.write_text("".join(json.dumps(d) + "\n" for d in docs))
        raise SystemExit(f"Wrote {len(docs)} documents of {run_id} to {args.export_docs}.")

    print(f"Extracting from {len(docs)} documents of {run_id}.")
    entities, relations, stats = kg.extract(docs, onto, args.workers)
    report(stats)

    entities, relations = kg.normalize(entities, relations)
    kg.compile_schema(onto)
    kg.load_neo4j(run_id, {r["doc_id"]: r["cluster_id"] for r in reps}, entities, relations, onto)
    print(f"Neo4j: {len(entities)} doc-entity links, {len(relations)} relations for {run_id}.")

    docs_by_cluster = defaultdict(list)
    for r in reps:
        docs_by_cluster[r["cluster_id"]].append(r["doc_id"])
    keywords = {r["cluster_id"]: r["keywords"] for r in db.query(
        "SELECT cluster_id, keywords FROM cluster_info WHERE run_id = %s", (run_id,))}

    for cid, docs in sorted(docs_by_cluster.items()):
        G = kg.cluster_graph(docs, entities, relations, onto)
        kg.export_graphml(G, run_id, cid)
        label = kg.pagerank_label(G, keywords[cid][0][0] if keywords.get(cid) else None)
        db.execute("UPDATE cluster_info SET label = %s WHERE run_id = %s AND cluster_id = %s", (label, run_id, cid))
        print(f"cluster {cid}: {label}")
