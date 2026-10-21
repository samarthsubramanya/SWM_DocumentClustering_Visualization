#!/usr/bin/env python3
"""Compute evidence for every document in the selected run (+ optional LLM rationales for a sample).

Rationales for the rest are generated lazily by the UI / ExplainAgent and cached.
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from swm import db, explain, represent  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", help="run_id (default: SWM_RUN_ID or best NMI)")
    p.add_argument("--rationale-sample", type=int, default=0, help="pre-generate N LLM rationales")
    args = p.parse_args()

    run_id = args.run or db.default_run()
    embedding = db.query("SELECT embedding FROM runs WHERE run_id = %s", (run_id,))[0]["embedding"]
    rows = db.query(
        """SELECT dc.doc_id, dc.cluster_id, dc.is_noise, dc.prob, d.text
           FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id
           WHERE dc.run_id = %s ORDER BY dc.doc_id""",
        (run_id,),
    )
    keywords = {r["cluster_id"]: r["keywords"] for r in db.query(
        "SELECT cluster_id, keywords FROM cluster_info WHERE run_id = %s", (run_id,))}

    emb_ids, X = represent.load(embedding)
    pos = {doc_id: i for i, doc_id in enumerate(emb_ids)}
    ids = [r["doc_id"] for r in rows]

    evidence = explain.compute_evidence(
        ids,
        [r["text"] for r in rows],
        X[[pos[d] for d in ids]],
        np.array([r["cluster_id"] for r in rows]),
        np.array([r["is_noise"] for r in rows]),
        [r["prob"] for r in rows],
        keywords,
    )
    explain.save_evidence(run_id, ids, evidence)
    print(f"{run_id}: evidence for {len(ids)} documents.")

    for doc_id in tqdm(random.Random(42).sample(ids, min(args.rationale_sample, len(ids))), desc="rationales"):
        explain.rationale(doc_id, run_id)
