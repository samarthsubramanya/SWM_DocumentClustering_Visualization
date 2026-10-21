#!/usr/bin/env python3
"""Compute and cache TF-IDF+LSA, SBERT, and BGE embeddings to data/cache/*.npy."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from swm import ingest, represent  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(represent.MODELS), choices=represent.MODELS)
    p.add_argument("--force", action="store_true", help="recompute even if the cache matches")
    args = p.parse_args()

    docs = ingest.load_documents()
    if not docs:
        raise SystemExit("No documents; run scripts/01_ingest.py first.")
    for model in args.models:
        X = represent.embed(docs, model, force=args.force)
        print(f"{model}: {X.shape}")
