#!/usr/bin/env python3
"""Load 20 Newsgroups into Postgres. Run once; downstream scripts read from the DB."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from swm import db, ingest  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="all", choices=["train", "test", "all"])
    p.add_argument("--min-tokens", type=int, default=20)
    p.add_argument("--archive", type=Path, default=ingest.DEFAULT_ARCHIVE,
                   help="20news-bydate.tar.gz; downloaded here if missing")
    args = p.parse_args()

    db.init_schema()
    docs = ingest.load_20ng(args.subset, args.min_tokens, args.archive)
    ingest.save_documents(docs)
    print(f"Stored {len(docs)} documents.")
