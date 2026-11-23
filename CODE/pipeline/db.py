"""Postgres access for the tabular artifacts: documents, runs, doc_cluster, cluster_info, explanations.

Embeddings are not stored here; they live in data/cache/*.npy (see pipeline.represent).
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb  # noqa: F401  (re-exported for callers writing JSONB)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,           -- e.g. train/sci.space/60151
    text        TEXT NOT NULL,
    label       INT,                        -- gold newsgroup index (20NG) / NULL for unlabeled
    label_name  TEXT,
    split       TEXT,
    source      TEXT NOT NULL DEFAULT '20ng'
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,           -- <embedding>__<algo>
    embedding   TEXT NOT NULL,
    algo        TEXT NOT NULL,
    params      JSONB NOT NULL DEFAULT '{}',
    metrics     JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS doc_cluster (
    run_id        TEXT REFERENCES runs ON DELETE CASCADE,
    doc_id        TEXT REFERENCES documents ON DELETE CASCADE,
    cluster_id    INT NOT NULL,             -- HDBSCAN noise reassigned to nearest centroid
    is_noise      BOOLEAN NOT NULL DEFAULT false,
    prob          REAL,                     -- HDBSCAN membership; NULL for k-means/agglomerative
    outlier_score REAL,
    x REAL,                                 -- 2-D UMAP for the scatter
    y REAL,
    PRIMARY KEY (run_id, doc_id)
);

CREATE TABLE IF NOT EXISTS cluster_info (
    run_id      TEXT REFERENCES runs ON DELETE CASCADE,
    cluster_id  INT,
    size        INT NOT NULL,
    keywords    JSONB NOT NULL,             -- [[term, c-TF-IDF weight], ...]
    label       TEXT,                       -- PageRank KG label, set by scripts/05_kg.py
    PRIMARY KEY (run_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS explanations (
    run_id      TEXT,
    doc_id      TEXT,
    evidence    JSONB NOT NULL,
    rationale   TEXT,                       -- LLM rationale, generated lazily
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, doc_id),
    FOREIGN KEY (run_id, doc_id) REFERENCES doc_cluster ON DELETE CASCADE
);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "swm"),
        user=os.getenv("POSTGRES_USER", "swm"),
        password=os.getenv("POSTGRES_PASSWORD", "swm"),
        row_factory=dict_row,
        options=os.getenv("POSTGRES_OPTIONS") or None,  # Neon + old libpq: "endpoint=<ep-id>" (no SNI)
        prepare_threshold=None,  # pgbouncer-style poolers (Neon -pooler hosts) reject server-side prepared statements
    )


def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)


def query(sql: str, params: tuple | None = None) -> list[dict]:
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def execute(sql: str, params: tuple | None = None) -> None:
    with connect() as conn:
        conn.execute(sql, params)


def executemany(sql: str, rows: list[tuple]) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)


def default_run() -> str:
    """SWM_RUN_ID if set, else the run with the best NMI."""
    if run_id := os.getenv("SWM_RUN_ID"):
        return run_id
    rows = query(
        "SELECT run_id FROM runs "
        "ORDER BY (metrics->>'nmi')::float DESC NULLS LAST, created_at DESC LIMIT 1"
    )
    if not rows:
        raise LookupError("No clustering runs yet; run scripts/03_cluster.py first.")
    return rows[0]["run_id"]


# ==========================================================
# Opaque post ids, shared by the dashboard and the agents
# ==========================================================

DISPLAY_ID = re.compile(r"\bdoc-[0-9a-f]{8}\b", re.IGNORECASE)


def display_id(doc_id: str) -> str:
    """Opaque id shown to users: real ids (train/sci.space/60151) carry the gold newsgroup."""
    return "doc-" + hashlib.sha1(doc_id.encode()).hexdigest()[:8]


@lru_cache(maxsize=1)
def _display_ids() -> dict[str, str]:
    return {display_id(r["id"]): r["id"] for r in query("SELECT id FROM documents")}


def resolve_display_id(token: str) -> str | None:
    """doc-3f9a1c2e -> the real document id, or None if no document has that id."""
    token = token.strip().lower()
    if token not in _display_ids():
        _display_ids.cache_clear()       # documents may have been added since the cache was built
    return _display_ids().get(token)
