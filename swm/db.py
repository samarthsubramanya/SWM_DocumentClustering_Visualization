"""Postgres access for the tabular artifacts: documents, runs, doc_cluster, cluster_info, explanations.

Embeddings are not stored here; they live in data/cache/*.npy (see swm.represent).
"""

from __future__ import annotations

import os

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
