"""Ontology-anchored cluster knowledge graphs: extract -> normalize -> load -> label.

The ontology YAML is the only source of labels / relationship types: anything the LLM
returns outside it is rejected, and it is the only text ever interpolated into Cypher.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import islice
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from swm import db
from swm.represent import CACHE

DEFAULT_ONTOLOGY = Path(__file__).resolve().parents[2] / "ontology" / "news.yaml"


# ==========================================================
# Ontology
# ==========================================================

def _as_list(x) -> list:
    return x if isinstance(x, list) else [x]


def load_ontology(path=DEFAULT_ONTOLOGY) -> dict:
    onto = yaml.safe_load(Path(path).read_text())
    for spec in onto["relations"].values():
        spec["source"], spec["target"] = _as_list(spec["source"]), _as_list(spec["target"])
    return onto


def doc_relations(onto: dict) -> dict[str, str]:
    """entity type -> the Document-sourced relation that links a doc to it (mentions / about)."""
    return {t: rel for rel, spec in onto["relations"].items() if spec["source"] == ["Document"]
            for t in spec["target"]}


def entity_relations(onto: dict) -> dict[str, dict]:
    """Relations the LLM extracts (everything not sourced from Document)."""
    return {r: spec for r, spec in onto["relations"].items() if spec["source"] != ["Document"]}


def schema_text(onto: dict) -> str:
    """Graph schema description for Text2Cypher prompts."""
    lines = ["Node labels:", "- Document {id: string, cluster_id: integer, run_id: string}"]
    lines += [f"- {k} {{name: string, lowercase}}: {v['descriptor']}" for k, v in onto["entities"].items()]
    lines.append("Relationships (entity->entity relationships carry doc_ids: list of source document ids):")
    lines += [f"- (:{s})-[:{r}]->(:{t})" for r, spec in onto["relations"].items()
              for s in spec["source"] for t in spec["target"]]
    return "\n".join(lines)


def _ident(name: str, allowed) -> str:
    """Backtick-quote a label / relationship type after checking it is in the ontology."""
    if name not in allowed:
        raise ValueError(f"{name!r} is not in the ontology")
    return f"`{name}`"


# ==========================================================
# Extraction
# ==========================================================

class Entity(BaseModel):
    name: str = Field(description="Entity name exactly as written in the text")
    type: str = Field(description="One of the ontology entity types")


class Relation(BaseModel):
    source: str = Field(description="Name of the source entity (must appear in entities)")
    relation: str = Field(description="One of the ontology relation types")
    target: str = Field(description="Name of the target entity (must appear in entities)")


class Extraction(BaseModel):
    # Required (no defaults): with optional fields `{}` satisfies the JSON schema Ollama
    # constrains generation to, and llama3.1 took that shortcut on ~70% of documents.
    entities: list[Entity]
    relations: list[Relation]


# Bump when the prompt or schema changes: older cached extractions are then ignored.
EXTRACTION_VERSION = 2


def _prompt(onto: dict, text: str) -> str:
    ents = "\n".join(f"- {k}: {v['descriptor']}" for k, v in onto["entities"].items())
    rels = "\n".join(f"- {k}: {'|'.join(v['source'])} -> {'|'.join(v['target'])}. {v['descriptor']}"
                     for k, v in entity_relations(onto).items())
    return f"""Extract a knowledge graph from the document using ONLY this ontology.

Entity types:
{ents}

Relation types (source type -> target type):
{rels}

Rules:
- Only entities explicitly named in the text; use the exact surface form.
- At most 15 entities and 10 relations.
- Every relation's source and target must be entity names you listed, with matching types.
- If nothing fits, return empty lists.

Document:
\"\"\"{text[:3000]}\"\"\"
"""


def _clean(s) -> str:
    return " ".join(str(s or "").split())


def validate(raw: dict, onto: dict, doc_id: str) -> tuple[list[dict], list[dict], int]:
    """Keep only ontology-conformant entities and relations. Returns (entities, relations, n_rejected)."""
    linkable = set(doc_relations(onto))  # types a Document can point to
    rels = entity_relations(onto)
    ents: dict[str, tuple[str, str]] = {}
    rejected = 0
    for e in raw.get("entities", []):
        name = _clean(e.get("name"))
        if (e.get("type") in onto["entities"] and e.get("type") in linkable
                and 1 < len(name) <= 80 and name.lower() not in ENGLISH_STOP_WORDS):
            ents[name.lower()] = (name, e["type"])
        else:
            rejected += 1

    relations = []
    for r in raw.get("relations", []):
        spec = rels.get(r.get("relation"))
        s = ents.get(_clean(r.get("source")).lower())
        t = ents.get(_clean(r.get("target")).lower())
        if spec and s and t and s[1] in spec["source"] and t[1] in spec["target"]:
            relations.append({"source": s[0], "source_type": s[1], "relation": r["relation"],
                              "target": t[0], "target_type": t[1], "doc_id": doc_id})
        else:
            rejected += 1
    entities = [{"name": n, "type": t, "doc_id": doc_id} for n, t in ents.values()]
    return entities, relations, rejected


def extraction_cache_dir() -> Path:
    from swm.llm import model_name

    return CACHE / "kg" / f"{re.sub(r'[^\w.-]', '_', model_name())}-v{EXTRACTION_VERSION}"


def _extract_one(doc: dict, onto: dict) -> dict | None:
    """Cached raw extraction for one doc. None = Ollama unreachable (not cached, retried next run)."""
    from swm.llm import BadOutput, structured

    path = extraction_cache_dir() / (re.sub(r"[^\w.-]", "_", doc["id"]) + ".json")
    if path.exists():
        return json.loads(path.read_text())
    try:
        raw = structured(_prompt(onto, doc["text"]), Extraction).model_dump()
    except BadOutput:
        # Deterministic at temperature 0, so cache the failure; delete the file to retry.
        raw = {"entities": [], "relations": [], "failed": True}
    except RuntimeError:
        return None
    path.write_text(json.dumps(raw))
    return raw


def extract(docs: list[dict], onto: dict, workers: int = 1) -> tuple[list[dict], list[dict], dict]:
    """LLM extraction per doc ({id, text}), cached by (model, extraction version, doc id).

    workers > 1 sends that many requests concurrently; pair it with OLLAMA_NUM_PARALLEL on the
    server. Validation reruns on every call, so editing the ontology never needs new LLM calls.
    Returns (entities, relations, stats); stats feed the ontology-conformance metric.
    """
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    extraction_cache_dir().mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        raws = list(tqdm(pool.map(lambda d: _extract_one(d, onto), docs), total=len(docs),
                         desc="KG extraction"))

    entities, relations, stats = [], [], Counter()
    for d, raw in zip(docs, raws):
        if raw is None:
            stats["unreachable_docs"] += 1
            continue
        if raw.get("failed"):
            stats["failed_docs"] += 1
            continue
        e, r, rejected = validate(raw, onto, d["id"])
        entities += e
        relations += r
        stats["entities"] += len(e)
        stats["relations"] += len(r)
        stats["rejected"] += rejected
    return entities, relations, dict(stats)


# ==========================================================
# Normalization
# ==========================================================

def normalize(entities: list[dict], relations: list[dict], threshold: float = 0.9):
    """Lowercase names, then merge aliases within a type by embedding cosine > threshold."""
    from swm.represent import encode_texts

    freq = Counter((e["type"], e["name"].lower()) for e in entities)
    canon: dict[tuple[str, str], str] = {}
    # ponytail: greedy O(names x canonicals) per type; add blocking / ANN past ~50k names.
    for etype in {t for t, _ in freq}:
        names = sorted((n for t, n in freq if t == etype), key=lambda n: -freq[(etype, n)])
        V = encode_texts(names)
        keep: list[int] = []
        for i, name in enumerate(names):
            if keep:
                sims = V[keep] @ V[i]
                j = int(np.argmax(sims))
                if sims[j] > threshold:
                    canon[(etype, name)] = names[keep[j]]
                    continue
            keep.append(i)
            canon[(etype, name)] = name

    def c(etype, name):
        return canon[(etype, name.lower())]

    ents = {(c(e["type"], e["name"]), e["type"], e["doc_id"]) for e in entities}
    rels = {(c(r["source_type"], r["source"]), r["source_type"], r["relation"],
             c(r["target_type"], r["target"]), r["target_type"], r["doc_id"]) for r in relations}
    return (
        [{"name": n, "type": t, "doc_id": d} for n, t, d in sorted(ents)],
        [{"source": s, "source_type": st, "relation": rel, "target": t, "target_type": tt, "doc_id": d}
         for s, st, rel, t, tt, d in sorted(rels) if (s, st) != (t, tt)],
    )


# ==========================================================
# Neo4j
# ==========================================================

@lru_cache(maxsize=1)
def driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "changeme")),
    )


def compile_schema(onto: dict) -> None:
    """Ontology -> Neo4j constraints: unique Document.id and unique name per entity label."""
    drv = driver()
    drv.execute_query("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
    for label in onto["entities"]:
        drv.execute_query(
            f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
            f"FOR (n:{_ident(label, onto['entities'])}) REQUIRE n.name IS UNIQUE"
        )


def load_neo4j(run_id: str, doc_clusters: dict[str, int], entities: list[dict],
               relations: list[dict], onto: dict) -> None:
    """Replace the graph with one run's KG. doc_clusters: {doc_id: cluster_id} for extracted docs.

    The graph always mirrors exactly one run; extraction is cached, so reloading is cheap.
    """
    drv = driver()
    drv.execute_query("MATCH (n) DETACH DELETE n")
    drv.execute_query(
        "UNWIND $rows AS r MERGE (d:Document {id: r.id}) SET d.cluster_id = r.cluster_id, d.run_id = $run",
        rows=[{"id": d, "cluster_id": int(c)} for d, c in doc_clusters.items()],
        run=run_id,
    )

    doc_rel = doc_relations(onto)
    by_type = defaultdict(list)
    for e in entities:
        by_type[e["type"]].append(e)
    for etype, rows in by_type.items():
        label = _ident(etype, onto["entities"])
        rel = _ident(doc_rel[etype], onto["relations"])
        drv.execute_query(
            f"UNWIND $rows AS r MATCH (d:Document {{id: r.doc_id}}) "
            f"MERGE (e:{label} {{name: r.name}}) MERGE (d)-[:{rel}]->(e)",
            rows=rows,
        )

    by_rel = defaultdict(list)
    for r in relations:
        by_rel[(r["source_type"], r["relation"], r["target_type"])].append(r)
    for (st, rel, tt), rows in by_rel.items():
        drv.execute_query(
            f"""UNWIND $rows AS r
                MATCH (a:{_ident(st, onto['entities'])} {{name: r.source}}),
                      (b:{_ident(tt, onto['entities'])} {{name: r.target}})
                MERGE (a)-[x:{_ident(rel, onto['relations'])}]->(b)
                SET x.doc_ids = CASE WHEN r.doc_id IN coalesce(x.doc_ids, [])
                                     THEN x.doc_ids ELSE coalesce(x.doc_ids, []) + r.doc_id END""",
            rows=rows,
        )


# Write clauses are rejected before Neo4j sees the query; read_query also runs in a READ tx.
WRITE_CLAUSE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL)\b", re.IGNORECASE)
# EXPLAIN notifications Neo4j raises for names that don't exist in the graph (hallucinated schema).
SCHEMA_WARNINGS = ("UnknownLabelWarning", "UnknownRelationshipTypeWarning", "UnknownPropertyKeyWarning")


def validate_cypher(cypher: str) -> list[str]:
    """Problems with an LLM-written query, found without running it; [] means valid.

    1. write clauses (regex), 2. syntax/semantic errors (Neo4j EXPLAIN compiles the query),
    3. labels / relationship types / properties absent from the graph (EXPLAIN notifications).
    """
    from neo4j.exceptions import Neo4jError

    if m := WRITE_CLAUSE.search(cypher):
        return [f"writes to the graph ({m.group(0)}); only read queries are allowed"]
    try:
        with driver().session() as session:
            summary = session.run("EXPLAIN " + cypher).consume()
    except Neo4jError as exc:
        return [f"invalid Cypher: {exc.message}"]
    return [f"{n.title}: {n.description}" for n in summary.summary_notifications
            if n.code.endswith(SCHEMA_WARNINGS)]


def same_results(rows_a: list[dict], rows_b: list[dict]) -> bool:
    """Execution-accuracy match: equal multisets of row values, ignoring column names, column
    order, row order, string case and int/float representation."""
    def norm(v):
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(sorted(map(norm, v))) + "]"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return repr(round(float(v), 6))
        return str(v).strip().lower()

    def key(rows):
        return Counter(tuple(sorted(norm(v) for v in row.values())) for row in rows)

    return key(rows_a) == key(rows_b)


def read_query(cypher: str, max_rows: int = 50) -> list[dict]:
    """Run untrusted (LLM-written) Cypher in a READ transaction; Neo4j rejects writes there."""
    def work(tx):
        return [rec.data() for rec in islice(tx.run(cypher), max_rows)]

    with driver().session() as session:
        return session.execute_read(work)


SUBGRAPH_QUERY = """
MATCH (d:Document {run_id: $run, cluster_id: $cid})-->(e)
WITH e, count(DISTINCT d) AS freq ORDER BY freq DESC LIMIT $limit
WITH collect(e) AS ents, collect(freq) AS freqs
UNWIND range(0, size(ents) - 1) AS i
WITH ents, ents[i] AS e, freqs[i] AS freq
OPTIONAL MATCH (e)-[r]->(o) WHERE o IN ents
RETURN labels(e)[0] AS type, e.name AS name, freq,
       [x IN collect({relation: type(r), type: labels(o)[0], name: o.name}) WHERE x.relation IS NOT NULL] AS out
"""


def cluster_subgraph(run_id: str, cluster_id: int, limit: int = 40) -> tuple[list[dict], list[dict], str]:
    """Top entities of a cluster + edges among them, as (nodes, edges, source).

    Reads Neo4j; falls back to the GraphML export when Neo4j is down or holds another run.
    """
    from neo4j.exceptions import DriverError, Neo4jError

    try:
        recs, _, _ = driver().execute_query(SUBGRAPH_QUERY, run=run_id, cid=int(cluster_id), limit=limit)
        if recs:
            nodes = [{"id": f'{r["type"]}:{r["name"]}', "name": r["name"], "type": r["type"], "freq": r["freq"]}
                     for r in recs]
            edges = [{"source": n["id"], "target": f'{o["type"]}:{o["name"]}', "relation": o["relation"]}
                     for r, n in zip(recs, nodes) for o in r["out"]]
            return nodes, edges, "neo4j"
    except (DriverError, Neo4jError):
        pass

    path = graphml_path(run_id, cluster_id)
    if not path.exists():
        return [], [], "none"
    G = nx.read_graphml(path)
    ents = [n for n, a in G.nodes(data=True) if a.get("kind") == "entity"]
    freq = {n: sum(G.nodes[p].get("kind") == "document" for p in G.predecessors(n)) for n in ents}
    top = sorted(ents, key=freq.get, reverse=True)[:limit]
    keep = set(top)
    nodes = [{"id": n, "name": G.nodes[n]["name"], "type": G.nodes[n]["type"], "freq": freq[n]} for n in top]
    edges = [{"source": u, "target": v, "relation": a.get("relation", "")}
             for u, v, a in G.edges(data=True) if u in keep and v in keep]
    return nodes, edges, "graphml"


def doc_entities(doc_id: str) -> list[str]:
    """Entity names a document links to; [] if Neo4j is down or the doc wasn't extracted."""
    from neo4j.exceptions import DriverError, Neo4jError

    try:
        recs, _, _ = driver().execute_query("MATCH (:Document {id: $id})-->(e) RETURN e.name AS name", id=doc_id)
        return [r["name"] for r in recs]
    except (DriverError, Neo4jError):
        return []


# ==========================================================
# Cluster graphs, labels, GraphML fallback
# ==========================================================

def representative_docs(run_id: str, n: int | None = 30) -> list[dict]:
    """Top-n non-noise docs per cluster by cosine to the centroid (needs 04_explain evidence).

    n=None returns every doc of the run, noise included. Rows come round-robin by rank
    (each cluster's most central doc first), so an interrupted run still covers every
    cluster's core; extraction is cached, so rerunning resumes where it stopped.
    """
    return db.query(
        """SELECT doc_id, cluster_id, text FROM (
               SELECT e.doc_id, dc.cluster_id, d.text,
                      row_number() OVER (PARTITION BY dc.cluster_id
                                         ORDER BY (e.evidence->>'own_centroid_sim')::float DESC) AS rn
               FROM explanations e
               JOIN doc_cluster dc USING (run_id, doc_id)
               JOIN documents d ON d.id = e.doc_id
               WHERE e.run_id = %s AND (%s OR NOT dc.is_noise)
           ) ranked
           WHERE rn <= %s
           ORDER BY rn, cluster_id""",
        (run_id, n is None, n if n is not None else 2**31 - 1),
    )


def cluster_graph(cluster_docs, entities: list[dict], relations: list[dict], onto: dict) -> nx.DiGraph:
    """Document + entity nodes for one cluster, doc->entity and entity->entity edges."""
    docs = set(cluster_docs)
    doc_rel = doc_relations(onto)
    G = nx.DiGraph()
    for e in entities:
        if e["doc_id"] in docs:
            node, doc = f'{e["type"]}:{e["name"]}', f'Document:{e["doc_id"]}'
            G.add_node(node, kind="entity", type=e["type"], name=e["name"])
            G.add_node(doc, kind="document", type="Document", name=e["doc_id"])
            G.add_edge(doc, node, relation=doc_rel[e["type"]])
    for r in relations:
        if r["doc_id"] in docs:
            G.add_edge(f'{r["source_type"]}:{r["source"]}', f'{r["target_type"]}:{r["target"]}',
                       relation=r["relation"])
    return G


def pagerank_label(G: nx.DiGraph, top_term: str | None = None, k: int = 3) -> str:
    """Top-k PageRank entities (+ top c-TF-IDF term) as a human-readable cluster label."""
    if G.number_of_nodes() == 0:
        return top_term or ""
    pr = nx.pagerank(G.to_undirected())
    ranked = (G.nodes[n]["name"] for n in sorted(pr, key=pr.get, reverse=True) if G.nodes[n]["kind"] == "entity")
    top = list(dict.fromkeys(ranked))[:k]  # same name under two types (Location/Organization) counts once
    return " · ".join(top) + (f" ({top_term})" if top_term else "")


def graphml_path(run_id: str, cluster_id: int) -> Path:
    return CACHE / "graphml" / run_id / f"{int(cluster_id)}.graphml"


def export_graphml(G: nx.DiGraph, run_id: str, cluster_id: int) -> None:
    path = graphml_path(run_id, cluster_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, path)
