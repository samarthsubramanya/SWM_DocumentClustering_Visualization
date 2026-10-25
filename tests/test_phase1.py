"""Pure-logic checks for Phase 1 (no Postgres / Neo4j / Ollama needed)."""

import numpy as np

from swm import cluster, explain, kg


def test_purity_and_noise_assignment():
    assert cluster.purity(np.array([0, 0, 1, 1]), np.array([5, 5, 7, 7])) == 1.0
    X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.95, 0.05]])
    labels = np.array([0, 0, 1, 1, -1])
    assert cluster.assign_noise(X, labels).tolist() == [0, 0, 1, 1, 0]


def test_ctfidf_and_evidence():
    texts = ["rocket orbit launch nasa"] * 3 + ["hockey goal playoff team"] * 3
    labels = np.array([0, 0, 0, 1, 1, 1])
    kw = explain.ctfidf(texts, labels, min_df=1)
    assert {t for t, _ in kw[0]} >= {"rocket", "orbit"} and "hockey" not in {t for t, _ in kw[0]}

    X = np.array([[1, 0], [0.9, 0.1], [0.95, 0], [0, 1], [0.1, 0.9], [0, 0.95]], dtype=float)
    ev = explain.compute_evidence([f"d{i}" for i in range(6)], texts, X, labels,
                                  np.zeros(6, bool), None, kw)
    assert ev[0]["margin"] > 0 and ev[0]["runner_up_cluster"] == 1
    assert {h["term"] for h in ev[0]["term_hits"]} >= {"rocket", "nasa"}
    assert all(e["doc_id"] != "d0" for e in ev[0]["exemplars"])
    text = explain.template_rationale(ev[0])
    assert len(explain.quoted_terms(text, explain.evidence_terms(ev[0]))) >= 2


def test_ontology_validation_rejects_off_schema():
    onto = kg.load_ontology()
    raw = {
        "entities": [{"name": "NASA", "type": "Organization"}, {"name": "Shuttle", "type": "Artifact"},
                     {"name": "Bob", "type": "Spaceship"}, {"name": "I", "type": "Person"}],
        "relations": [{"source": "NASA", "relation": "produces", "target": "Shuttle"},
                      {"source": "Shuttle", "relation": "produces", "target": "NASA"},
                      {"source": "NASA", "relation": "invented", "target": "Shuttle"}],
    }
    entities, relations, rejected = kg.validate(raw, onto, "d1")
    assert {e["name"] for e in entities} == {"NASA", "Shuttle"}
    assert [(r["source"], r["target"]) for r in relations] == [("NASA", "Shuttle")]
    assert rejected == 4


def test_encoded_attachments_are_detected():
    from swm.ingest import MAX_ENCODED_SHARE, encoded_share

    uuencoded = "begin 644 a.gif\n" + "\n".join(["M1TE&.#=AN@*6`?```````/___RP`````N@*6`0`\"_@Q@J)O-[XQ4#IYX::Z3"] * 20)
    assert encoded_share(uuencoded) > MAX_ENCODED_SHARE
    prose = "Thanks, I think I've figured out the driver problem with my card now. " * 3
    assert encoded_share(prose + "\n" + "-" * 70 + "\nBob") < MAX_ENCODED_SHARE


def test_extraction_schema_forces_both_keys():
    # Optional fields let the constrained decoder emit `{}`; that emptied ~70% of extractions.
    assert set(kg.Extraction.model_json_schema()["required"]) == {"entities", "relations"}


def test_text2cypher_result_matching_and_write_guard():
    gold = [{"name": "nasa"}, {"name": "ibm"}]
    assert kg.same_results([{"o.name": "IBM"}, {"x": "nasa"}], gold)       # names/case/order ignored
    assert not kg.same_results([{"name": "nasa"}], gold)                     # missing row
    assert kg.same_results([{"count(d)": 3}], [{"n": 3.0}])                   # int vs float
    assert kg.WRITE_CLAUSE.search("MATCH (n) DETACH DELETE n")
    assert not kg.WRITE_CLAUSE.search("MATCH (d:Document) RETURN d.id")
