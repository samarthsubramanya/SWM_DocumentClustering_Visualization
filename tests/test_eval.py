"""Pure-logic checks for the Phase 1 evaluation (no Postgres / Neo4j / LLM / SBERT needed)."""

import random

import numpy as np

from swm import evaluate as ev
from swm import kg


def test_delete_and_random_terms():
    text = "The Rocket reaches orbit; rocket engines and orbit insertion matter."
    out = ev.delete_terms(text, ["rocket", "orbit"])
    assert "rocket" not in out.lower() and "orbit" not in out.lower() and "engines" in out
    rnd = ev.random_terms(text, 2, ["rocket", "orbit"], random.Random(0))
    assert len(rnd) == 2 and not {"rocket", "orbit", "the", "and"} & set(rnd)
    assert ev.random_terms("rocket", 2, [], random.Random(0)) == []   # not enough vocabulary


def test_faithfulness_cited_hurts_more_than_random():
    # 2-d embedding: every word in CITED pushes the doc toward cluster 1; others are inert.
    C = np.array([[1.0, 0.0], [0.0, 1.0]])

    def encode(texts):
        v = np.array([[1.0, 0.1 + 0.4 * t.lower().count("rocket")] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    items = [{"doc_id": f"d{i}", "cluster_id": 1, "text": "rocket rocket alpha beta gamma delta epsilon",
              "cited": ["rocket"]} for i in range(5)]
    out = ev.faithfulness(items, [0, 1], C, encode, draws=2)
    assert out["n"] == 5 and out["frac_cited_hurts_more"] == 1.0
    assert out["margin_drop_cited"] > 0 and abs(out["margin_drop_random"]) < 1e-6
    assert out["margin_ratio"] is None   # random deletion changed nothing: ratio undefined, not inf


def test_cited_terms_ignore_text_the_encoder_never_sees():
    hits = [{"term": "rocket"}, {"term": "orbit"}]
    far = "x " * ev.TEXT_CAP + "rocket orbit"
    assert ev.cited_terms(hits, far) == []
    assert ev.cited_terms(hits, "rocket and orbit") == ["rocket", "orbit"]


def test_alpha_and_summaries():
    assert ev.krippendorff_alpha([[1, 2, 3]]) is None                       # one judge
    assert ev.krippendorff_alpha([[1, 2, 5, 4], [1, 2, 5, 4]]) == 1.0
    s = ev.summarize_plausibility({"a": [{"score": 4}, {"score": 9}, None], "b": [{"score": 3}, {"score": 4}, None]})
    assert s["invalid_outputs"] == 2 and abs(s["mean"] - np.mean([4, 5, 3, 4])) < 1e-9
    assert ev.summarize_triples({"a": [{"correct": True}, {"correct": False}, None]})["precision"] == 0.5
    assert ev.summarize_labels([True, True, False, True], 2, {})["kg_preferred"] == 0.75


def test_label_prompts_cover_both_orders():
    c = {"kg_label": "nasa", "ctfidf_label": "space, orbit", "samples": ["a", "b"]}
    (p1, f1), (p2, f2) = ev.label_prompts(c)
    assert (f1, f2) == ("kg", "ctfidf") and "Label A: nasa" in p1 and "Label A: space, orbit" in p2


def test_duplicate_rate_and_targets():
    ents = [{"type": "Organization", "name": n} for n in ("NASA", "nasa.", "The NASA", "esa")]
    assert ev.duplicate_rate(ents) == 0.5
    t = ev.check_targets({"best_nmi": 0.6, "faithfulness": {"margin_ratio": 1.5},
                          "kg": {"ontology_conformance": 0.97}})
    assert t["nmi"]["pass"] and not t["faithfulness_ratio"]["pass"] and t["triple_precision"]["pass"] is None


def test_cached_extract_never_calls_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "extraction_cache_dir", lambda: tmp_path)
    (tmp_path / "d1.json").write_text('{"entities":[{"name":"NASA","type":"Organization"}],"relations":[]}')
    onto = kg.load_ontology()
    ents, rels, stats = kg.cached_extract([{"id": "d1", "text": ""}, {"id": "d2", "text": ""}], onto)
    assert [e["name"] for e in ents] == ["NASA"] and stats["unreachable_docs"] == 1
