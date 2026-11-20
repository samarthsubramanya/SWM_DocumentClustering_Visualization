"""Text2Cypher prompt and gold sets (no Neo4j or LLM needed)."""

from pathlib import Path

import yaml

from src.agents.kg_agent import MAX_ROWS, PROMPT
from pipeline import kg

ROOT = Path(__file__).resolve().parents[1]


def test_schema_names_only_the_name_property():
    schema = kg.schema_text(kg.load_ontology())
    assert "{name: string}" in schema
    assert "lowercase}" not in schema   # once read as a property: {lowercase: 'nasa'}


def test_prompt_formats_with_examples():
    text = PROMPT.format(schema="S", max_rows=MAX_ROWS, feedback="", question="Q?")
    assert "{cluster_id: 3}" in text and f"LIMIT {MAX_ROWS}" in text
    assert "CONTAINS" not in text


def test_gold_sets_are_well_formed():
    ids = set()
    for name in ("text2cypher.yaml", "text2cypher_heldout.yaml"):
        questions = yaml.safe_load((ROOT / "eval" / name).read_text())["questions"]
        for q in questions:
            assert {"id", "pattern", "question", "gold"} <= q.keys()
            assert q["gold"].lstrip().upper().startswith("MATCH")
            assert q["id"] not in ids
            ids.add(q["id"])
    assert len(ids) == 30


def test_empty_llm_summary_falls_back_to_rows():
    from src.agents.kg_agent import rows_answer

    assert rows_answer([]) == "No matching data was found."
    text = rows_answer([{"d.cluster_id": i} for i in range(25)], shown=20)
    assert text.startswith("25 results:") and "… and 5 more" in text


def test_llm_failure_is_a_failed_attempt_not_a_crash(monkeypatch):
    import asyncio

    from src.agents import kg_agent

    class Broken:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt):
            raise RuntimeError("length limit was reached")

    monkeypatch.setattr(kg_agent, "get_llm", lambda: Broken())
    monkeypatch.setattr(kg_agent.kg, "load_ontology", lambda: {"entities": {}, "relations": {}})
    out = asyncio.run(kg_agent.text2cypher("q?"))
    assert out["rows"] is None and len(out["problems"]) == kg_agent.ATTEMPTS
    assert out["problems"][0]["problems"][0].startswith("no query produced")
