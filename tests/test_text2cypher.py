"""Text2Cypher prompt and gold sets (no Neo4j or LLM needed)."""

from pathlib import Path

import yaml

from src.agents.kg_agent import MAX_ROWS, PROMPT
from swm import kg

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
