"""Text2Cypher prompt (no Neo4j or LLM needed)."""

from src.agents.kg_agent import MAX_ROWS, PROMPT
from swm import kg


def test_schema_names_only_the_name_property():
    schema = kg.schema_text(kg.load_ontology())
    assert "{name: string}" in schema
    assert "lowercase}" not in schema   # once read as a property: {lowercase: 'nasa'}


def test_prompt_formats_with_examples():
    text = PROMPT.format(schema="S", max_rows=MAX_ROWS, feedback="", question="Q?")
    assert "{cluster_id: 3}" in text and f"LIMIT {MAX_ROWS}" in text
    assert "CONTAINS" not in text

