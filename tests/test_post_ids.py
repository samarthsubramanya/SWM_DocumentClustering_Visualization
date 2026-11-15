"""Opaque post ids shared by the dashboard and the agents (Postgres mocked)."""

from swm import db

IDS = ["train/sci.space/60151", "test/rec.autos/103007"]


def fake_query(sql, params=None):
    assert "FROM documents" in sql
    return [{"id": i} for i in IDS]


def test_display_id_round_trip(monkeypatch):
    monkeypatch.setattr(db, "query", fake_query)
    db._display_ids.cache_clear()
    shown = db.display_id(IDS[0])
    assert shown.startswith("doc-") and "sci.space" not in shown
    assert db.resolve_display_id(shown.upper()) == IDS[0]
    assert db.resolve_display_id("doc-00000000") is None


def test_explain_agent_accepts_dashboard_ids(monkeypatch):
    from src.agents import explain_agent

    monkeypatch.setattr(db, "query", fake_query)
    db._display_ids.cache_clear()
    question = f"why is {db.display_id(IDS[1])} in its cluster?"
    assert explain_agent._find_doc(question) == IDS[1]


def test_kg_rows_hide_real_ids():
    from src.agents.kg_agent import opaque_rows

    rows = opaque_rows([{"d.id": IDS[0], "n": 3, "ids": [IDS[1]], "name": "nasa"}])
    assert rows == [{"d.id": db.display_id(IDS[0]), "n": 3, "ids": [db.display_id(IDS[1])], "name": "nasa"}]
