"""Read-only explore API for the web dashboard (no Postgres or Neo4j needed)."""

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes import explore

client = TestClient(app)


def test_reports_are_listed_newest_first_and_readable():
    listed = client.get("/api/v1/explore/reports").json()
    assert listed and listed == sorted(listed, key=lambda r: r["stamp"], reverse=True)
    first = client.get(f"/api/v1/explore/reports/{listed[0]['name']}").json()
    assert first["name"] == listed[0]["name"] and first["content"]


def test_only_files_inside_eval_reports_are_served():
    assert client.get("/api/v1/explore/reports/..%2F..%2F.env").status_code == 404
    assert client.get("/api/v1/explore/reports/missing.md").status_code == 404


def test_unknown_post_is_a_404(monkeypatch):
    monkeypatch.setattr(explore.db, "resolve_display_id", lambda token: None)
    r = client.get("/api/v1/explore/runs/r/posts/doc-00000000")
    assert r.status_code == 404 and "doc-00000000" in r.json()["detail"]
