"""ClusterAgent intents: clusters named by label words, term questions, run comparison."""

from src.agents import cluster_agent as ca

CLUSTERS = {
    0: {"cluster_id": 0, "size": 400, "label": "braves · cubs · phillies (year)",
        "keywords": [["year", 0.04], ["game", 0.03], ["team", 0.02]]},
    8: {"cluster_id": 8, "size": 500, "label": "nasa · moon · earth (space)",
        "keywords": [["space", 0.05], ["nasa", 0.03], ["orbit", 0.02]]},
}


def test_find_cluster_by_number_label_or_term():
    assert ca.find_cluster("show cluster 8", CLUSTERS) == 8
    assert ca.find_cluster("cluster 99", CLUSTERS) is None
    assert ca.find_cluster("describe the braves cluster", CLUSTERS) == 0
    assert ca.find_cluster("the orbit one", CLUSTERS) == 8
    assert ca.find_cluster("tell me about clusters", CLUSTERS) is None


def test_why_term_question_is_parsed():
    m = ca.WHY_TERM.search("Why is year in braves cluster ?")
    assert m.group("term") == "year" and ca.find_cluster(m.group("ref"), CLUSTERS) == 0


def test_term_answer_and_routing(monkeypatch):
    def fake_query(sql, params=None):
        if "FROM cluster_info" in sql:
            return list(CLUSTERS.values())
        if "~*" in sql:
            return [{"hits": 120, "hits_all": 160}]
        if "FROM runs" in sql:
            return [{"run_id": "r", "metrics": {"nmi": 0.6}}]
        return []

    monkeypatch.setattr(ca.db, "query", fake_query)
    monkeypatch.setattr(ca.db, "default_run", lambda: "r")
    text = ca._answer("Why is year in braves cluster ?")["answer"]
    assert "#1 term" in text and "top term, in parentheses" in text and "30% of this cluster's 400 posts" in text
    assert ca._answer("compare hdbscan vs kmeans")["answer"].startswith("| run |")
    assert ca._answer("tell me about clusters")["answer"].startswith("Which cluster")
