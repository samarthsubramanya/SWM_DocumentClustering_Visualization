"""Cluster core entities and the post -> core path shown in the explanation card."""

import networkx as nx

from swm import kg


def toy_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    ent = {"nasa": "Organization", "general dynamics": "Organization", "ssrt": "Artifact", "houston": "Location"}
    for name, typ in ent.items():
        G.add_node(f"{typ}:{name}", kind="entity", type=typ, name=name)
    for d, names in {"d1": ["nasa", "general dynamics"], "d2": ["ssrt", "nasa"], "d3": ["houston"],
                     "d4": ["houston", "general dynamics"]}.items():
        G.add_node(f"Document:{d}", kind="document", type="Document", name=d)
        for n in names:
            G.add_edge(f"Document:{d}", f"{ent[n]}:{n}", relation="mentions")
    G.add_edge("Organization:general dynamics", "Artifact:ssrt", relation="produces")
    return G


def test_label_is_built_from_the_core():
    G = toy_graph()
    core = kg.top_entities(G, 2)
    assert len(core) == 2 and all(G.nodes[n]["kind"] == "entity" for n in core)
    assert kg.pagerank_label(G, None, 2) == " · ".join(G.nodes[n]["name"] for n in core)


def test_extracted_post_starts_from_its_own_node():
    out = kg.path_to_core(toy_graph(), "d3", "", k=2)
    assert out["mentions"] == ["Location:houston"]
    assert out["path"][0] == "Document:d3" and out["path"][-1] in out["core"]


def test_other_posts_enter_through_named_entities():
    G = toy_graph()
    out = kg.path_to_core(G, "unseen", "Flew out of Houston, Texas.", k=1)
    assert out["mentions"] == ["Location:houston"]
    assert out["path"][0] == "Location:houston" and out["path"][-1] == out["core"][0]
    assert kg.path_to_core(G, "unseen", "Nothing relevant; houstonian is not a match.", k=1)["path"] is None
