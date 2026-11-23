"""ponytail: one smoke test per stage as it lands, not a full suite up front."""


def test_ontology_yaml_loads():
    import yaml

    with open("ontology/news.yaml") as f:
        onto = yaml.safe_load(f)
    assert "entities" in onto and "relations" in onto
    assert "Person" in onto["entities"]
    assert "mentions" in onto["relations"]
