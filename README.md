# Explainable News Clustering with Ontology-Anchored Knowledge Graphs

CSE 573 Semantic Web Mining — implementation repo. 

## Implementation goals

1. **Foundation (required).** Cluster the 20 Newsgroups corpus, explain *why* each
   document lands in its cluster with checkable evidence (not just a keyword list),
   and overlay an ontology-anchored knowledge graph per cluster, all through a
   linked Streamlit dashboard and a set of harness agents.
2. **Optional A — Dynamic ingestion.** Assign new articles (pasted text or RSS) to
   an existing cluster or flag them as emerging, with the same explanation.
3. **Optional B — PubMed adaptation.** Re-run the unchanged pipeline on PubMed
   abstracts by swapping only the ingest loader and the ontology file.

## Architecture (overview)

```
Streamlit UI ── linked views: UMAP scatter | cluster panel | KG subgraph (PyVis) | explanation card | chat
     │ HTTP
     ▼
FastAPI -> Base Orchestrator (LangGraph)
     │
     ├─ ClusterAgent   "show cluster 3" / "compare HDBSCAN vs k-means"
     ├─ ExplainAgent   "why is doc 42 in cluster 3?" -> evidence -> LLM rationale
     ├─ KGAgent        Text2Cypher: NL question -> Cypher -> Neo4j (mirrors Text2SQL)
     └─ IngestAgent    (Optional A) new article -> embed -> assign or flag emerging
     │
     ▼
swm/  core library (pure Python; every stage caches its output)
  ingest/     represent/       cluster/           explain/              kg/
  20NG loader  TF-IDF+LSA       UMAP -> HDBSCAN    c-TF-IDF terms        ontology.yaml
  PubMed*      SBERT mpnet      k-means, agg.      centroid margin       LLM triple extraction
  RSS*         BGE-base         NMI/ARI/silhouette exemplars, LLM        normalize -> Neo4j
                                                    rationale             PageRank labels
     │
     ▼
PostgreSQL (tabular: documents, embeddings, runs, doc_cluster, explanations)
Neo4j       (graph: ontology entity types as labels, relations as relationship types)
Ollama      (Llama 3.1 / Qwen 2.5, structured output via pydantic)
```

Three rules drive every design choice here:

- **Cached artifacts between stages.** Embeddings are `.npy` files; cluster runs
  and explanations live in Postgres. Re-running a stage never recomputes its
  upstream, and the demo never depends on a live model call for the static views.
- **Evidence-first explanations.** The LLM only verbalizes structured evidence
  (cluster keywords present, centroid margin, nearest exemplars, KG entities) —
  it cannot invent evidence. This is what makes the faithfulness test meaningful.
- **Ontology as config, not code.** `ontology/news.yaml` compiles directly into
  the Neo4j schema (labels, relationship types, constraints). Optional B is a
  file swap, not a rewrite.


## Getting started

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in Ollama endpoint (college cloud / supercomputer)
docker compose up -d        # postgres + neo4j (+ local ollama, optional)
pytest                      # smoke test: ontology.yaml parses
```

## Layout

```
swm/            core library — ingest, represent, cluster, explain, kg
src/agents/     harness agent wrappers around swm/
scripts/        01_ingest -> 02_embed -> 03_cluster -> 04_explain -> 05_kg -> 06_eval
ontology/       news.yaml (foundation); medical.yaml added for Optional B
ui/             Streamlit dashboard (extends the course harness's ui/)
data/cache/     .npy embedding cache (gitignored)
```

