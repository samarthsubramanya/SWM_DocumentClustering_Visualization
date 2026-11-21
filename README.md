# Explainable News Clustering with Ontology-Anchored Knowledge Graphs

CSE 573 Semantic Web Mining · Fall 2026 · Group 17, Project 9

Samarth Agasthya Mandya Subramanya · Mohak Narendrakumar Rathod · Asmi Umesh Pulgam · Pratik Chaudhari

## Aim

Clustering shows *what* is grouped together, but not *why* a document belongs to its group. This
project clusters the 20 Newsgroups posts and then explains every assignment with evidence that
can be checked:

- **Evidence first, LLM last.** For each post we compute the evidence: the cluster's top terms
  found in the post, its margin over the closest competing cluster, its nearest posts, and the
  knowledge-graph entities it names. An LLM then only puts that evidence into words, and must quote it.
- **Faithfulness is tested, not assumed.** Deleting the terms an explanation cites should change
  assignments far more than deleting random words. It does: 10.5× as many flips among posts
  nearest their own cluster center (1.7× over all posts).
- **An ontology-anchored knowledge graph per cluster.** An LLM extracts typed entities and
  relations under a fixed ontology into Neo4j. PageRank over each cluster's graph names the cluster.
- **Questions in plain English.** Agents answer questions about clusters, posts and the graph.
  Graph questions become Cypher, which is validated, repaired if needed and run read-only.
  Its accuracy is measured against hand-written gold queries.
- **New posts as they arrive.** A new post joins the cluster its nearest posts vote for, with the
  same evidence, or is flagged as *emerging* when no cluster fits.

## Repository structure

```
CODE/            everything that runs (see CODE/README.md for step-by-step instructions)
  pipeline/        core library: ingest, embeddings, clustering, explanations, knowledge graph, evaluation
  src/             agent harness: LangGraph router, cluster / explain / Text2Cypher agents, FastAPI
  scripts/         pipeline stages 01_ingest … 07_human_eval
  ui/              Streamlit dashboard
  web/             Vue 3 + Material 3 dashboard
  ontology/        news.yaml: entity and relation types
  eval/            gold Text2Cypher questions
  tests/           pytest suite
DATA/            the 20 Newsgroups corpus (compressed archive) and its description
EVALUATIONS/     every evaluation output, by topic, with an index of headline results
  clustering/  phase1/  knowledge_graph/  text2cypher/  human_ratings/  dynamic/
```

## Pipeline

```
20 Newsgroups ─► clean ─► embed (TF-IDF+LSA · mpnet · BGE) ─► UMAP ─► cluster (k-means · agglomerative · HDBSCAN)
                                                                          │
              per-post evidence (terms, margin, nearest posts) ◄──────────┤
              LLM triple extraction ─► ontology check ─► Neo4j ─► PageRank cluster labels
                                                                          │
              dashboards (map · cluster panel · KG view · explanation card) + agents (chat, Text2Cypher)
```

## Headline results

| Component | Result | Target |
|---|---|---|
| Clustering (BGE + agglomerative, 17,090 posts) | NMI 0.663 | ≥ 0.55 |
| Explanation faithfulness (cited vs random deletions) | 10.5× more assignment flips (posts nearest their own cluster center; 1.7× over all posts) | ≥ 2× |
| Text2Cypher (held-out questions) | 100% valid on the first try, execution accuracy 9/9 | ≥ 0.70 |
| Knowledge-graph ontology conformance | 0.824 | ≥ 0.95 |
| Dynamic ingestion (6,785 unseen test posts) | agreement with batch 0.829, p95 0.033 s per post | ≥ 0.85, < 3 s |

Details, every report and the remaining human ratings are in [`EVALUATIONS/`](EVALUATIONS/README.md).

## Quick start

```bash
cd CODE
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env                 # database and LLM settings
python scripts/01_ingest.py --archive ../DATA/20newsgroups/20news-bydate.tar.gz
```

The rest of the pipeline, the evaluations and the dashboards are in [`CODE/README.md`](CODE/README.md).
