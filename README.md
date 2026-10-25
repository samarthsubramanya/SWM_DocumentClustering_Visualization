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

## Phase 1 run order (foundation)

```bash
docker compose up -d postgres neo4j          # + ollama, or point OLLAMA_BASE_URL at the college endpoint
python scripts/01_ingest.py                  # 20NG -> documents table
python scripts/02_embed.py                   # tfidf_lsa / mpnet / bge -> data/cache/*.npy
python scripts/03_cluster.py                 # UMAP + HDBSCAN/k-means/agglomerative grid -> runs, doc_cluster, cluster_info
python scripts/04_explain.py                 # evidence for every doc of the best-NMI run (--run to choose)
python scripts/05_kg.py                      # LLM triples (30 docs/cluster) -> Neo4j + PageRank labels + GraphML
python scripts/05_kg.py --all-docs           # optional: every doc (~60 h locally; cached, resumable)
python scripts/06_eval.py export             # -> data/eval/bundle (metrics, sampled explanations / triples / labels)
python scripts/06_eval.py run                # faithfulness + LLM-judge metrics -> eval/reports/phase1-*.{json,md}
uvicorn src.api.app:app --port 8000          # agents (chat)
streamlit run ui/streamlit_app.py            # dashboard
```

Each stage caches its output, so re-running one never recomputes upstream. `SWM_RUN_ID`
pins the run used by 04/05, the UI default, and the agents. Neo4j Browser: http://localhost:7474.

## Running Phase 1 on Sol (Apptainer + Voyager)

Everything runs on one interactive GPU node; Postgres and Neo4j run as Apptainer containers, the LLM is
Voyager (`LLM_PROVIDER=openai`, no Ollama).

```bash
# 0. once, on a login node: env, images, data
git clone/rsync the repo to $SCRATCH/swm && cd $SCRATCH/swm
module load mamba && mamba create -n swm python=3.11 -y && source activate swm && pip install -r requirements.txt
mkdir -p $SCRATCH/svc && cd $SCRATCH/svc
apptainer pull postgres.sif docker://postgres:16
apptainer pull neo4j.sif docker://neo4j:5-community
cd $SCRATCH/swm && mkdir -p data/raw && curl -L -o data/raw/20news-bydate.tar.gz https://ndownloader.figshare.com/files/5975967
cp .env.example .env     # set LLM_PROVIDER=openai, LLM_API_KEY, LLM_MODEL, POSTGRES_PORT=5433

# 1. interactive node
interactive -p public -q public -G a100:1 -c 8 --mem=48G -t 0-08:00   # adjust to your Sol allocation
source activate swm; cd $SCRATCH/swm; set -a; source .env; set +a

# 2. services (background, this node)
PG=$SCRATCH/svc/pg; mkdir -p $PG/data $PG/run
apptainer exec -B $PG:$PG ../svc/postgres.sif /usr/lib/postgresql/16/bin/initdb -D $PG/data -U swm --auth=trust
apptainer exec -B $PG:$PG ../svc/postgres.sif /usr/lib/postgresql/16/bin/pg_ctl -D $PG/data -l $PG/log \
  -o "-p 5433 -k $PG/run -c listen_addresses=127.0.0.1" start
apptainer exec -B $PG:$PG ../svc/postgres.sif /usr/lib/postgresql/16/bin/createdb -h 127.0.0.1 -p 5433 -U swm swm
N=$SCRATCH/svc/neo4j; mkdir -p $N/data $N/logs
nohup apptainer run --env NEO4J_AUTH=neo4j/changeme -B $N/data:/data -B $N/logs:/logs ../svc/neo4j.sif > $N/out.log 2>&1 &

# 3. pipeline
python scripts/01_ingest.py && python scripts/02_embed.py && python scripts/03_cluster.py
python scripts/04_explain.py
python scripts/05_kg.py --workers 8          # Voyager; 30 docs/cluster, cached and resumable
python scripts/06_eval.py export --n-plaus 100 --n-triples 200
python scripts/06_eval.py run --workers 8 [--judges modelA,modelB]   # >=2 judges -> Krippendorff alpha
# report: eval/reports/phase1-*.md
```

`SWM_RUN_ID=all-mpnet-base-v2__hdbscan` pins the run (faithfulness needs an SBERT run, not tfidf_lsa).
Judgments are cached in `data/cache/eval/`; rerunning resumes. Text2Cypher accuracy is not in this script.
