# Code: Explainable News Clustering with Ontology-Anchored Knowledge Graphs

CSE 573 Semantic Web Mining, Group 17. This folder holds the pipeline, the agents, both
dashboards and the tests. The dataset is in `../DATA/` and every evaluation output in
`../EVALUATIONS/`. **Run every command below from this `CODE/` folder.**

## Architecture (overview)

```
Web UI (Vue 3, web/)  ·  Streamlit UI (ui/) ── map of posts | cluster panel | KG subgraph | explanation card | chat
     │ HTTP
     ▼
FastAPI (src/api) -> Base Orchestrator (LangGraph)          + read-only /api/v1/explore/* for the web UI
     │
     ├─ ClusterAgent   "describe the braves cluster" / "why is year in braves?" / "compare the runs"
     ├─ ExplainAgent   "why is doc-3f9a1c2e in its cluster?" -> evidence -> LLM rationale
     ├─ KGAgent        Text2Cypher: question -> Cypher -> validate (EXPLAIN) -> repair -> read-only Neo4j
     └─ IngestAgent    (optional, not implemented) new article -> assign or flag emerging
     │
     ▼
pipeline/  core library (pure Python; every stage caches its output)
  ingest/      represent/       cluster/           explain/              kg/
  20NG loader  TF-IDF+LSA       UMAP -> HDBSCAN    c-TF-IDF terms        ontology/news.yaml
               SBERT mpnet      k-means, agg.      centroid margin       LLM triple extraction
               BGE-base         NMI/ARI/silhouette exemplars, LLM        normalize -> Neo4j
                                                   rationale             PageRank labels
     │
     ▼
PostgreSQL (tabular: documents, runs, doc_cluster, cluster_info, explanations)
Neo4j      (graph: ontology entity types as labels, relations as relationship types)
LLM        (Ollama locally, or an OpenAI-compatible endpoint such as ASU RC)
```

Three rules drive every design choice here:

- **Cached artifacts between stages.** Embeddings are `.npy` files; cluster runs
  and explanations live in Postgres. Re-running a stage never recomputes its
  upstream, and the dashboards never depend on a live model call for the static views.
- **Evidence-first explanations.** The LLM only verbalizes structured evidence
  (cluster keywords present, centroid margin, nearest exemplars, KG entities). It
  cannot invent evidence, which is what makes the faithfulness test meaningful.
- **Ontology as config, not code.** `ontology/news.yaml` compiles directly into
  the Neo4j schema (labels, relationship types, constraints).

## Layout

```
pipeline/       core library: ingest, represent, cluster, explain, kg, evaluate, db, llm
src/agents/     harness agents (cluster, explain, KG / Text2Cypher, ingest stub) around pipeline/
src/api/        FastAPI app: /api/v1/query (chat), /api/v1/explore/* (web UI data), /health
scripts/        01_ingest -> 02_embed -> 03_cluster -> 04_explain -> 05_kg -> 06_eval -> 07_human_eval
ontology/       news.yaml: entity and relation types for the knowledge graph
ui/             Streamlit dashboard
web/            Vue 3 + Material 3 dashboard (see web/README.md)
eval/           gold Text2Cypher questions (inputs to scripts/06_eval.py)
tests/          pytest suite (no database or LLM needed)
data/           local caches: embeddings, extraction cache, GraphML (gitignored)
```

## Step by step

### 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest                      # 34 tests, no database or LLM needed
```

### 2. Configure `.env`

- **Databases:**
  - **Cloud:** set `POSTGRES_HOST/PORT/DB/USER/PASSWORD` (Neon) and `NEO4J_URI/USER/PASSWORD` (Aura).
  - **Local:** start Postgres and Neo4j with `docker compose up -d postgres neo4j`. Neo4j Browser: http://localhost:7474.
- **LLM:**
  - **ASU RC:** `LLM_PROVIDER=openai`, `LLM_MODEL=qwen3-30b-a3b`, `LLM_API_KEY=…`, and `LLM_MAX_TOKENS=8192` (it is a thinking model).
  - **Local Ollama:** `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.1:8b`.
- **Run:** `SWM_RUN_ID` pins the run used by steps 4–5, the dashboards' default and the agents.
- **Certificates:** if Python cannot verify Aura's certificate (`CERTIFICATE_VERIFY_FAILED`), add
  `SSL_CERT_FILE=<path printed by python -m certifi>`.

### 3. Build the pipeline

```bash
python scripts/01_ingest.py --archive ../DATA/20newsgroups/20news-bydate.tar.gz   # 20NG -> documents table
python scripts/02_embed.py                   # tfidf_lsa / mpnet / bge -> data/cache/*.npy
python scripts/03_cluster.py                 # UMAP + HDBSCAN/k-means/agglomerative grid -> runs, doc_cluster, cluster_info
python scripts/04_explain.py                 # evidence for every post of the run (--run to choose)
python scripts/05_kg.py                      # LLM triples (30 posts/cluster) -> Neo4j + PageRank labels + GraphML
python scripts/05_kg.py --all-docs           # optional: every post (~60 h locally; cached, resumable)
python scripts/05b_kg_per_run.py --run <run> # other runs: regroup the loaded graph by their clusters (no LLM, Neo4j untouched)
```

Each stage caches its output, so re-running one never recomputes upstream. Without
`--archive`, step 1 downloads the dataset to `data/raw/` and checks its SHA-256.

### 4. Evaluate (outputs go to `../EVALUATIONS/`)

```bash
python scripts/03b_k_select.py --embedding bge-base-en-v1.5   # label-free k        -> EVALUATIONS/clustering/
python scripts/03_cluster.py --seeds 42 43 44                 # seed stability      -> EVALUATIONS/clustering/
python scripts/06_eval.py export                              # -> data/eval/bundle (sampled explanations / triples / labels)
python scripts/06_eval.py run --judges <judge1>,<judge2>      # faithfulness + LLM judges -> EVALUATIONS/phase1/
python scripts/06_eval.py text2cypher                         # Text2Cypher validity + accuracy -> EVALUATIONS/text2cypher/
python scripts/06_eval.py text2cypher --questions eval/text2cypher_heldout.yaml
python scripts/07_human_eval.py make --raters samarth mohak asmi pratik   # rating sheets in eval/human/
python scripts/07_human_eval.py score                         # -> EVALUATIONS/human_ratings/
```

### 5. Run the dashboards

```bash
uvicorn src.api.app:app --port 8000          # agents (chat) + data API, needed by both dashboards
streamlit run ui/streamlit_app.py            # Streamlit dashboard, http://localhost:8501
cd web && npm install && npm run dev         # web dashboard, http://localhost:5173
```

## Running on Sol (Apptainer + Voyager)

Everything runs on one interactive GPU node. Postgres and Neo4j run as Apptainer containers,
and the LLM is Voyager (`LLM_PROVIDER=openai`, no Ollama).

```bash
# 0. once, on a login node: env, images, data
git clone/rsync the repo to $SCRATCH/swm && cd $SCRATCH/swm/CODE
module load mamba && mamba create -n swm python=3.11 -y && source activate swm && pip install -r requirements.txt
mkdir -p $SCRATCH/svc && cd $SCRATCH/svc
apptainer pull postgres.sif docker://postgres:16
apptainer pull neo4j.sif docker://neo4j:5-community
cd $SCRATCH/swm/CODE
cp .env.example .env     # set LLM_PROVIDER=openai, LLM_API_KEY, LLM_MODEL, POSTGRES_PORT=5433

# 1. interactive node
interactive -p public -q public -G a100:1 -c 8 --mem=48G -t 0-08:00   # adjust to your Sol allocation
source activate swm; cd $SCRATCH/swm/CODE; set -a; source .env; set +a

# 2. services (background, this node)
PG=$SCRATCH/svc/pg; mkdir -p $PG/data $PG/run
apptainer exec -B $PG:$PG $SCRATCH/svc/postgres.sif /usr/lib/postgresql/16/bin/initdb -D $PG/data -U swm --auth=trust
apptainer exec -B $PG:$PG $SCRATCH/svc/postgres.sif /usr/lib/postgresql/16/bin/pg_ctl -D $PG/data -l $PG/log \
  -o "-p 5433 -k $PG/run -c listen_addresses=127.0.0.1" start
apptainer exec -B $PG:$PG $SCRATCH/svc/postgres.sif /usr/lib/postgresql/16/bin/createdb -h 127.0.0.1 -p 5433 -U swm swm
N=$SCRATCH/svc/neo4j; mkdir -p $N/data $N/logs
nohup apptainer run --env NEO4J_AUTH=neo4j/changeme -B $N/data:/data -B $N/logs:/logs $SCRATCH/svc/neo4j.sif > $N/out.log 2>&1 &

# 3. pipeline
python scripts/01_ingest.py --archive ../DATA/20newsgroups/20news-bydate.tar.gz
python scripts/02_embed.py && python scripts/03_cluster.py
python scripts/04_explain.py
python scripts/05_kg.py --workers 8          # Voyager; 30 posts/cluster, cached and resumable
python scripts/06_eval.py export --n-plaus 100 --n-triples 200
python scripts/06_eval.py run --workers 8 [--judges modelA,modelB]   # >=2 judges -> Krippendorff alpha
# report: ../EVALUATIONS/phase1/phase1-*.md
```

`SWM_RUN_ID=bge-base-en-v1.5__agglomerative` pins the presented run (faithfulness needs an SBERT/BGE
run, not tfidf_lsa). Judgments are cached in `data/cache/eval/`; rerunning resumes.

Chunked embeddings can also be computed on a Sol GPU without a database: run
`sbatch scripts/sol/embed.sbatch`, then copy `data/cache/*.npy` and `*.meta.json` back to the laptop.
