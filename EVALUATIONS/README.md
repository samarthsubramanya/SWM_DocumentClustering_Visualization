# Evaluations

Every evaluation output from our runs, grouped by what it measures. The scripts in `CODE/` write
new reports into these same folders. The headline numbers are below, and each file holds the details.

| Folder | What it measures | Written by |
|---|---|---|
| `clustering/` | Clustering stability over 3 seeds; choosing k without labels | `scripts/03_cluster.py --seeds`, `scripts/03b_k_select.py` |
| `phase1/` | Final clustering run, faithfulness of explanations, KG conformance, LLM-judge ratings | `scripts/06_eval.py run` |
| `knowledge_graph/` | KG extraction quality per LLM (Qwen3-30B vs Llama 4 Scout) | `scripts/05_kg.py --extract-only` |
| `text2cypher/` | Natural-language to Cypher: validity before and after repair, execution accuracy | `scripts/06_eval.py text2cypher` |
| `human_ratings/` | Team ratings of explanations, cluster labels and triples | `scripts/07_human_eval.py score` |

## Headline results

**Clustering** (`clustering/clustering-seeds-*.json`, 17,090 posts, NMI mean over seeds 42–44)

| Encoder | k-means | Agglomerative | HDBSCAN |
|---|---|---|---|
| TF-IDF + LSA | 0.452 | 0.454 | 0.402 |
| all-mpnet-base-v2 | 0.641 | 0.639 | 0.574 |
| bge-base-en-v1.5 | 0.660 | **0.662** | 0.598 |

Every transformer run clears the NMI ≥ 0.55 target; the encoder matters more than the algorithm.
Without labels, the k-means elbow picks k = 20, while silhouette prefers 10–12 (`clustering/k-select-*.json`).

**Phase 1** (`phase1/`, final run `bge-base-en-v1.5__agglomerative`)

| Metric | Result | Target |
|---|---|---|
| NMI | 0.663 | ≥ 0.55 |
| Faithfulness: assignment flips, cited vs random terms (`phase1-20261108-152018`) | 10.5× among posts nearest their own center; 1.7× over all posts | ≥ 2× |
| Ontology conformance of extracted triples | 0.824 | ≥ 0.95 |
| Plausibility, LLM judges (`phase1-20261108-145651`) | 4.27 / 5, α = 0.19 | ≥ 3.5, α ≥ 0.6 |
| KG label preferred over c-TF-IDF label, LLM judges | 0.475 | ≥ 0.60 |
| Triple precision, LLM judges | 0.61 (0.89 vs 0.31 by judge) | ≥ 0.75 |

The two LLM judges disagree too much to replace human raters, so these three rows are re-scored
by people in `human_ratings/`. `phase1-20261108-150947` is an intermediate run of the same evaluation.

**Knowledge-graph extraction** (`knowledge_graph/`, the same 576 posts)

| Model | Entities | Relations | Ontology conformance | Failed posts |
|---|---|---|---|---|
| qwen3-30b-a3b (used) | 7,050 | 3,437 | 0.824 | 23 |
| llama4-scout-17b | 5,028 | 1,911 | 0.838 | 0 |

**Text2Cypher** (`text2cypher/`; this answers the mentor's feedback on hallucinated Cypher)

| Report | Valid on first try | Valid after repair | Execution accuracy |
|---|---|---|---|
| `text2cypher-20261108-152910` (BGE graph, ontology v4) | 85% | 100% | 0.833 (15/18) |
| Old prompt, main set (cloud graph) | 95% | 100% | 0.643 (9/14) |
| **New prompt, main set (cloud graph)** | **100%** | **100%** | **1.000 (14/14)** |
| Old prompt, held-out set | 100% | 100% | 0.778 (7/9) |
| **New prompt, held-out set** | **100%** | **100%** | **1.000 (9/9)** |

`text2cypher/text2cypher-20261115-121500-prompt-ab.md` explains the old-vs-new comparison. The
held-out questions were never used to tune the prompt. The samples are small, so read 9/9 as
"no errors seen", not as perfect accuracy.
