# Dynamic ingestion: all-mpnet-base-v2__agglomerative

6785 test posts streamed into a run clustered on 10305 train posts (nearest cluster center, `all-mpnet-base-v2` embeddings).

| Metric | Result | Target |
|---|---|---|
| Agreement with batch clustering | 0.767 | ≥ 0.85 |
| Latency per post, p95 (p50) | 0.099 s (0.023 s) | < 3 s |
| NMI of streamed posts vs newsgroups | 0.603 (run on train: 0.654) | ≥ 0.55 |
| ARI / purity of streamed posts | 0.465 / 0.604 | |
| Flagged as emerging | 3.5% (threshold: similarity < 0.180) | |

The batch reference re-clusters train and test together with Ward (k = 20) on the cached 10-d UMAP. On the train posts it agrees with the stored run 0.848 of the time, which bounds how high streaming agreement can go.

Latency was measured one post at a time on 200 random test posts (max 0.277 s). The scores use the cached test-post vectors; fresh encodings of 100 posts matched them with cosine ≥ 1.0000.

## Lowest-similarity posts (emerging candidates)

| Post | Nearest cluster | Similarity | Newsgroup | Opening |
|---|---|---|---|---|
| `doc-fd65f089` | 18 | 0.035 | sci.med | I was just wondering if exercises specific to particular regions of the body (such as thig |
| `doc-50c7c1e4` | 8 | 0.041 | sci.med | Do you swim nude in a public swimming pool? :) I doubt sperm can penetrate swimsuit materi |
| `doc-1c40b4b5` | 1 | 0.044 | sci.electronics | <lots of stuff deleted> Might depend on where you live .. I know locally, for most of the  |
| `doc-a2ab0192` | 3 | 0.046 | alt.atheism | Yes, unless the observer is at rest with respect to the singularity at infinite distance a |
| `doc-959550f6` | 2 | 0.058 | sci.med | When I have trouble it's usually because of water trapped by some remaining wax. I don't s |
| `doc-d1cc4cc1` | 17 | 0.060 | sci.electronics | I hope David isn't going to be too upset with me for sticking my nose in here again, but h |
| `doc-764e1680` | 11 | 0.061 | sci.med | There are two different mechanisms here: toning of muscles and reduction of fat. Exercises |
| `doc-65d2d4a7` | 19 | 0.062 | sci.space | Well, there are several bursts in species diversity I can think of. The Cambrian and Ordov |
