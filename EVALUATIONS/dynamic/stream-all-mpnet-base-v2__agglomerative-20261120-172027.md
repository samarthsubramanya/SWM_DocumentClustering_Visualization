# Dynamic ingestion: all-mpnet-base-v2__agglomerative

6785 test posts streamed into a run clustered on 10305 train posts (15-nearest-neighbor vote, `all-mpnet-base-v2` embeddings).

| Metric | Result | Target |
|---|---|---|
| Agreement with batch clustering | 0.829 | ≥ 0.85 |
| Latency per post, p95 (p50) | 0.039 s (0.021 s) | < 3 s |
| NMI of streamed posts vs newsgroups | 0.621 (run on train: 0.654) | ≥ 0.55 |
| ARI / purity of streamed posts | 0.467 / 0.606 | |
| Flagged as emerging | 3.5% (threshold: similarity < 0.180) | |

The batch reference re-clusters train and test together with Ward (k = 20) on the cached 10-d UMAP. On the train posts it agrees with the stored run 0.848 of the time, which bounds how high streaming agreement can go.

Latency was measured one post at a time on 200 random test posts (max 0.065 s). The scores use the cached test-post vectors; fresh encodings of 100 posts matched them with cosine ≥ 1.0000.

## Lowest-similarity posts (emerging candidates)

| Post | Nearest cluster | Similarity | Newsgroup | Opening |
|---|---|---|---|---|
| `doc-06d70d1a` | 1 | -0.039 | sci.space | Yes, weightlessness does feel like falling. It may feel strange at first, but the body doe |
| `doc-23cd4b1c` | 2 | -0.029 | comp.sys.mac.hardware | I vote for turning them off at night. THe stress of off/on is a little harmful, so don't t |
| `doc-76eda639` | 1 | -0.018 | sci.med | The best thing to do is leave it, it will work its own way out to the surface. Anything yo |
| `doc-46c358ab` | 3 | -0.018 | comp.graphics | A map of Italy showing the states/provinces(?) is in the FreeHand 3.1 for Windows clip art |
| `doc-43cadbba` | 15 | -0.017 | sci.med | I think I can explain the "missing part of a leaf" story. I have actually seen a reproduct |
| `doc-3abf6a95` | 15 | -0.016 | sci.electronics | You can design for ramp shutoff, brick-wall current limit or even fold-back cutoff....soun |
| `doc-9201fd06` | 1 | -0.011 | sci.electronics | What I usually use...and this stuff is only good on glass....is either Acetone, or a littl |
| `doc-a9eb9c1d` | 1 | -0.008 | sci.electronics | If you want to try a non-toxic solvent, there's one made of citrus that works very well an |
