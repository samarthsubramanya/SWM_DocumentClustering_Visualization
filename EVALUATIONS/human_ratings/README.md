# Human ratings

Ratings by the four team members, replacing the LLM judges whose agreement was too low (α = 0.19):

- **Plausibility:** each explanation scored 1–5; agreement measured with Krippendorff's α.
- **Cluster labels:** blind A/B choice between the KG (PageRank) label and the c-TF-IDF label.
- **Triple precision:** whether each extracted triple is supported by its post, scored by two raters (Cohen's κ).

The rating sheets come from `CODE/scripts/07_human_eval.py make`, and `CODE/scripts/07_human_eval.py score` writes
`human-<time>.json` / `.md` into this folder. Put the completed rating sheets here as well.
