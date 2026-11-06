#!/usr/bin/env python3
"""Human evaluation (slide 8 / proposal VII): blind rating sheets from the 06_eval bundle, then scores.

  python scripts/07_human_eval.py make --raters samarth mohak asmi pratik
      -> eval/human/<rater>/plausibility.csv, labels.csv  (every rater)
         eval/human/<rater>/triples.csv                   (the first two raters: double annotation)
         eval/human/key.json                              (which label is which; do not open while rating)
  ...each rater fills the empty column in their CSVs (Excel / Sheets, keep the file as CSV)...
  python scripts/07_human_eval.py score
      -> eval/reports/human-<time>.json / .md: plausibility mean + Krippendorff alpha (ordinal),
         KG-vs-c-TF-IDF label preference, triple precision + Cohen's kappa

Blinding: item order is shuffled per rater, labels are shown as A/B in a random order per rater,
and sheets carry no run, model or method names.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from swm import evaluate as ev  # noqa: E402

BUNDLE = ROOT / "data" / "eval" / "bundle" / "bundle.json"
HUMAN = ROOT / "eval" / "human"
REPORTS = ROOT / "eval" / "reports"
SHEETS = {
    "plausibility": ["item", "post", "cluster_keywords", "explanation", "score_1_to_5", "comment"],
    "labels": ["item", "posts_from_cluster", "label_A", "label_B", "better_A_or_B", "comment"],
    "triples": ["item", "document", "source", "source_type", "relation", "target", "target_type",
                "correct_1_or_0", "comment"],
}
INSTRUCTIONS = """Rating instructions (read once before starting)

plausibility.csv  Score each explanation 1-5 against the post itself:
  1 wrong or misleading, 2 weak or mostly generic, 3 acceptable but vague,
  4 good, specific and accurate, 5 excellent, specific, accurate and clearly grounded in the post.
labels.csv        Which label better describes what the posts have in common (accurate, specific,
                  readable)? Write A or B.
triples.csv       1 if the document states or clearly implies the relation between the two entities
                  AND both entity types are right; otherwise 0.

Rate alone, do not discuss items with others until everyone is done. Leave 'comment' empty unless useful.
"""


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel opens it as UTF-8
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def make(args) -> None:
    b = json.loads(args.bundle.read_text())
    key = {"raters": args.raters, "bundle_run": b["run_id"], "labels": {}, "triple_raters": args.raters[:2]}
    plaus = b["plausibility_items"][: args.n_plaus]
    clusters = [c for c in b["clusters"] if c.get("kg_label") and c.get("ctfidf_label")
                and c["kg_label"] != c["ctfidf_label"]]
    triples = b["triples"][: args.n_triples]

    for r_i, rater in enumerate(args.raters):
        rng = random.Random(f"{args.seed}-{rater}")
        rows = [[f"P{i:03d}", " ".join(it["text"].split())[:900], ", ".join(it["cluster_keywords"][:8]), it["rationale"], "", ""]
                for i, it in enumerate(plaus)]
        rng.shuffle(rows)
        write_csv(HUMAN / rater / "plausibility.csv", SHEETS["plausibility"], rows)

        rows = []
        for c in clusters:
            kg_first = rng.random() < 0.5
            a, bb = (c["kg_label"], c["ctfidf_label"]) if kg_first else (c["ctfidf_label"], c["kg_label"])
            item = f"L{c['cluster_id']:03d}"
            key["labels"].setdefault(rater, {})[item] = "A" if kg_first else "B"   # where the KG label sits
            rows.append([item, "\n".join(f"- {s}" for s in c["samples"]), a, bb, "", ""])
        rng.shuffle(rows)
        write_csv(HUMAN / rater / "labels.csv", SHEETS["labels"], rows)

        if r_i < 2:
            rows = [[f"T{i:03d}", " ".join(t["text"].split())[:1500], t["source"], t["source_type"], t["relation"],
                     t["target"], t["target_type"], "", ""] for i, t in enumerate(triples)]
            rng.shuffle(rows)
            write_csv(HUMAN / rater / "triples.csv", SHEETS["triples"], rows)
        (HUMAN / rater / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS)

    (HUMAN / "key.json").write_text(json.dumps(key, indent=1))
    print(f"Sheets for {', '.join(args.raters)} in {HUMAN}: {len(plaus)} explanations, {len(clusters)} label pairs, "
          f"{len(triples)} triples (raters {', '.join(args.raters[:2])}).")


def _num(v: str, lo: int, hi: int):
    words = {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0}
    if hi == 1 and str(v).strip().lower() in words:
        return words[str(v).strip().lower()]
    try:
        x = int(float(v.strip()))
    except (ValueError, AttributeError):
        return None
    return x if lo <= x <= hi else None


def score(args) -> None:
    from sklearn.metrics import cohen_kappa_score

    key = json.loads((HUMAN / "key.json").read_text())
    raters = [r for r in key["raters"] if (HUMAN / r).exists()]
    report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": key["bundle_run"], "raters": raters}

    # plausibility: items x raters
    by_rater = {r: {row["item"]: _num(row["score_1_to_5"], 1, 5) for row in read_csv(HUMAN / r / "plausibility.csv")}
                for r in raters}
    items = sorted(set().union(*by_rater.values()))
    matrix = [[by_rater[r].get(i) for i in items] for r in raters]
    flat = [x for row in matrix for x in row if x is not None]
    report["plausibility"] = {
        "n_items": len(items), "mean": float(np.mean(flat)) if flat else None,
        "mean_by_rater": {r: float(np.mean([x for x in row if x is not None] or [np.nan])) for r, row in zip(raters, matrix)},
        "krippendorff_alpha_ordinal": ev.krippendorff_alpha(matrix),
        "missing": sum(x is None for row in matrix for x in row),
    }

    # labels: KG label chosen?
    wins, per = [], {}
    for r in raters:
        k = key["labels"].get(r, {})
        picks = [row["better_A_or_B"].strip().upper()[:1] == k[row["item"]]
                 for row in read_csv(HUMAN / r / "labels.csv") if row["better_A_or_B"].strip().upper()[:1] in ("A", "B")]
        per[r] = float(np.mean(picks)) if picks else None
        wins += picks
    report["labels"] = {"kg_preferred": float(np.mean(wins)) if wins else None, "n_judgments": len(wins),
                        "kg_preferred_by_rater": per}

    # triples: precision + Cohen's kappa between the two annotators
    t = {}
    for r in key["triple_raters"]:
        if (HUMAN / r / "triples.csv").exists():
            t[r] = {row["item"]: _num(row["correct_1_or_0"], 0, 1) for row in read_csv(HUMAN / r / "triples.csv")}
    if t:
        all_votes = [v for d in t.values() for v in d.values() if v is not None]
        report["triples"] = {"precision": float(np.mean(all_votes)) if all_votes else None,
                             "precision_by_rater": {r: float(np.mean([v for v in d.values() if v is not None] or [np.nan]))
                                                    for r, d in t.items()}}
        if len(t) == 2:
            a, b = t.values()
            both = [i for i in a if a[i] is not None and b.get(i) is not None]
            report["triples"] |= {"n_double_annotated": len(both),
                                  "cohen_kappa": float(cohen_kappa_score([a[i] for i in both], [b[i] for i in both]))
                                  if both else None}

    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"human-{time.strftime('%Y%m%d-%H%M%S')}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
    p, lab, tr = report["plausibility"], report["labels"], report.get("triples", {})
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    md = (f"# Human evaluation ({', '.join(raters)})\n\n| metric | value | target |\n|---|---|---|\n"
          f"| plausibility mean (1-5) | {f(p['mean'])} | >= 3.5 |\n"
          f"| plausibility Krippendorff alpha | {f(p['krippendorff_alpha_ordinal'])} | >= 0.6 |\n"
          f"| KG label preferred over c-TF-IDF | {f(lab['kg_preferred'])} | >= 0.6 |\n"
          f"| triple precision | {f(tr.get('precision'))} | >= 0.75 |\n"
          f"| triple Cohen's kappa | {f(tr.get('cohen_kappa'))} | (report) |\n")
    stem.with_suffix(".md").write_text(md)
    print(md + f"\nReport: {stem}.json / .md")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="write blind rating sheets from the 06_eval bundle")
    m.add_argument("--raters", nargs="+", required=True)
    m.add_argument("--bundle", type=Path, default=BUNDLE)
    m.add_argument("--n-plaus", type=int, default=60)
    m.add_argument("--n-triples", type=int, default=100)
    m.add_argument("--seed", type=int, default=7)
    m.set_defaults(fn=make)
    s = sub.add_parser("score", help="read the filled sheets and compute the human metrics")
    s.set_defaults(fn=score)
    a = p.parse_args()
    a.fn(a)
