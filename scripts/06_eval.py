#!/usr/bin/env python3
"""Evaluation (PLAN.md section 5). Writes reports to eval/reports/.

  --text2cypher   KGAgent query translation: validity and execution accuracy on
                  eval/text2cypher.yaml (needs Neo4j loaded by 05_kg.py and Ollama).

TODO: clustering metrics table, explanation faithfulness test, KG triple precision.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import yaml  # noqa: E402

from swm import kg  # noqa: E402

REPORTS = ROOT / "eval" / "reports"


def eval_text2cypher(questions_path: Path) -> dict:
    from src.agents.kg_agent import MAX_ROWS, text2cypher

    questions = yaml.safe_load(questions_path.read_text())["questions"]
    results = []
    for q in questions:
        gold = kg.read_query(q["gold"], MAX_ROWS + 1)
        t = time.time()
        out = asyncio.run(text2cypher(q["question"]))
        valid = out["rows"] is not None
        results.append({
            "id": q["id"], "pattern": q["pattern"], "question": q["question"],
            "gold_cypher": q["gold"].strip(), "cypher": out["cypher"],
            "valid": valid,
            "valid_first_try": valid and out["attempts"] == 1,
            "attempts": out["attempts"],
            "correct": valid and kg.same_results(out["rows"], gold),
            "gold_empty": not gold,               # empty == empty proves little
            "gold_truncated": len(gold) > MAX_ROWS,
            "problems": out["problems"],
            "seconds": round(time.time() - t, 1),
        })
        r = results[-1]
        print(f"{r['id']} {'OK ' if r['correct'] else 'BAD' if r['valid'] else 'INV'} "
              f"attempts={r['attempts']} {r['question']}")

    def rate(rows, field):
        return round(sum(r[field] for r in rows) / max(len(rows), 1), 3)

    informative = [r for r in results if not r["gold_empty"] and not r["gold_truncated"]]
    by_pattern = defaultdict(list)
    for r in informative:
        by_pattern[r["pattern"]].append(r)
    return {
        "n_questions": len(results),
        "n_informative": len(informative),
        "validity": rate(results, "valid"),                     # some attempt passed validation + ran
        "validity_first_try": rate(results, "valid_first_try"),  # before any repair
        "execution_accuracy": rate(informative, "correct"),     # same rows as the gold query
        "accuracy_by_pattern": {p: rate(rs, "correct") for p, rs in sorted(by_pattern.items())},
        "excluded": [r["id"] for r in results if r["gold_empty"] or r["gold_truncated"]],
        "results": results,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text2cypher", action="store_true", help="evaluate the KGAgent")
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "text2cypher.yaml")
    args = p.parse_args()
    if not args.text2cypher:
        p.error("choose an evaluation, e.g. --text2cypher")

    report = eval_text2cypher(args.questions)
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"text2cypher-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    summary = {k: v for k, v in report.items() if k != "results"}
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")
