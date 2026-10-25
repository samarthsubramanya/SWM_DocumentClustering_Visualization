#!/usr/bin/env python3
"""Phase 1 evaluation (PLAN.md section 5), split like scripts/05_kg.py so the LLM part can run on Sol
without Postgres / Neo4j:

  local:  python scripts/06_eval.py export                  # DB + caches -> data/eval/bundle/
          rsync -a data/eval/bundle <sol>:<repo>/data/eval/
  Sol:    sbatch scripts/sol/eval_judge.sbatch              # python scripts/06_eval.py run
  local:  rsync -a <sol>:<repo>/eval/reports/ eval/reports/

`run` computes faithfulness (re-embeds with the run's SBERT model) and the LLM-judge metrics (Voyager /
any OpenAI-compatible endpoint, one or several judge models) and writes eval/reports/phase1-*.{json,md}.
Judgments are cached under data/cache/eval/, so an interrupted run resumes.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from swm import evaluate as ev  # noqa: E402

DEFAULT_BUNDLE = ROOT / "data" / "eval" / "bundle"
REPORTS = ROOT / "eval" / "reports"
PARTS = ("faithfulness", "plausibility", "triples", "labels")


def export(args) -> None:
    from tqdm import tqdm

    from swm import db, explain, kg, represent

    run_id = args.run or db.default_run()
    runs = db.query("SELECT run_id, algo, embedding, metrics FROM runs ORDER BY run_id")
    embedding = next(r["embedding"] for r in runs if r["run_id"] == run_id)
    rows = db.query(
        """SELECT dc.doc_id, dc.cluster_id, dc.is_noise, d.text, e.evidence
           FROM doc_cluster dc JOIN documents d ON d.id = dc.doc_id
           JOIN explanations e ON e.run_id = dc.run_id AND e.doc_id = dc.doc_id
           WHERE dc.run_id = %s ORDER BY dc.doc_id""", (run_id,))
    if not rows:
        raise SystemExit(f"No evidence for {run_id}; run scripts/04_explain.py --run {run_id} first.")
    info = {r["cluster_id"]: r for r in db.query(
        "SELECT cluster_id, keywords, label FROM cluster_info WHERE run_id = %s", (run_id,))}
    bundle = {"run_id": run_id, "embedding": embedding, "seed": args.seed, "n_docs": len(rows),
              "runs": [{k: r[k] for k in ("run_id", "algo", "embedding", "metrics")} for r in runs]}
    args.out.mkdir(parents=True, exist_ok=True)

    # --- faithfulness: sampled non-noise docs with cited terms visible to the encoder + centroids
    if embedding in represent.SBERT_MODELS:
        ids, X = represent.load(embedding)
        pos = {d: i for i, d in enumerate(ids)}
        order = [pos[r["doc_id"]] for r in rows]
        cids, C = ev.cluster_centroids(X[order], [r["cluster_id"] for r in rows], [r["is_noise"] for r in rows])
        np.save(args.out / "centroids.npy", C)
        np.save(args.out / "centroid_ids.npy", cids)
        usable = []
        for r in rows:
            cited = [] if r["is_noise"] else ev.cited_terms(r["evidence"]["term_hits"], r["text"])
            if cited:
                usable.append({"doc_id": r["doc_id"], "cluster_id": r["cluster_id"],
                               "text": r["text"][:ev.TEXT_CAP], "cited": cited})
        bundle["faithfulness_items"] = ev.sample(usable, args.n_faith, args.seed)
    else:
        print(f"WARNING: {embedding} is not an SBERT model; faithfulness needs one to re-embed. Skipped.")
        bundle["faithfulness_items"] = []

    # --- plausibility: LLM rationales (cached by 04_explain, generated here when missing)
    by_id = {r["doc_id"]: r for r in rows}
    items = []
    for doc_id in tqdm(ev.sample(by_id, args.n_plaus, args.seed), desc="rationales"):
        r = by_id[doc_id]
        text = explain.rationale(doc_id, run_id)
        e = r["evidence"]
        items.append({"doc_id": doc_id, "text": r["text"], "cluster_keywords": e["cluster_keywords"],
                      "term_hits": e["term_hits"], "margin": e["margin"], "rationale": text,
                      "rationale_source": "llm" if explain.cached_rationale(doc_id, run_id) else "template"})
    bundle["plausibility_items"] = items

    # --- KG: conformance + duplicate rate over the extraction cache, sampled triples to judge
    onto = kg.load_ontology()
    reps = kg.representative_docs(run_id, args.docs_per_cluster)
    docs = [{"id": r["doc_id"], "text": r["text"]} for r in reps]
    entities, relations, stats = kg.cached_extract(docs, onto)
    kept = stats.get("entities", 0) + stats.get("relations", 0)
    entities_n, _ = kg.normalize(entities, relations)
    text_of = {d["id"]: d["text"] for d in docs}
    bundle["kg"] = {"docs": len(docs), "stats": stats, "ontology_conformance": kept / max(kept + stats.get("rejected", 0), 1),
                    "duplicate_rate": ev.duplicate_rate(entities_n), "n_relations": len(relations)}
    if stats.get("unreachable_docs"):
        print(f"WARNING: {stats['unreachable_docs']} of {len(docs)} docs have no cached extraction (run 05_kg.py).")
    bundle["triples"] = [{**t, "text": text_of[t["doc_id"]]} for t in ev.sample(relations, args.n_triples, args.seed)]

    # --- label preference: KG PageRank label vs c-TF-IDF top terms, 5 central docs per cluster
    samples = {}
    for r in kg.representative_docs(run_id, 5):
        samples.setdefault(r["cluster_id"], []).append(" ".join(r["text"].split())[:400])
    bundle["clusters"] = [
        {"cluster_id": c, "kg_label": i["label"], "ctfidf_label": ", ".join(t for t, _ in i["keywords"][:3]),
         "samples": samples.get(c, [])}
        for c, i in sorted(info.items()) if i["label"] and samples.get(c)]
    if not bundle["clusters"]:
        print("WARNING: no KG cluster labels (run 05_kg.py); label preference skipped.")

    (args.out / "bundle.json").write_text(json.dumps(bundle, indent=1, default=str))
    print(f"Bundle for {run_id} ({embedding}) in {args.out}: {len(bundle['faithfulness_items'])} faithfulness docs, "
          f"{len(items)} explanations, {len(bundle['triples'])} triples, {len(bundle['clusters'])} clusters.")


def run(args) -> None:
    from swm import kg
    from swm.llm import BadOutput, structured

    bundle = json.loads((args.bundle / "bundle.json").read_text())
    judges = [m.strip() for m in (args.judges or os.getenv("LLM_MODEL", "")).split(",") if m.strip()]
    only = set(args.only or PARTS)
    report = {"run_id": bundle["run_id"], "embedding": bundle["embedding"], "judges": judges,
              "judge_is_proxy_for_human_raters": True, "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    # clustering table straight from the runs recorded at export time
    report["clustering"] = {r["run_id"]: r["metrics"] for r in bundle["runs"]}
    report["best_nmi"] = report["clustering"].get(bundle["run_id"], {}).get("nmi")  # the evaluated run, not the best of the grid
    report["kg"] = bundle["kg"]

    if only & {"plausibility", "triples", "labels"}:
        if not judges:
            raise SystemExit("No judge model: set LLM_MODEL or pass --judges a,b (LLM_PROVIDER=openai for Voyager).")
        for m in judges:  # fail fast (ConfigError / unreachable) on a bad key or model id before the long calls
            try:
                structured("Reply with a JSON object.", ev.LabelChoice, attempts=1, model=m)
            except BadOutput:
                pass  # the model answered, which is all this probes

    if "faithfulness" in only and bundle["faithfulness_items"]:
        from swm.represent import encode_texts

        C = np.load(args.bundle / "centroids.npy")
        cids = np.load(args.bundle / "centroid_ids.npy")
        report["faithfulness"] = ev.faithfulness(
            bundle["faithfulness_items"], cids, C, lambda t: encode_texts(t, bundle["embedding"]),
            draws=args.random_draws, seed=bundle["seed"])

    if "plausibility" in only:
        items = bundle["plausibility_items"]
        by_judge = {m: ev.judge_all(m, "plausibility", [ev.plausibility_prompt(i) for i in items],
                                    ev.Plausibility, args.workers) for m in judges}
        report["plausibility"] = ev.summarize_plausibility(by_judge)
        report["plausibility"]["rationales_from_llm"] = float(np.mean([i["rationale_source"] == "llm" for i in items]))
        report["plausibility"]["items"] = [
            {"doc_id": i["doc_id"], "rationale": i["rationale"],
             "scores": {m: (by_judge[m][k] or {}).get("score") for m in judges},
             "reasons": {m: (by_judge[m][k] or {}).get("reason") for m in judges}}
            for k, i in enumerate(items)]

    if "triples" in only and bundle["triples"]:
        onto = kg.load_ontology()
        by_judge = {m: ev.judge_all(m, "triple", [ev.triple_prompt(t, onto) for t in bundle["triples"]],
                                    ev.TripleCheck, args.workers) for m in judges}
        report["triples"] = ev.summarize_triples(by_judge)
        report["triples"]["items"] = [
            {k2: t[k2] for k2 in ("doc_id", "source", "relation", "target")}
            | {"correct": {m: (by_judge[m][k] or {}).get("correct") for m in judges}}
            for k, t in enumerate(bundle["triples"])]

    if "labels" in only and bundle["clusters"]:
        wins, per = [], {}
        for m in judges:
            jobs = [(c["cluster_id"], p, first) for c in bundle["clusters"] for p, first in ev.label_prompts(c)]
            outs = ev.judge_all(m, "label", [p for _, p, _ in jobs], ev.LabelChoice, args.workers)
            per[m] = [(o["better"] == "A") == (first == "kg") for (_, _, first), o in zip(jobs, outs) if o]
            wins += per[m]
        report["labels"] = ev.summarize_labels(wins, len(bundle["clusters"]), per)

    report["targets"] = ev.check_targets(report)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"phase1-{time.strftime('%Y%m%d-%H%M%S')}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    stem.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))
    print(f"Report: {stem}.json / .md")


def markdown(r: dict) -> str:
    def f(v):
        return "n/a" if v is None else f"{v:.3f}"

    out = [f"# Phase 1 evaluation: {r['run_id']}", "",
           f"Judges (LLM stand-in for human raters): {', '.join(r['judges']) or 'none'}", "",
           "## Targets (PLAN.md section 5)", "", "| metric | value | target | |", "|---|---|---|---|"]
    out += [f"| {k} | {f(t['value'])} | >= {t['target']} | {'n/a' if t['pass'] is None else 'PASS' if t['pass'] else 'FAIL'} |"
            for k, t in r["targets"].items()]
    out += ["", "## Clustering runs", "", "| run | NMI | ARI | purity | silhouette | noise |", "|---|---|---|---|---|---|"]
    out += [f"| {k} | {f(m.get('nmi'))} | {f(m.get('ari'))} | {f(m.get('purity'))} | {f(m.get('silhouette'))} | "
            f"{f(m.get('noise_rate'))} |" for k, m in r["clustering"].items()]
    if fa := r.get("faithfulness"):
        out += ["", "## Faithfulness (delete cited terms vs equal number of random terms)", "",
                f"- docs: {fa['n']}", f"- centroid-margin drop: cited {f(fa['margin_drop_cited'])} vs random "
                f"{f(fa['margin_drop_random'])} (ratio {f(fa['margin_ratio'])})",
                f"- own-centroid cosine drop: cited {f(fa['sim_drop_cited'])} vs random {f(fa['sim_drop_random'])} "
                f"(ratio {f(fa['sim_ratio'])})", f"- docs where cited hurts more: {f(fa['frac_cited_hurts_more'])}",
                f"- assignment flips (margin < 0): cited {f(fa['assignment_flips_cited'])} vs random "
                f"{f(fa['assignment_flips_random'])}"]
    if p := r.get("plausibility"):
        out += ["", "## Explanation plausibility (LLM judges, 1-5)", "", f"- mean {f(p['mean'])} over {p['n_items']} "
                f"explanations; Krippendorff alpha {f(p['krippendorff_alpha'])}; LLM-written rationales "
                f"{f(p['rationales_from_llm'])}", f"- by judge: {p['mean_by_judge']}"]
    if t := r.get("triples"):
        out += ["", "## KG triple precision", "", f"- precision {f(t['precision'])} over {t['n_triples']} triples; "
                f"by judge {t['precision_by_judge']}"]
    k = r["kg"]
    out += ["", "## KG extraction", "", f"- ontology conformance {f(k['ontology_conformance'])}; entity duplicate rate "
            f"{f(k['duplicate_rate'])}; {k['n_relations']} relations from {k['docs']} docs; stats {k['stats']}"]
    if lab := r.get("labels"):
        out += ["", "## Cluster labels: KG (PageRank) vs c-TF-IDF", "", f"- KG label preferred {f(lab['kg_preferred'])} "
                f"over {lab['n_clusters']} clusters (both presentation orders); by judge {lab['kg_preferred_by_judge']}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="DB + caches -> bundle (needs Postgres; run 04_explain and 05_kg first)")
    e.add_argument("--run", help="run_id (default: SWM_RUN_ID or best NMI)")
    e.add_argument("--out", type=Path, default=DEFAULT_BUNDLE)
    e.add_argument("--n-faith", type=int, default=200)
    e.add_argument("--n-plaus", type=int, default=100, help="explanations to judge (PLAN: 100)")
    e.add_argument("--n-triples", type=int, default=200, help="triples to judge (PLAN: 200)")
    e.add_argument("--docs-per-cluster", type=int, default=30, help="must match the 05_kg.py run")
    e.add_argument("--seed", type=int, default=42)
    e.set_defaults(fn=export)

    r = sub.add_parser("run", help="faithfulness + LLM judging from a bundle (no Postgres / Neo4j)")
    r.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    r.add_argument("--out", type=Path, default=REPORTS)
    r.add_argument("--judges", help="comma-separated judge model ids (default: LLM_MODEL); >=2 enables alpha")
    r.add_argument("--only", nargs="+", choices=PARTS, help="subset of evaluations")
    r.add_argument("--workers", type=int, default=8, help="concurrent judge requests")
    r.add_argument("--random-draws", type=int, default=3, help="random-deletion draws per doc")
    r.set_defaults(fn=run)

    args = p.parse_args()
    args.fn(args)
