"""Phase 1 evaluation (PLAN.md section 5): explanation faithfulness plus LLM-judge stand-ins for
the human studies (explanation plausibility, KG triple precision, cluster-label preference).

Pure logic, no Postgres / Neo4j: scripts/06_eval.py exports a bundle from the DB, and `run`
consumes it anywhere (e.g. Sol). Judgments are cached per (model, prompt), so reruns resume.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

from swm.cluster import unit
from swm.explain import STOP_WORDS, TOKEN_PATTERN
from swm.represent import CACHE

TEXT_CAP = 2500  # chars; roughly what the SBERT encoders see (mpnet 384 tokens), so edits beyond it are invisible
TARGETS = {  # PLAN.md section 5
    "nmi": 0.55, "faithfulness_ratio": 2.0, "plausibility_mean": 3.5, "plausibility_alpha": 0.6,
    "kg_label_preferred": 0.6, "triple_precision": 0.75, "ontology_conformance": 0.95,
}


def sample(items, n: int, seed: int = 42) -> list:
    items = list(items)
    return random.Random(seed).sample(items, min(n, len(items)))


# ==========================================================
# Faithfulness: does deleting the cited evidence hurt the assignment more than deleting random terms?
# ==========================================================

def cluster_centroids(X, assigned, is_noise) -> tuple[np.ndarray, np.ndarray]:
    """(cluster ids, unit centroids over non-noise members); same math as explain.compute_evidence."""
    Xn, assigned = unit(np.asarray(X, dtype=np.float32)), np.asarray(assigned)
    core = ~np.asarray(is_noise, dtype=bool)
    cids = np.unique(assigned)
    return cids, unit(np.stack([Xn[(assigned == c) & core].mean(axis=0) for c in cids]))


def cited_terms(term_hits: list[dict], text: str, k: int = 3) -> list[str]:
    """Top-k cluster terms the explanation can cite that are visible to the encoder."""
    head = text[:TEXT_CAP]
    return [h["term"] for h in term_hits if re.search(rf"\b{re.escape(h['term'])}\b", head, re.I)][:k]


def delete_terms(text: str, terms) -> str:
    terms = sorted(set(terms), key=len, reverse=True)
    if not terms:
        return text
    return re.sub(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b", " ", text, flags=re.I)


def random_terms(text: str, k: int, exclude, rng: random.Random) -> list[str]:
    """k distinct non-stopword words of the visible text, other than the cited ones."""
    skip = set(STOP_WORDS) | {t.lower() for t in exclude}
    vocab = sorted({w.lower() for w in re.findall(TOKEN_PATTERN, text[:TEXT_CAP])} - skip)
    return rng.sample(vocab, k) if len(vocab) >= k else []


def _margin(E: np.ndarray, C: np.ndarray, own: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sims = E @ C.T
    rows = np.arange(len(E))
    own_sim = sims[rows, own]
    sims[rows, own] = -np.inf
    return own_sim, own_sim - sims.max(axis=1)


def faithfulness(items: list[dict], cids, C, encode: Callable, draws: int = 3, seed: int = 42) -> dict:
    """items: {doc_id, text, cluster_id, cited}. encode(texts) -> unit-norm embeddings of the
    run's embedding model. Per doc: drop in own-centroid cosine and in centroid margin after
    deleting the cited terms vs. an equal number of random terms (mean over `draws`)."""
    rng = random.Random(seed)
    cids = list(map(int, cids))
    kept, variants = [], []
    for it in items:
        text = it["text"][:TEXT_CAP]
        rnd = [random_terms(text, len(it["cited"]), it["cited"], rng) for _ in range(draws)]
        if it["cited"] and all(rnd):
            kept.append(it)
            variants += [text, delete_terms(text, it["cited"])] + [delete_terms(text, r) for r in rnd]
    if not kept:
        return {"n": 0, "per_doc": []}

    E = np.asarray(encode(variants), dtype=np.float32).reshape(len(kept), 2 + draws, -1)
    own = np.array([cids.index(int(it["cluster_id"])) for it in kept])
    sim, mar = zip(*(_margin(E[:, v], C, own) for v in range(2 + draws)))
    sim, mar = np.stack(sim, 1), np.stack(mar, 1)            # docs x variants
    d_sim_cited, d_sim_rand = sim[:, 0] - sim[:, 1], sim[:, 0] - sim[:, 2:].mean(axis=1)
    d_mar_cited, d_mar_rand = mar[:, 0] - mar[:, 1], mar[:, 0] - mar[:, 2:].mean(axis=1)

    def ratio(a, b):
        return float(a.mean() / b.mean()) if b.mean() > 0 else None

    return {
        "n": len(kept),
        "margin_drop_cited": float(d_mar_cited.mean()), "margin_drop_random": float(d_mar_rand.mean()),
        "margin_ratio": ratio(d_mar_cited, d_mar_rand),
        "sim_drop_cited": float(d_sim_cited.mean()), "sim_drop_random": float(d_sim_rand.mean()),
        "sim_ratio": ratio(d_sim_cited, d_sim_rand),
        "frac_cited_hurts_more": float((d_mar_cited > d_mar_rand).mean()),
        "assignment_flips_cited": float((mar[:, 1] < 0).mean()),
        "assignment_flips_random": float((mar[:, 2:] < 0).mean()),
        "per_doc": [{"doc_id": it["doc_id"], "cited": it["cited"], "margin_drop_cited": float(a),
                     "margin_drop_random": float(b)} for it, a, b in zip(kept, d_mar_cited, d_mar_rand)],
    }


# ==========================================================
# LLM judge (stand-in for the human raters)
# ==========================================================

class Plausibility(BaseModel):
    reason: str = Field(description="One sentence justifying the score")
    score: int = Field(description="Integer 1-5")


class TripleCheck(BaseModel):
    reason: str = Field(description="One sentence justifying the verdict")
    correct: bool = Field(description="True only if the document supports the triple, types included")


class LabelChoice(BaseModel):
    reason: str = Field(description="One sentence justifying the choice")
    better: Literal["A", "B"]


PLAUSIBILITY_PROMPT = """You are a strict evaluator of explanations produced by a document-clustering system.
The system put the news-group post below into a cluster and wrote an explanation of why.
Rate how plausible and well supported the explanation is, from 1 to 5:
1 = wrong or misleading; 2 = weak or mostly generic; 3 = acceptable but vague;
4 = good, specific and accurate; 5 = excellent, specific, accurate and clearly grounded in the post.
Judge against the post itself, not only the evidence listed.

Cluster keywords: {keywords}
Evidence: cluster terms found in the post: {hits}; centroid margin {margin}
Explanation: {rationale}

Post (truncated):
\"\"\"{text}\"\"\"
"""

TRIPLE_PROMPT = """Check one knowledge-graph triple extracted from the document below.
Entity types: {types}
Triple: ({source} [{source_type}]) -{relation}-> ({target} [{target_type}])
Relation meaning: {relation_doc}
Answer correct=true only if the document states or clearly implies this relation between these two
entities and both entity types are right. Otherwise false.

Document:
\"\"\"{text}\"\"\"
"""

LABEL_PROMPT = """Below are posts from one cluster of a news-group corpus, and two candidate labels for the cluster.
Pick the label that better describes what the posts have in common: accurate, specific, readable.

Posts:
{posts}

Label A: {a}
Label B: {b}
"""


def judge(model: str, task: str, prompt: str, schema):
    """Cached structured judgment (dict) or None when the model never produced valid output."""
    from swm.llm import BadOutput, structured

    key = hashlib.sha1(f"{task}\n{prompt}".encode()).hexdigest()[:20]
    path = CACHE / "eval" / re.sub(r"[^\w.-]", "_", model) / f"{task}-{key}.json"
    if path.exists():
        out = json.loads(path.read_text())
    else:
        try:
            out = structured(prompt, schema, model=model).model_dump()
        except BadOutput:
            out = {"failed": True}  # deterministic at temperature 0, so cache the failure
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return None if out.get("failed") else out


def judge_all(model: str, task: str, prompts: list[str], schema, workers: int = 8) -> list:
    from swm.llm import _chat
    from tqdm import tqdm

    _chat(model)  # build the client before the worker threads start
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(lambda p: judge(model, task, p, schema), prompts),
                         total=len(prompts), desc=f"{task} [{model}]"))


def plausibility_prompt(it: dict) -> str:
    return PLAUSIBILITY_PROMPT.format(
        keywords=", ".join(it["cluster_keywords"]),
        hits=", ".join(f'{h["term"]} x{h["count"]}' for h in it["term_hits"][:8]) or "none",
        margin=it["margin"], rationale=it["rationale"], text=it["text"][:1500])


def triple_prompt(t: dict, onto: dict) -> str:
    return TRIPLE_PROMPT.format(
        types="; ".join(f"{k}: {v['descriptor']}" for k, v in onto["entities"].items()),
        relation_doc=onto["relations"][t["relation"]]["descriptor"], text=t["text"][:3000],
        **{k: t[k] for k in ("source", "source_type", "relation", "target", "target_type")})


def label_prompts(c: dict) -> list[tuple[str, str]]:
    """Both presentation orders, to cancel position bias: [(prompt, which label sits at 'A')]."""
    posts = "\n".join(f'- """{s}"""' for s in c["samples"])
    return [(LABEL_PROMPT.format(posts=posts, a=a, b=b), first)
            for first, a, b in (("kg", c["kg_label"], c["ctfidf_label"]), ("ctfidf", c["ctfidf_label"], c["kg_label"]))]


# ==========================================================
# Aggregation
# ==========================================================

def krippendorff_alpha(ratings: list[list]) -> float | None:
    """Ordinal alpha over judges x items (None = missing). Needs >=2 judges and some variation."""
    if len(ratings) < 2:
        return None
    import krippendorff

    data = np.array([[np.nan if r is None else r for r in row] for row in ratings], dtype=float)
    try:
        return float(krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal"))
    except ValueError:  # every rating identical: alpha undefined
        return None


def summarize_plausibility(by_judge: dict[str, list]) -> dict:
    scores = {m: [None if o is None else min(5, max(1, int(o["score"]))) for o in outs]
              for m, outs in by_judge.items()}
    flat = [s for v in scores.values() for s in v if s is not None]
    return {
        "n_items": len(next(iter(scores.values()), [])),
        "mean": float(np.mean(flat)) if flat else None,
        "mean_by_judge": {m: float(np.mean([s for s in v if s is not None] or [np.nan])) for m, v in scores.items()},
        "krippendorff_alpha": krippendorff_alpha(list(scores.values())),
        "invalid_outputs": sum(s is None for v in scores.values() for s in v),
    }


def summarize_triples(by_judge: dict[str, list]) -> dict:
    prec = {m: [o["correct"] for o in outs if o is not None] for m, outs in by_judge.items()}
    flat = [c for v in prec.values() for c in v]
    return {
        "n_triples": len(next(iter(by_judge.values()), [])),
        "precision": float(np.mean(flat)) if flat else None,
        "precision_by_judge": {m: float(np.mean(v)) if v else None for m, v in prec.items()},
        "invalid_outputs": sum(o is None for outs in by_judge.values() for o in outs),
    }


def summarize_labels(wins: list[bool], n_clusters: int, by_judge: dict[str, list[bool]]) -> dict:
    return {
        "n_clusters": n_clusters,
        "kg_preferred": float(np.mean(wins)) if wins else None,
        "kg_preferred_by_judge": {m: float(np.mean(v)) if v else None for m, v in by_judge.items()},
    }


def duplicate_rate(entities: list[dict]) -> float:
    """Share of distinct (type, name) entities that collapse into another one once case,
    punctuation and a leading 'the' are ignored; measured after kg.normalize."""
    names = {(e["type"], e["name"]) for e in entities}
    keys = {(t, re.sub(r"[^a-z0-9]", "", re.sub(r"^the\s+", "", n.lower()))) for t, n in names}
    return 1 - len(keys) / len(names) if names else 0.0


def check_targets(report: dict) -> dict[str, dict]:
    """Report value vs PLAN.md target for every metric that was computed."""
    f, p, t, lab = (report.get(k) or {} for k in ("faithfulness", "plausibility", "triples", "labels"))
    kg = report.get("kg") or {}
    values = {
        "nmi": report.get("best_nmi"), "faithfulness_ratio": f.get("margin_ratio"),
        "plausibility_mean": p.get("mean"), "plausibility_alpha": p.get("krippendorff_alpha"),
        "kg_label_preferred": lab.get("kg_preferred"), "triple_precision": t.get("precision"),
        "ontology_conformance": kg.get("ontology_conformance"),
    }
    return {k: {"value": v, "target": TARGETS[k], "pass": None if v is None else v >= TARGETS[k]}
            for k, v in values.items()}
