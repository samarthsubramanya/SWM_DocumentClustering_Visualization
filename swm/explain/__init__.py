"""Per-document evidence and LLM rationale. Evidence-first: the LLM verbalizes
structured evidence it cannot invent, so every rationale is checkable."""

from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from swm import db
from swm.cluster import unit

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
N_TERMS = 15
# Contraction fragments ("don't" -> "don") and email-domain tokens are noise as cluster evidence.
STOP_WORDS = list(ENGLISH_STOP_WORDS | {
    "don", "doesn", "didn", "isn", "wasn", "aren", "won", "ve", "ll", "edu", "com",
    # Usenet filler: topical-sounding to c-TF-IDF in small clusters, but explains nothing.
    "just", "like", "know", "think", "does", "good", "really", "say", "said", "going", "thing",
    "got", "want", "need", "make", "lot", "better", "im", "yes", "ok", "thanks", "anybody", "anyone",
})


# ==========================================================
# Cluster keywords
# ==========================================================

def ctfidf(texts: list[str], labels, top_n: int = N_TERMS, min_df: int = 5) -> dict[int, list]:
    """Class-based TF-IDF (BERTopic formula) -> {cluster_id: [[term, weight], ...]}; noise excluded."""
    labels = np.asarray(labels)
    core = np.flatnonzero(labels >= 0)
    cv = CountVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN, min_df=min_df, max_df=0.5)
    counts = cv.fit_transform([texts[i] for i in core])
    ids, rows = np.unique(labels[core], return_inverse=True)
    member = sparse.csr_matrix(
        (np.ones(len(core)), (rows, np.arange(len(core)))), shape=(len(ids), len(core))
    )
    tf = np.asarray((member @ counts).todense(), dtype=np.float64)          # clusters x vocab
    idf = np.log(1 + tf.sum() / len(ids) / np.maximum(tf.sum(axis=0), 1))
    w = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1) * idf
    vocab = cv.get_feature_names_out()
    return {
        int(c): [[str(vocab[j]), round(float(w[i, j]), 6)] for j in np.argsort(-w[i])[:top_n] if w[i, j] > 0]
        for i, c in enumerate(ids)
    }


# ==========================================================
# Evidence
# ==========================================================

def compute_evidence(ids, texts, X, assigned, is_noise, probs, keywords: dict[int, list],
                     n_exemplars: int = 3) -> list[dict]:
    """Evidence for every doc of one run; all inputs aligned with ids.

    keywords: c-TF-IDF terms per cluster (cluster_info.keywords).
    """
    assigned, is_noise = np.asarray(assigned), np.asarray(is_noise, dtype=bool)
    Xn = unit(np.asarray(X, dtype=np.float32))
    n, rows = len(ids), np.arange(len(ids))
    cids = np.array(sorted(keywords))
    core = ~is_noise

    # Centroid margin: own centroid vs runner-up (original embedding space).
    C = unit(np.stack([Xn[(assigned == c) & core].mean(axis=0) for c in cids]))
    sims = Xn @ C.T
    own = np.searchsorted(cids, assigned)
    own_sim = sims[rows, own]
    runner = runner_sim = None
    if len(cids) > 1:
        other = sims.copy()
        other[rows, own] = -np.inf
        runner = other.argmax(axis=1)
        runner_sim = other[rows, runner]

    # Cluster-keyword hits per doc.
    weights = {c: {t: w for t, w in kw} for c, kw in keywords.items()}
    vocab = sorted({t for kw in keywords.values() for t, _ in kw})
    counts = CountVectorizer(vocabulary=vocab, token_pattern=TOKEN_PATTERN).transform(texts).tocsr()

    # Nearest exemplars among the cluster's core members (chunked to bound memory).
    exemplars: dict[int, list] = {}
    for c in cids:
        docs_c = np.flatnonzero(assigned == c)
        cands = np.flatnonzero((assigned == c) & core)
        k = min(n_exemplars, len(cands))
        for start in range(0, len(docs_c), 1024):
            chunk = docs_c[start:start + 1024]
            S = Xn[chunk] @ Xn[cands].T
            S[chunk[:, None] == cands[None, :]] = -np.inf  # not yourself
            top = np.argsort(-S, axis=1)[:, :k]
            for r, d in enumerate(chunk):
                exemplars[d] = [
                    {"doc_id": str(ids[cands[j]]), "similarity": round(float(S[r, j]), 4)}
                    for j in top[r] if np.isfinite(S[r, j])
                ]

    out = []
    for i in range(n):
        c = int(assigned[i])
        lo, hi = counts.indptr[i], counts.indptr[i + 1]
        hits = sorted(
            ({"term": vocab[j], "count": int(v), "cluster_weight": weights[c][vocab[j]]}
             for j, v in zip(counts.indices[lo:hi], counts.data[lo:hi]) if vocab[j] in weights[c]),
            key=lambda h: -h["cluster_weight"],
        )
        p = None if probs is None else probs[i]
        out.append({
            "cluster_id": c,
            "is_noise": bool(is_noise[i]),
            "membership_prob": None if p is None or np.isnan(p) else round(float(p), 4),
            "own_centroid_sim": round(float(own_sim[i]), 4),
            "runner_up_cluster": None if runner is None else int(cids[runner[i]]),
            "runner_up_sim": None if runner is None else round(float(runner_sim[i]), 4),
            "margin": None if runner is None else round(float(own_sim[i] - runner_sim[i]), 4),
            "term_hits": hits,
            "cluster_keywords": [t for t, _ in keywords[c]],
            "exemplars": exemplars.get(i, []),
        })
    return out


def save_evidence(run_id: str, ids, evidence: list[dict]) -> None:
    """Upsert evidence; a stale rationale is cleared because its evidence changed."""
    db.executemany(
        """INSERT INTO explanations (run_id, doc_id, evidence) VALUES (%s, %s, %s)
           ON CONFLICT (run_id, doc_id) DO UPDATE
           SET evidence = EXCLUDED.evidence, rationale = NULL, updated_at = now()""",
        [(run_id, str(d), db.Jsonb(e)) for d, e in zip(ids, evidence)],
    )


def evidence(doc_id: str, run_id: str) -> dict:
    """c-TF-IDF term hits, centroid margin, nearest exemplars, membership prob."""
    rows = db.query(
        "SELECT evidence FROM explanations WHERE run_id = %s AND doc_id = %s", (run_id, doc_id)
    )
    if not rows:
        raise LookupError(f"No evidence for {doc_id} in {run_id}; run scripts/04_explain.py --run {run_id}")
    return rows[0]["evidence"]


# ==========================================================
# Rationale
# ==========================================================

RATIONALE_PROMPT = """You explain why a news-group post was placed in a cluster. Write exactly 3 sentences:
1. What the post is about, in your own words, from the post text.
2. How that matches the cluster's theme (its top terms and nearest posts). If the match is weak, say so.
3. What the embedding evidence adds (centroid cosine and margin). A margin below 0.05 is borderline.
Quote at least {need} of these terms verbatim: {terms}.
Use only the facts below and the post itself. Never present a generic word (e.g. "just", "like") as the reason,
and never justify the placement by similarity numbers alone.

Cluster {cluster_id}{label}: top terms {keywords}
Evidence:
- cluster terms found in the post (term x count): {hits}
- cosine to own centroid {own}; runner-up cluster {runner} at {runner_sim}; margin {margin}
- membership probability: {prob}
- nearest posts in the cluster: {exemplars}
- knowledge-graph entities mentioned: {entities}

Post (truncated):
\"\"\"{text}\"\"\"
"""


def evidence_terms(ev: dict, kg_entities=()) -> list[str]:
    return [h["term"] for h in ev["term_hits"]] + list(kg_entities)


def quoted_terms(text: str, terms) -> set[str]:
    return {t for t in terms if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE)}


def template_rationale(ev: dict, kg_entities=()) -> str:
    """Deterministic rationale built only from evidence: no-LLM mode and LLM fallback."""
    parts = [f"This document is in cluster {ev['cluster_id']}"
             + (" (HDBSCAN marked it noise; assigned to the nearest centroid)" if ev["is_noise"] else "") + "."]
    hits = [h["term"] for h in ev["term_hits"][:5]]
    if hits:
        parts.append("It uses the cluster's characteristic terms " + ", ".join(f'"{t}"' for t in hits) + ".")
    if ev.get("runner_up_cluster") is not None:
        parts.append(
            f"Its embedding is closer to this cluster's centroid (cosine {ev['own_centroid_sim']:.2f}) "
            f"than to cluster {ev['runner_up_cluster']} ({ev['runner_up_sim']:.2f}), a margin of {ev['margin']:.2f}."
        )
    if kg_entities:
        parts.append("Knowledge-graph entities: " + ", ".join(list(kg_entities)[:5]) + ".")
    return " ".join(parts)


def cached_rationale(doc_id: str, run_id: str) -> str | None:
    rows = db.query(
        "SELECT rationale FROM explanations WHERE run_id = %s AND doc_id = %s", (run_id, doc_id)
    )
    return rows[0]["rationale"] if rows else None


def _exemplar_snippets(exemplars: list[dict]) -> str:
    """Opening of each nearest post, so the LLM sees what the cluster's posts say (ids mean nothing to it)."""
    if not exemplars:
        return "none"
    rows = db.query("SELECT id, text FROM documents WHERE id = ANY(%s)", ([e["doc_id"] for e in exemplars],))
    return " | ".join('"' + " ".join(r["text"].split())[:150] + '"' for r in rows) or "none"


def rationale(doc_id: str, run_id: str, kg_entities=(), attempts: int = 3, force: bool = False) -> str:
    """LLM rationale that must quote >=2 evidence terms; regenerate on failure.

    Validated rationales are cached in explanations.rationale. If every attempt fails
    (or Ollama is down) the template rationale is returned and NOT cached.
    """
    rows = db.query(
        """SELECT e.evidence, e.rationale, d.text, ci.label
           FROM explanations e
           JOIN documents d ON d.id = e.doc_id
           LEFT JOIN cluster_info ci
             ON ci.run_id = e.run_id AND ci.cluster_id = (e.evidence->>'cluster_id')::int
           WHERE e.run_id = %s AND e.doc_id = %s""",
        (run_id, doc_id),
    )
    if not rows:
        raise LookupError(f"No evidence for {doc_id} in {run_id}; run scripts/04_explain.py --run {run_id}")
    row = rows[0]
    if row["rationale"] and not force:
        return row["rationale"]

    ev = row["evidence"]
    # The post may share no literal term with its cluster, so the cluster's own top terms are citable too.
    terms = list(dict.fromkeys(evidence_terms(ev, kg_entities) + ev["cluster_keywords"][:5]))
    need = min(2, len(set(terms)))
    prompt = RATIONALE_PROMPT.format(
        need=need,
        terms=", ".join(dict.fromkeys(terms)) or "(none)",
        cluster_id=ev["cluster_id"],
        label=f' ("{row["label"]}")' if row["label"] else "",
        keywords=", ".join(ev["cluster_keywords"]),
        hits=", ".join(f'{h["term"]} x{h["count"]}' for h in ev["term_hits"]) or "none",
        own=ev["own_centroid_sim"],
        runner=ev["runner_up_cluster"],
        runner_sim=ev["runner_up_sim"],
        margin=ev["margin"],
        prob=ev["membership_prob"],
        exemplars=_exemplar_snippets(ev["exemplars"]),
        entities=", ".join(kg_entities) or "none",
        text=row["text"][:1500],
    )

    from swm.llm import chat

    for _ in range(attempts):
        try:
            text = chat(prompt).strip()
        except Exception:  # Ollama unreachable / transient error
            continue
        if len(quoted_terms(text, terms)) >= need:
            db.execute(
                "UPDATE explanations SET rationale = %s, updated_at = now() WHERE run_id = %s AND doc_id = %s",
                (text, run_id, doc_id),
            )
            return text
    return template_rationale(ev, kg_entities)
