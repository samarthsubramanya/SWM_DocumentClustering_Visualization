# Text2Cypher: old vs new prompt (Nov 15)

Same graph, same model, same questions; only the KG agent's prompt and schema text differ.

- **Old prompt:** commit `120732e`, with only the crash fix from `55cf578` applied so a run can finish.
- **New prompt:** branch `samarth/ui-text2cypher` at `55cf578`.
- **Graph:** cloud Neo4j (Aura): `all-mpnet-base-v2__hdbscan`, 10,305 posts, the pre-v4 ontology.
  This is not the graph behind the Nov 8 report (BGE, ontology v4, 576 posts), so the scores
  are not comparable with its 0.833.
- **Model:** `qwen3-30b-a3b` on ASU RC (`LLM_MAX_TOKENS=8192`).
- **Questions:**
  - `eval/text2cypher.yaml`, the main set: 20 questions, 14 scored.
  - `eval/text2cypher_heldout.yaml`, the held-out set: 10 questions, 9 scored. These were not used
    to tune the prompt.
  - Unscored questions have a gold answer that is empty or longer than 50 rows on this graph.

| Question set | Prompt | Execution accuracy | Valid on first try | Valid after repair | Wrong |
|---|---|---|---|---|---|
| Main | Old | 0.643 (9/14) | 95% | 100% | q08, q10, q12, q15, q18 |
| Main | New | 1.000 (14/14) | 100% | 100% | none |
| Held-out | Old | 0.778 (7/9) | 100% | 100% | h07, h09 |
| Held-out | New | 1.000 (9/9) | 100% | 100% | none |

With the old prompt, the weakest pattern was entity lookup, at 0/2 on the main set and 1/2 on the
held-out set. Its one repair was for the invented `lowercase` property. The new prompt
answers every scored question correctly on the first try.

The samples are small, so read 9/9 as "no errors seen", not as perfect accuracy.

Unscored on this graph: q07, q11, q13, q14, q16, q20 and h05. Rewording them as counts would
make them scorable.

Per-question results are in the four `text2cypher-20261115-*-prompt-*.json` files next to
this report.
