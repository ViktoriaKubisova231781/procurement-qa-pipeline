# Procurement Q&A Dataset — KPI Self-Evaluation

Final dataset size: **25 pairs**

## Results vs. targets

| KPI | Target | Result | Pass |
|-----|--------|--------|------|
| Topic diversity | >= 5 sub-topics | 7 topics | PASS |
| Groundedness | >= 90% traceable | 100% (verbatim quote: 88%) | PASS |
| Relevance | >= 85% relevant | 100% | PASS |
| Answer quality | >= 80% correct & complete | 100% rated 4-5 | PASS |
| Low duplication | <= 10% near-dupes | 0% (deduped at cos >= 0.9) | PASS |
| Reproducibility | re-run from README | see README.md | PASS |

## How each KPI was measured

- **Topic diversity** — categorical distribution over the 7-topic taxonomy.
- **Groundedness** — objective verbatim match of each answer's `supporting_quote` against its source passage (string match), combined with answer-to-passage embedding similarity. No LLM judgment.
- **Relevance** — LLM-as-judge boolean (Llama 3.1), a different model family than the generator (Qwen 2.5) to avoid self-preference.
- **Answer quality** — LLM-as-judge 5-point rubric (correctness + completeness); the reported figure is the share rated 4-5.
- **Low duplication** — pairwise cosine similarity of question embeddings; any pair >= 0.9 is collapsed.

## Distributions

**By topic:** e-procurement and data standards (4), procurement policy and compliance (4), tender process (4), risk and sustainability (4), contract management (4), supplier evaluation (3), logistics and KPIs (2)

**By question type:** factual (20), analytical (5)

**By difficulty:** easy (13), medium (12)

## Honest limitations

- The LLM judge (8B) is lenient and clusters scores at the top of the scale even with a strict rubric; groundedness — measured objectively — is therefore the primary quality gate.
- Verbatim-quote matching confirms the cited span exists in the source, but does not by itself guarantee every clause of the answer is entailed; a natural-language-inference check is future work.
- Some source passages retain PDF/OCR artifacts, which can lower the verbatim-match score even when the answer is correct.
