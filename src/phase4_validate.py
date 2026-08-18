"""
Phase 4 — Validation, Scoring, Deduplication & Selection
========================================================

Turns the ~50 raw candidates from Phase 3 into the final 25 high-quality pairs,
measuring the KPIs from the assignment along the way. Nothing is hand-picked —
every pair is scored and filtered by explicit, reproducible rules.

Five checks per candidate
-------------------------
1. GROUNDEDNESS (provable): does the model's `supporting_quote` actually occur
   in the source passage? We check verbatim, then fall back to a fuzzy overlap
   for minor whitespace/OCR differences. This is the core of the groundedness
   guarantee — a pair whose "evidence" isn't in the source is dropped.
   We combine this with answer<->passage embedding similarity for a 0-1 score.

2. RELEVANCE: is the question actually about procurement / supply chain? Scored
   by embedding similarity of the question against a procurement reference text.

3. LLM-JUDGE (quality): a second model call scores correctness + completeness
   on a 1-5 rubric AND flags "junk" questions (about citations, book/journal
   titles, tables of contents, navigation menus, section numbers) — the failure
   mode we saw in Phase 3 from reference lists and page furniture.

4. DEDUP: near-duplicate questions (cosine >= threshold) are collapsed so the
   final set is diverse (low-duplication KPI).

5. SELECTION: survivors are ranked by a blended score and picked round-robin
   across topics, so the final 25 stay diverse (topic-diversity KPI).

Output
------
data/processed/validated.json  — all candidates with scores + keep/drop reason
(The final .xlsx is produced in Phase 5 from the kept rows.)

Run:  python src/phase4_validate.py   (requires Ollama running)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import OllamaClient  # noqa: E402
from embeddings import embed, cosine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# The LLM judge assesses procurement relevance directly (an embedding cosine of
# a short question against a reference paragraph proved too noisy to be useful).

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "integer"},       # 1-5 accuracy vs passage
        "complete": {"type": "integer"},       # 1-5 completeness
        "relevant": {"type": "boolean"},       # is it a procurement/SC question?
        "is_junk": {"type": "boolean"},        # about citations/ToC/nav/etc.
        "reason": {"type": "string"},
    },
    "required": ["correct", "complete", "relevant", "is_junk", "reason"],
}

JUDGE_SYSTEM = (
    "You are a strict data-quality reviewer for a procurement Q&A dataset. "
    "You judge whether a question/answer pair is accurate, complete, and "
    "genuinely useful for learning about procurement — not trivia about "
    "document structure."
)

JUDGE_PROMPT = """\
Evaluate this question/answer pair against the SOURCE PASSAGE it came from.

SOURCE PASSAGE:
\"\"\"
{passage}
\"\"\"

QUESTION: {question}
ANSWER: {answer}

Score on a 1-5 scale:
- correct: is the ANSWER accurate and fully supported by the passage?
  (5 = fully supported, 1 = contradicted or unsupported)
- complete: does the ANSWER fully address the QUESTION? (5 = complete)

Also judge two booleans:
- relevant: is this QUESTION genuinely about procurement, purchasing, tendering,
  supplier/vendor management, contracts, supply chain, logistics, or procurement
  policy/data? true if a procurement professional would find it on-topic.
- is_junk: true if the pair is NOT useful procurement knowledge, e.g. the
  question is about a citation/reference, a book or journal title, an author,
  a table of contents, a list of section numbers, a website/URL, a navigation
  menu, or document formatting. Otherwise false.

Give a one-sentence reason. Return ONLY JSON with keys:
correct, complete, relevant, is_junk, reason."""


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_grounded(quote: str, passage: str) -> float:
    """1.0 if the quote appears verbatim (normalized) in the passage;
    otherwise a fuzzy ratio for near-matches (OCR/whitespace drift)."""
    q, p = _normalize(quote), _normalize(passage)
    if not q:
        return 0.0
    if q in p:
        return 1.0
    # sliding fuzzy match against the closest window
    return SequenceMatcher(None, q, p).ratio()


def groundedness_score(cand: dict, embedder: str) -> tuple[float, float]:
    """Return (quote_match, answer_similarity)."""
    qmatch = quote_grounded(cand["supporting_quote"], cand["source_passage"])
    vecs = embed(embedder, [cand["answer"], cand["source_passage"]])
    ans_sim = cosine(vecs[0], vecs[1])
    return qmatch, ans_sim


def run(config: dict) -> list[dict]:
    proc = ROOT / config["paths"]["processed"]
    cands = json.loads((proc / "candidates.json").read_text(encoding="utf-8"))

    m = config["models"]
    v = config["validation"]
    embedder = m["embedder"]
    judge = OllamaClient(model=m["judge"], host=m["ollama_host"],
                         temperature=0.0)  # judge is deterministic

    if not judge.health_check():
        print("!! Ollama not reachable. Start it and pull the model.")
        sys.exit(1)

    print(f"Validating {len(cands)} candidates...\n")

    # --- score every candidate --------------------------------------------
    for i, c in enumerate(cands, 1):
        qmatch, ans_sim = groundedness_score(c, embedder)
        # groundedness = mostly the verbatim quote check, backed by answer sim
        c["groundedness_score"] = round(0.7 * qmatch + 0.3 * max(0.0, ans_sim), 3)
        c["quote_verbatim"] = round(qmatch, 3)

        # LLM judge (correctness, completeness, relevance, junk)
        try:
            jr = judge.generate_json(
                JUDGE_SYSTEM,
                JUDGE_PROMPT.format(passage=c["source_passage"],
                                    question=c["question"], answer=c["answer"]),
                schema=JUDGE_SCHEMA,
            )
            c["judge_correct"] = int(jr.get("correct", 0))
            c["judge_complete"] = int(jr.get("complete", 0))
            c["judge_relevant"] = bool(jr.get("relevant", False))
            c["judge_is_junk"] = bool(jr.get("is_junk", False))
            c["judge_reason"] = jr.get("reason", "")
        except Exception as e:  # noqa: BLE001
            c["judge_correct"] = c["judge_complete"] = 0
            c["judge_relevant"] = False
            c["judge_is_junk"] = True
            c["judge_reason"] = f"judge error: {type(e).__name__}"
        # a single 1-5 quality number for convenience (min of the two rubric axes)
        c["judge_quality"] = min(c["judge_correct"], c["judge_complete"])
        # relevance_score kept as a 0/1 for the KPI rollup + xlsx column
        c["relevance_score"] = 1.0 if c["judge_relevant"] else 0.0
        print(f"  [{i:>2}/{len(cands)}] {c['id']}  "
              f"ground={c['groundedness_score']:.2f} "
              f"rel={'Y' if c['judge_relevant'] else 'N'} "
              f"judge={c['judge_quality']} "
              f"{'JUNK' if c['judge_is_junk'] else ''}")

    # --- apply pass/fail gates --------------------------------------------
    for c in cands:
        reasons = []
        if c["judge_is_junk"]:
            reasons.append("judge:junk")
        if not c["judge_relevant"]:
            reasons.append("relevance")
        if c["groundedness_score"] < v["groundedness_min"]:
            reasons.append("groundedness")
        if c["judge_quality"] < v["judge_quality_min"]:
            reasons.append("judge:quality")
        c["passed_gates"] = not reasons
        c["drop_reasons"] = reasons

    survivors = [c for c in cands if c["passed_gates"]]

    # --- deduplicate survivors by question similarity ---------------------
    survivors = _dedup(survivors, embedder, v["dedup_cosine_max"])

    # --- select final N, balanced across topics ---------------------------
    target = config["target_pairs"]
    final = _select_balanced(survivors, target)
    final_ids = {c["id"] for c in final}
    for c in cands:
        c["selected"] = c["id"] in final_ids

    out = proc / "validated.json"
    out.write_text(json.dumps(cands, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    _summary(cands, final, out, v)
    return cands


def _dedup(cands: list[dict], embedder: str, max_cos: float) -> list[dict]:
    """Greedy near-duplicate removal on question embeddings; keep higher score."""
    if not cands:
        return []
    ranked = sorted(cands, key=_blended_score, reverse=True)
    qvecs = {c["id"]: embed(embedder, [c["question"]])[0] for c in ranked}
    kept: list[dict] = []
    for c in ranked:
        if all(cosine(qvecs[c["id"]], qvecs[k["id"]]) < max_cos for k in kept):
            kept.append(c)
        else:
            c["drop_reasons"] = c.get("drop_reasons", []) + ["near_duplicate"]
            c["passed_gates"] = False
    return kept


def _blended_score(c: dict) -> float:
    """Rank key: quality dominates, then groundedness, then relevance."""
    return (c.get("judge_quality", 0) / 5.0) * 0.5 \
        + c.get("groundedness_score", 0) * 0.3 \
        + c.get("relevance_score", 0) * 0.2


def _select_balanced(cands: list[dict], target: int) -> list[dict]:
    """Round-robin across topics, best-first within each topic."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for c in sorted(cands, key=_blended_score, reverse=True):
        by_topic[c["topic_tag"]].append(c)
    topics = list(by_topic.keys())
    idx = {t: 0 for t in topics}
    picked: list[dict] = []
    while len(picked) < target and any(idx[t] < len(by_topic[t]) for t in topics):
        for t in topics:
            if len(picked) >= target:
                break
            if idx[t] < len(by_topic[t]):
                picked.append(by_topic[t][idx[t]])
                idx[t] += 1
    return picked


def _summary(cands, final, out_path, v) -> None:
    from collections import Counter
    passed = [c for c in cands if c.get("passed_gates")]
    print("\n" + "=" * 66)
    print(f"Validated {len(cands)} candidates -> {out_path.relative_to(ROOT)}")
    print("-" * 66)
    # KPI-style rollup on the FINAL set
    n = len(final)
    if n:
        gnd = sum(1 for c in final if c["groundedness_score"] >= v["groundedness_min"]) / n
        rel = sum(1 for c in final if c["judge_relevant"]) / n
        verb = sum(1 for c in final if c["quote_verbatim"] >= 0.99) / n
        qual = sum(1 for c in final if c["judge_quality"] >= 4) / n
        print(f"FINAL SET: {n} pairs")
        print(f"  Topic diversity : {len(set(c['topic_tag'] for c in final))} topics")
        print(f"  Groundedness    : {gnd*100:.0f}% >= threshold")
        print(f"  Verbatim quote  : {verb*100:.0f}% exact-match to source")
        print(f"  Relevance       : {rel*100:.0f}% >= threshold")
        print(f"  Judge quality   : {qual*100:.0f}% rated 4-5")
        print("  By topic:", dict(Counter(c["topic_tag"] for c in final)))
        print("  By type :", dict(Counter(c["question_type"] for c in final)))
    print("-" * 66)
    dropped = [c for c in cands if not c.get("selected")]
    drop_reasons = Counter(r for c in dropped for r in c.get("drop_reasons", []))
    print(f"Dropped {len(cands)-len(final)} (reasons): {dict(drop_reasons)}")
    print("=" * 66)


if __name__ == "__main__":
    run(load_config())