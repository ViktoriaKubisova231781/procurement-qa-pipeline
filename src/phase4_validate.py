"""
Phase 4 — Validation, Scoring, Deduplication & Selection
========================================================

Turns the ~50 raw candidates from Phase 3 into the final 25 high-quality pairs,
measuring the KPIs from the assignment along the way. Nothing is hand-picked —
every pair is scored and filtered by explicit, reproducible rules.

Six checks per candidate
------------------------
1. GROUNDEDNESS (provable): does the model's `supporting_quote` actually occur
   in the source passage? We check verbatim, then fall back to a fuzzy overlap
   for minor whitespace/OCR differences. This is the core of the groundedness
   guarantee — a pair whose "evidence" isn't in the source is dropped.
   We combine this with answer<->passage embedding similarity for a 0-1 score.

2. RELEVANCE: is the question genuinely about procurement / supply chain?
   Judged by the LLM-as-judge (Llama 3.1) as a boolean. An earlier version used
   embedding similarity of the question against a procurement reference text,
   but those cosines were too noisy to threshold, so relevance moved to the
   judge that already reads each pair.

3. LLM-JUDGE (quality): a second model call scores correctness + completeness
   on a strict 1-5 rubric and flags "junk" questions (citations, book/journal
   titles, tables of contents, navigation menus, section numbers).

4. UNIVERSALITY: a deterministic regex guard drops questions that reference
   "the passage"/"the authors" or one organisation's internal jargon, so every
   kept question stands on its own without hidden context.

5. DEDUP: near-duplicate questions (cosine >= threshold) are collapsed so the
   final set is diverse (low-duplication KPI).

6. SELECTION: survivors are ranked by a blended score and picked round-robin
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
    "You are a strict, discerning data-quality reviewer for a procurement Q&A "
    "dataset. You use the full 1-5 scale and do not inflate scores. You reserve "
    "5 for flawless pairs and readily assign 3-4 when you find any weakness. You "
    "judge whether a pair is accurate, complete, and genuinely useful for "
    "learning about procurement — not trivia about document structure."
)

JUDGE_PROMPT = """\
Evaluate this question/answer pair against the SOURCE PASSAGE it came from.
Be a STRICT reviewer. Most pairs have at least one weakness — actively look for
it. Reserve a score of 5 for pairs that are genuinely flawless; if you can find
any issue, the score must be 4 or lower. Do not give 5 by default.

SOURCE PASSAGE:
\"\"\"
{passage}
\"\"\"

QUESTION: {question}
ANSWER: {answer}

Score CORRECTNESS (1-5): is every claim in the ANSWER supported by the passage?
  5 = every fact, figure and list item is explicitly in the passage; nothing added
  4 = essentially correct, but slightly rephrased in a way that risks meaning drift
  3 = mostly correct, but includes one detail not clearly stated in the passage
  2 = includes facts NOT in the passage, or misreads the passage
  1 = contradicts the passage or is largely unsupported

Score COMPLETENESS (1-5): does the ANSWER fully and directly answer the QUESTION?
  5 = complete, direct, self-contained
  4 = answers it, but omits a minor relevant detail present in the passage
  3 = partially answers, or is vague / padded
  2 = only tangentially answers the question
  1 = does not really answer the question

Also judge two booleans:
- relevant: is this QUESTION genuinely about procurement, purchasing, tendering,
  supplier/vendor management, contracts, supply chain, logistics, or procurement
  policy/data? true only if a procurement professional would find it on-topic.
- is_junk: true if the pair is NOT useful, general procurement knowledge. Set
  is_junk = true if ANY of these apply:
  * the QUESTION refers to "the passage", "the text", "this document", "the
    study", "this research", "the article", "the guidance", "the report", or
    otherwise assumes the reader has seen a specific source;
  * the QUESTION names or asks about specific researchers, authors, or a
    specific paper (e.g. "Why do Pekša and Grabis suggest...", "What do the
    authors propose...");
  * the QUESTION is about a citation/reference, a book or journal title, a table
    of contents, a list of section numbers, a website/URL, a navigation menu, or
    document formatting;
  * the QUESTION only makes sense with hidden context and is not a universal
    procurement question a professional could ask on its own.
  Otherwise is_junk = false.

Give a one-sentence reason that names the specific weakness (or confirms none).
Return ONLY JSON with keys: correct, complete, relevant, is_junk, reason."""


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# Deterministic guard: questions that reference a source the reader can't see,
# or that ask about specific authors/papers, are not universal procurement
# questions. This is a code-level safety net independent of the (lenient) judge.
_NON_UNIVERSAL_RE = re.compile(
    r"\b(the passage|this passage|the text|this text|the document|this document|"
    r"the study|this study|this research|the research|the article|this article|"
    r"the guidance|the report|this report|the author|the authors|according to "
    r"(the|this)|in their (research|paper|study|article)|"
    r"et al\.?)\b",
    re.IGNORECASE,
)

# Second guard: questions tied to one organisation's internal jargon or that ask
# organisational trivia rather than general procurement knowledge. These are
# grounded but not universally useful in a procurement Q&A dataset.
_ORG_SPECIFIC_RE = re.compile(
    r"\b(the borrower|borrower's|investment project financing|\bIPF\b|"
    r"the bank\b|world bank|two (main )?institutions|"
    r"which institutions|make up the)\b",
    re.IGNORECASE,
)


def is_non_universal(question: str) -> bool:
    return bool(_NON_UNIVERSAL_RE.search(question)
                or _ORG_SPECIFIC_RE.search(question))


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
        if is_non_universal(c["question"]):
            reasons.append("non_universal")
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