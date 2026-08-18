"""
Phase 3 — Q&A Generation (LLM-assisted, structured & grounded)
==============================================================

Reads the tagged chunks from Phase 2 and, for a balanced sample of them,
asks the local generator model to produce ONE grounded question/answer pair
per passage. Output is written to data/processed/candidates.json for Phase 4
to validate and trim to the final 25.

Why this design (not "prompt an LLM for 25 pairs")
--------------------------------------------------
Every pair is tied to a specific source passage, and the model is REQUIRED to
return a `supporting_quote` — the exact sentence(s) from that passage that
justify the answer (Innovation #1: structured generation with cited spans).
Phase 4 then verifies that quote actually occurs in the source, which is what
makes groundedness *provable* rather than asserted.

Balanced sampling (protects the diversity KPI)
----------------------------------------------
Chunk topics are uneven (some sources are huge). If we sampled randomly we'd
over-represent big topics. Instead we round-robin across the 7 topics and cap
how many candidates come from any single source, so the candidate pool is
diverse before Phase 4 even runs.

Run:  python src/phase3_generate.py   (requires Ollama running)
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import Chunk, QACandidate  # noqa: E402
from llm import OllamaClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# JSON schema handed to Ollama to CONSTRAIN the output. The model must return
# exactly these fields — this is what makes the cited-span approach reliable.
QA_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "supporting_quote": {"type": "string"},
        "question_type": {"type": "string",
                          "enum": ["factual"]},
        "difficulty": {"type": "string",
                       "enum": ["easy", "medium", "hard"]},
    },
    "required": ["question", "answer", "supporting_quote",
                 "question_type", "difficulty"],
}

SYSTEM = (
    "You are a procurement/supply-chain domain expert building a high-quality "
    "question-answer dataset for training an assistant. You write clear, "
    "self-contained questions and accurate answers that are FULLY grounded in "
    "the passage you are given. You never invent facts not present in the "
    "passage."
)

PROMPT_TEMPLATE = """\
Below is a passage from a procurement/supply-chain source document.

PASSAGE:
\"\"\"
{passage}
\"\"\"

Write ONE high-quality FACTUAL question-answer pair that this passage answers.
A FACTUAL question asks about a definition, a specific fact, a figure, a rule,
a purpose, or a concrete piece of information stated in the passage. Do NOT write
open-ended "discuss" questions or hypothetical scenarios.

CRITICAL RULE — the QUESTION must be a UNIVERSAL procurement question that a
procurement professional could ask and understand WITHOUT ever seeing this
passage. It must be self-contained and general.

The QUESTION must NEVER:
- refer to "the passage", "the text", "this document", "the study", "this
  research", "the article", "the guidance", "the author(s)", or "the report";
- name or reference specific researchers, authors, or papers (e.g. do NOT write
  "Why do Pekša and Grabis suggest..." or "What do the authors propose...");
- ask about a specific figure, table, section, or citation in a document;
- depend on any context that a general reader would not already share.

GOOD question (universal): "What is the purpose of a framework agreement in
public procurement?"
BAD question (passage-bound): "What does this guidance say about framework
agreements?"
BAD question (author-bound): "Why do the authors recommend decoupling decision
logic from ERP systems?"

Other requirements:
- The ANSWER must be accurate, complete, and grounded ONLY in the passage. Every
  claim, figure, and list item in the answer MUST be supported by the passage.
  Do NOT add facts, numbers, or examples not stated in the passage, even if they
  sound plausible or you know them to be true.
- supporting_quote must be an EXACT substring copied verbatim from the passage
  that justifies the answer (do not paraphrase it). The answer must not claim
  more than this quote (and the surrounding passage) actually supports.
- difficulty: rate the question "easy" (a simple definition or single fact),
  "medium" (requires understanding a rule or relationship), or "hard" (requires
  connecting several details or precise domain knowledge).
- Prefer substantive, general procurement knowledge over trivia about document
  structure, section numbers, authors, citations, contact details, or websites.

Return ONLY a JSON object with keys: question, answer, supporting_quote,
question_type, difficulty. Always set question_type to "factual"."""


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(config: dict) -> list[Chunk]:
    path = ROOT / config["paths"]["processed"] / "chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**c) for c in data]


def select_chunks(chunks: list[Chunk], config: dict) -> list[Chunk]:
    """Balanced, capped sampling across topics and sources."""
    g = config["generation"]
    rnd = random.Random(config["random_seed"])

    # Filter out too-short chunks (usually ToC / stubs).
    pool = [c for c in chunks if c.n_tokens >= g["min_chunk_tokens"]]
    rnd.shuffle(pool)

    n_target = g["n_candidates"]
    max_per_source = g["max_per_source"]

    if not g.get("balance_by_topic", True):
        return pool[:n_target]

    # Group by topic, then round-robin so every topic contributes.
    by_topic: dict[str, list[Chunk]] = defaultdict(list)
    for c in pool:
        by_topic[c.topic_tag].append(c)

    selected: list[Chunk] = []
    per_source: dict[str, int] = defaultdict(int)
    topics = list(by_topic.keys())
    idx = {t: 0 for t in topics}

    # Round-robin across topics until we hit the target or exhaust the pool.
    while len(selected) < n_target and any(idx[t] < len(by_topic[t]) for t in topics):
        for t in topics:
            if len(selected) >= n_target:
                break
            # advance to the next chunk in this topic that respects source cap
            while idx[t] < len(by_topic[t]):
                cand = by_topic[t][idx[t]]
                idx[t] += 1
                if per_source[cand.source_id] < max_per_source:
                    selected.append(cand)
                    per_source[cand.source_id] += 1
                    break
    return selected


def generate(config: dict) -> list[dict]:
    chunks = load_chunks(config)
    picks = select_chunks(chunks, config)

    m = config["models"]
    client = OllamaClient(model=m["generator"], host=m["ollama_host"],
                          temperature=m["temperature"])

    if not client.health_check():
        print("!! Ollama not reachable or model missing.")
        print(f"   Start Ollama and run:  ollama pull {m['generator']}")
        sys.exit(1)

    print(f"Generating from {len(picks)} chunks "
          f"(model={m['generator']})...\n")

    candidates: list[dict] = []
    for i, chunk in enumerate(picks, 1):
        prompt = PROMPT_TEMPLATE.format(passage=chunk.text)
        try:
            raw = client.generate_json(SYSTEM, prompt, schema=QA_SCHEMA)
            qa = QACandidate(**raw)  # schema validation
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:>2}/{len(picks)}] {chunk.chunk_id:<34} skip ({type(e).__name__})")
            continue

        candidates.append({
            "id": f"cand_{len(candidates)+1:03d}",
            "question": qa.question.strip(),
            "answer": qa.answer.strip(),
            "supporting_quote": qa.supporting_quote.strip(),
            "question_type": qa.question_type,
            "difficulty": qa.difficulty,
            # provenance carried forward for Phase 4 + final xlsx
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "source_url": chunk.source_url,
            "license": chunk.license,
            "topic_tag": chunk.topic_tag,
            "source_passage": chunk.text,
        })
        print(f"  [{i:>2}/{len(picks)}] {chunk.chunk_id:<34} ok  [{chunk.topic_tag}]")

    out = ROOT / config["paths"]["processed"] / "candidates.json"
    out.write_text(json.dumps(candidates, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    _summary(candidates, out)
    return candidates


def _summary(cands: list[dict], out_path: Path) -> None:
    from collections import Counter
    print("\n" + "=" * 66)
    print(f"Generated {len(cands)} candidates  ->  {out_path.relative_to(ROOT)}")
    print("-" * 66)
    print("By topic:")
    for t, n in Counter(c["topic_tag"] for c in cands).most_common():
        print(f"  {n:>3}  {t}")
    print("By type:", dict(Counter(c["question_type"] for c in cands)))
    print("By difficulty:", dict(Counter(c["difficulty"] for c in cands)))
    print("=" * 66)


if __name__ == "__main__":
    generate(load_config())