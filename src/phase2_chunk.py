"""
Phase 2 — Semantic Chunking & Topic Tagging
============================================

Turns the raw source documents from Phase 1 into semantically coherent
passages ("chunks"), each tagged with a procurement sub-topic.

Why semantic chunking (Innovation #2)
-------------------------------------
Naive fixed-length chunking cuts every N tokens, often mid-idea, which
produces passages that answer nothing cleanly. Instead we:
  1. Split each document into sentences.
  2. Embed every sentence with the shared sentence-transformer.
  3. Measure the semantic *distance* between each pair of consecutive
     sentences. Where that distance spikes (a topic shift), we place a
     breakpoint. The `breakpoint_percentile` in config controls sensitivity.
  4. Group sentences between breakpoints into passages, respecting the
     target/min token guardrails so passages stay a sensible size.

The SAME embedding model is then reused to TAG each chunk: we embed the 7
topic labels once and assign each chunk the nearest label by cosine
similarity. (That model is reused again in Phase 4 for groundedness + dedup.)

Output
------
data/processed/chunks.json  — list of Chunk records
Plus a printed summary (chunk counts, topic distribution, size stats) so we
can INSPECT quality and TUNE the chunking parameters before generating Q&A.

Run:  python src/phase2_chunk.py
No LLM required (uses local embeddings only).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import Chunk, SourceDoc  # noqa: E402
from embeddings import embed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- sentence splitting ----------------------------------------------------
# Lightweight, dependency-free sentence splitter. Good enough for chunking;
# we deliberately avoid heavyweight NLP just to cut on punctuation.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    # Work line-by-line so we respect existing structure, then split further.
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = _SENT_SPLIT.split(line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def approx_tokens(text: str) -> int:
    # ~1.3 tokens per word is a decent rule of thumb for English.
    return int(len(text.split()) * 1.3)


# --- semantic breakpoint detection -----------------------------------------
def semantic_chunks(sentences: list[str], embedder: str,
                    target_tokens: int, min_tokens: int,
                    breakpoint_percentile: int) -> list[str]:
    """Group sentences into passages at points of high semantic shift."""
    if len(sentences) <= 1:
        return [" ".join(sentences)] if sentences else []

    vecs = embed(embedder, sentences)  # (n, d), normalised

    # distance between consecutive sentences = 1 - cosine similarity
    dists = np.array([1.0 - float(np.dot(vecs[i], vecs[i + 1]))
                      for i in range(len(sentences) - 1)])

    # A breakpoint is any gap whose distance exceeds the chosen percentile.
    threshold = np.percentile(dists, breakpoint_percentile)

    chunks: list[str] = []
    current: list[str] = []
    cur_tokens = 0

    for i, sent in enumerate(sentences):
        current.append(sent)
        cur_tokens += approx_tokens(sent)

        at_breakpoint = i < len(dists) and dists[i] >= threshold
        too_big = cur_tokens >= target_tokens

        # Close the chunk at a semantic breakpoint OR when it hits target size,
        # but only if it's already big enough to stand alone.
        if (at_breakpoint or too_big) and cur_tokens >= min_tokens:
            chunks.append(" ".join(current))
            current, cur_tokens = [], 0

    if current:  # leftover tail
        tail = " ".join(current)
        # merge a too-small tail into the previous chunk instead of dropping it
        if approx_tokens(tail) < min_tokens and chunks:
            chunks[-1] = chunks[-1] + " " + tail
        else:
            chunks.append(tail)
    return chunks


def fixed_chunks(sentences: list[str], target_tokens: int,
                 min_tokens: int) -> list[str]:
    """Fallback: pack sentences to target size without semantics.
    Kept so we can compare semantic vs fixed in the write-up."""
    chunks, current, cur = [], [], 0
    for sent in sentences:
        current.append(sent)
        cur += approx_tokens(sent)
        if cur >= target_tokens:
            chunks.append(" ".join(current))
            current, cur = [], 0
    if current:
        tail = " ".join(current)
        if approx_tokens(tail) < min_tokens and chunks:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)
    return chunks


# --- topic tagging ---------------------------------------------------------
def tag_topics(chunk_texts: list[str], topics: list[str],
               embedder: str) -> list[str]:
    """Assign each chunk its nearest topic label by cosine similarity."""
    if not chunk_texts:
        return []
    topic_vecs = embed(embedder, topics)
    chunk_vecs = embed(embedder, chunk_texts)
    sims = chunk_vecs @ topic_vecs.T          # (n_chunks, n_topics)
    best = sims.argmax(axis=1)
    return [topics[i] for i in best]


def run(config: dict) -> list[Chunk]:
    raw_dir = ROOT / config["paths"]["raw"]
    proc_dir = ROOT / config["paths"]["processed"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    docs = [SourceDoc(**d) for d in manifest]

    cfg_c = config["chunking"]
    embedder = config["models"]["embedder"]
    method = cfg_c["method"]

    all_chunks: list[Chunk] = []
    per_source_counts: dict[str, int] = {}

    print(f"Chunking {len(docs)} sources  (method={method}, "
          f"target={cfg_c['target_tokens']}t, min={cfg_c['min_tokens']}t, "
          f"pctile={cfg_c['breakpoint_percentile']})\n")

    for doc in docs:
        text = (ROOT / doc.text_path).read_text(encoding="utf-8")
        sentences = split_sentences(text)

        if method == "fixed":
            pieces = fixed_chunks(sentences, cfg_c["target_tokens"],
                                  cfg_c["min_tokens"])
        else:
            pieces = semantic_chunks(sentences, embedder,
                                     cfg_c["target_tokens"],
                                     cfg_c["min_tokens"],
                                     cfg_c["breakpoint_percentile"])

        tags = tag_topics(pieces, config["topics"], embedder)
        for i, (piece, tag) in enumerate(zip(pieces, tags)):
            all_chunks.append(Chunk(
                chunk_id=f"{doc.id}::c{i:03d}",
                source_id=doc.id,
                source_url=doc.url,
                license=doc.license,
                topic_tag=tag,
                text=piece,
                n_tokens=approx_tokens(piece),
            ))
        per_source_counts[doc.id] = len(pieces)
        print(f"  {doc.id:<28} -> {len(pieces):>3} chunks")

    out_path = proc_dir / "chunks.json"
    out_path.write_text(
        json.dumps([c.model_dump() for c in all_chunks], indent=2,
                   ensure_ascii=False),
        encoding="utf-8",
    )

    _summary(all_chunks, out_path)
    return all_chunks


def _summary(chunks: list[Chunk], out_path: Path) -> None:
    from collections import Counter
    print("\n" + "=" * 66)
    print(f"Total chunks: {len(chunks)}  ->  {out_path.relative_to(ROOT)}")
    print("-" * 66)
    print("Topic distribution:")
    for topic, n in Counter(c.topic_tag for c in chunks).most_common():
        print(f"  {n:>3}  {topic}")
    sizes = [c.n_tokens for c in chunks]
    if sizes:
        print("-" * 66)
        print(f"Chunk size (approx tokens):  min {min(sizes)}  "
              f"median {int(np.median(sizes))}  max {max(sizes)}  "
              f"mean {int(np.mean(sizes))}")
        tiny = sum(1 for s in sizes if s < 60)
        huge = sum(1 for s in sizes if s > 400)
        print(f"Outliers:  {tiny} very small (<60t)   {huge} very large (>400t)")
    print("=" * 66)


if __name__ == "__main__":
    run(load_config())