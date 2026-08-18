"""
Shared sentence-embedding model.

One small model (all-MiniLM-L6-v2, 22M params, CPU-friendly) does FOUR jobs,
which is deliberate — reusing one component keeps the pipeline lean and is part
of the innovation story (Innovation #2):

  1. Semantic chunking   — find topic boundaries inside a document.
  2. Topic tagging       — nearest-topic-label by cosine similarity.
  3. Groundedness check  — answer-vs-passage embedding similarity (Phase 4).
  4. Deduplication       — question-vs-question similarity (Phase 4).

Loaded once and cached so we don't re-instantiate per call.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed(model_name: str, texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings so dot product == cosine similarity."""
    model = get_embedder(model_name)
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False)
    return vecs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two already-normalised vectors."""
    return float(np.dot(a, b))
