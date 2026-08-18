"""
Thin wrapper around a locally-served Ollama model.

Why Ollama (justification for Slide 4):
  * Free & local — no API keys, no per-token cost, no data leaving the machine.
  * Reproducible — the model is pinned by name+tag in config.yaml; a grader
    runs `ollama pull <model>` and gets byte-for-byte the same weights.
  * Structured output — Ollama's `format=json` (and schema passing) lets us do
    constrained generation, which is how we force the model to emit a
    supporting quote (Innovation #1).

The generator and the judge are both just calls through this class with
different prompts, so swapping either model is a one-line config change.
"""
from __future__ import annotations

import json
from typing import Any

import requests


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 temperature: float = 0.4, timeout: int = 180):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate_json(self, system: str, prompt: str,
                      schema: dict[str, Any] | None = None) -> dict:
        """
        Call the model and parse a JSON object back.

        If `schema` is provided we pass it to Ollama as the `format` field,
        which constrains decoding to valid JSON matching that schema. This is
        far more reliable than "please answer in JSON" prompting.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        payload["format"] = schema if schema else "json"

        resp = requests.post(f"{self.host}/api/generate",
                             json=payload, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json()["response"]
        return _safe_json_loads(raw)

    def health_check(self) -> bool:
        """Return True if Ollama is up and the model is available."""
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=10).json()
            names = {m["name"] for m in tags.get("models", [])}
            # tags may be like "qwen2.5:7b-instruct"; accept prefix match too
            return any(self.model == n or n.startswith(self.model.split(":")[0])
                       for n in names)
        except requests.RequestException:
            return False


def _safe_json_loads(raw: str) -> dict:
    """Parse JSON, tolerating stray markdown fences or leading/trailing text."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # last resort: grab the outermost {...}
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise
