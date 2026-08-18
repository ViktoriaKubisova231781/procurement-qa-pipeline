"""
Phase 1 — Source Collection
===========================

Reads the `sources` list from config.yaml and, for each source:
  1. Downloads the content (HTML page or PDF).
  2. Extracts clean, readable text (strips nav/boilerplate for HTML;
     pulls text from PDFs).
  3. Saves the text to data/raw/<id>.txt
  4. Records metadata (url, license, topic, date accessed, char count)
     into data/raw/manifest.json  --> the audit trail for the citation KPI.

Design choices:
  * Each source is fetched INDEPENDENTLY inside a try/except, so one dead or
    JS-heavy URL never kills the whole run — it's logged as failed and we
    move on. This is what lets us "inspect, then swap" during development.
  * A realistic User-Agent is sent; several gov/standards portals reject the
    default python-requests agent.
  * The script prints a summary table at the end so we can immediately see
    which sources gave usable text (high char count) vs. which failed or
    returned near-empty boilerplate (candidates to swap).

Run:  python src/phase1_collect.py
No LLM required.
"""
from __future__ import annotations

import io
import json
import sys
from datetime import date
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

# --- make sibling modules importable when run as a script ------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import SourceDoc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

# HTML tags that never contain useful body text — dropped before extraction.
_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside",
               "form", "button", "noscript", "svg"]

# If a source yields fewer than this many characters, we treat the extraction
# as suspect (likely a JS wall or a redirect page) and flag it for review.
MIN_USABLE_CHARS = 800


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_html(url: str) -> str:
    """Download an HTML page and return cleaned, readable text."""
    resp = requests.get(url, headers=HEADERS, timeout=45)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # Prefer the main content region if the page marks one.
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    return _clean_text(text)


def fetch_pdf(url: str) -> str:
    """Download a PDF and extract its text."""
    from pypdf import PdfReader

    resp = requests.get(url, headers=HEADERS, timeout=90)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return _clean_text("\n".join(pages))


def _clean_text(text: str) -> str:
    """Collapse whitespace and drop empty / ultra-short lines."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 1]
    # de-duplicate consecutive identical lines (menus repeat a lot)
    cleaned: list[str] = []
    for ln in lines:
        if not cleaned or cleaned[-1] != ln:
            cleaned.append(ln)
    return "\n".join(cleaned).strip()


def collect(config: dict) -> list[SourceDoc]:
    raw_dir = ROOT / config["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    docs: list[SourceDoc] = []
    failures: list[tuple[str, str]] = []
    today = date.today().isoformat()

    for src in config["sources"]:
        sid, url, stype = src["id"], src["url"], src["type"]
        print(f"[fetch] {sid:<28} ({stype})  {url[:60]}...")
        try:
            text = fetch_pdf(url) if stype == "pdf" else fetch_html(url)
            if len(text) < MIN_USABLE_CHARS:
                failures.append((sid, f"only {len(text)} chars (suspect)"))
            text_path = raw_dir / f"{sid}.txt"
            text_path.write_text(text, encoding="utf-8")
            docs.append(SourceDoc(
                id=sid, url=url, license=src["license"], type=stype,
                date_accessed=today, n_chars=len(text),
                text_path=str(text_path.relative_to(ROOT)),
            ))
        except Exception as e:  # noqa: BLE001 — we want to keep going
            failures.append((sid, f"{type(e).__name__}: {e}"))
            print(f"        !! FAILED: {e}")

    # --- write manifest ----------------------------------------------------
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([d.model_dump() for d in docs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- summary -----------------------------------------------------------
    print("\n" + "=" * 66)
    print(f"Collected {len(docs)}/{len(config['sources'])} sources "
          f"-> {manifest_path.relative_to(ROOT)}")
    print("-" * 66)
    for d in sorted(docs, key=lambda x: x.n_chars):
        flag = "  <-- LOW, review" if d.n_chars < MIN_USABLE_CHARS else ""
        print(f"  {d.id:<28} {d.n_chars:>8,} chars{flag}")
    if failures:
        print("-" * 66)
        print("Issues to review / swap:")
        for sid, why in failures:
            print(f"  {sid:<28} {why}")
    print("=" * 66)
    return docs


if __name__ == "__main__":
    collect(load_config())