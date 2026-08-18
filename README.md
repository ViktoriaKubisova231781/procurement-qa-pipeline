# Procurement Q&A Dataset Pipeline

A reproducible, source-grounded pipeline that produces **25 high-quality
question/answer pairs** for fine-tuning or evaluating a procurement / supply-chain
chat assistant. Every pair traces back to a specific passage in an open,
non-licensed source, and every answer is validated against that passage before it
is kept.

The pipeline is fully config-driven: sources, models, thresholds, and targets all
live in `config.yaml`, so re-running it (or pointing it at new sources) produces a
fresh, valid dataset with no manual edits to the code.

---

## What it produces

Running the pipeline end-to-end writes:

- `data/output/procurement_qa_dataset.xlsx` — the final 25 pairs. The first sheet
  has exactly the required columns (`id`, `question`, `answer`, `source_passage`,
  `source_url`, `topic_tag`, `groundedness_score`, `relevance_score`); a second
  sheet keeps the full scoring detail.
- `data/output/kpi_report.md` — a self-evaluation against every KPI target.

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** running locally (serves the
  generation and judge models over `http://localhost:11434`).
- ~10 GB free disk for the two Ollama models, and an internet connection for
  Phase 1 (source download) and the first embedding-model download.

The pipeline runs comfortably on a laptop with 16 GB RAM; a GPU speeds up
generation but is not required.

---

## Setup

```bash
# 1. Create and activate an environment (conda shown; venv works too)
conda create -n procqa python=3.11 -y
conda activate procqa

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Ollama (see https://ollama.com/download), then pull the two models
ollama pull qwen2.5:7b-instruct   # generator
ollama pull llama3.1:8b           # judge (different family, on purpose)
```

Make sure Ollama is running (the desktop app or `ollama serve`) before Phase 3.

---

## Running the pipeline

Run the phases in order from the project root:

```bash
python src/phase1_collect.py    # download sources -> data/raw/ + manifest.json
python src/phase2_chunk.py      # semantic chunking + topic tagging -> chunks.json
python src/phase3_generate.py   # LLM Q&A generation -> candidates.json   (needs Ollama)
python src/phase4_validate.py   # scoring, validation, dedup, selection -> validated.json (needs Ollama)
python src/phase5_output.py     # final .xlsx + KPI report -> data/output/
```

Each phase reads the previous phase's output from `data/`, so they must be run in
sequence. Phases 1–2 need no LLM; Phases 3–4 call the local Ollama models.

> **Tip:** close `procurement_qa_dataset.xlsx` in Excel before re-running Phase 5,
> or Windows will lock the file.

---

## How it works (phase by phase)

**Phase 1 — Source collection.** Downloads the open sources listed in
`config.yaml`, extracts clean text (HTML or PDF), and records provenance
(URL, license, date, character count) in `data/raw/manifest.json`. Each source is
fetched independently and failures are logged, so one bad URL never stops the run.

**Phase 2 — Semantic chunking & topic tagging.** Splits each document into
semantically coherent passages by embedding sentences and cutting at points of
topic shift (not fixed length), then tags each passage with the nearest of seven
procurement sub-topics. Uses `sentence-transformers/all-MiniLM-L6-v2` locally.

**Phase 3 — Q&A generation.** For a topic-balanced sample of passages, the
generator model (`qwen2.5:7b-instruct`) writes one factual, self-contained
question, a grounded answer, and a **verbatim supporting quote** copied from the
passage. Structured JSON output is enforced so the quote can be verified later.

**Phase 4 — Validation, dedup & selection.** Each candidate is scored for:
- **groundedness** — objective check that the supporting quote actually appears in
  the source passage (string match), combined with answer↔passage similarity;
- **relevance** and **quality** — judged by a *different* model family
  (`llama3.1:8b`) to reduce self-preference bias;
- **universality** — a deterministic filter drops questions that reference "the
  passage", specific authors, or one source's internal jargon.

Near-duplicates are removed by question-embedding similarity, and the survivors
are selected round-robin across topics to keep the final 25 diverse.

**Phase 5 — Output & reporting.** Writes the final `.xlsx` and the KPI report.

---

## Configuration

Everything tunable lives in `config.yaml`:

- `sources` — the open documents to collect (id, URL, license, type, topic).
- `models` — generator, judge, and embedding model names.
- `chunking` — passage size and semantic-split sensitivity.
- `generation` — how many candidates to generate and per-source caps.
- `validation` — groundedness and dedup thresholds.
- `topics` / `topic_descriptions` — the sub-topic taxonomy used for tagging.

To use different sources, edit the `sources` list and re-run from Phase 1. To swap
models, change `models` and re-run from Phase 3. No code changes required.

---

## Project structure

```
procurement-qa-pipeline/
├── README.md
├── requirements.txt
├── config.yaml                 # all sources, models, thresholds
├── src/
│   ├── schemas.py              # shared data contracts (pydantic)
│   ├── llm.py                  # Ollama client (structured JSON output)
│   ├── embeddings.py           # shared sentence-transformer
│   ├── phase1_collect.py
│   ├── phase2_chunk.py
│   ├── phase3_generate.py
│   ├── phase4_validate.py
│   └── phase5_output.py
└── data/
    ├── raw/                    # downloaded text + manifest.json
    ├── processed/              # chunks.json, candidates.json, validated.json
    └── output/                 # procurement_qa_dataset.xlsx, kpi_report.md
```

---

## Notes on model choice

All models are open and run locally via Ollama — no API keys, no per-token cost,
and fully reproducible (models are pinned by name in `config.yaml`). An 8B-class
generator is sufficient for reading a passage and writing a grounded factual Q&A,
and keeps the pipeline runnable on a laptop. Using a different model family for the
judge (`llama3.1:8b` judging `qwen2.5`'s output) mitigates the self-preference bias
that arises when a model grades its own generations.
