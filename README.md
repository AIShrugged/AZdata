# AZdata — e-Invoice AI

An AI system for Azerbaijani tax-authority **electronic-invoice** data, delivering two capabilities through one web app:

1. **Natural-language → SQL** — ask questions about the invoice data in Azerbaijani or English; the system writes *safe, read-only* SQL and returns the answer.
2. **Invoice-item classification** — given an invoice line item, decide **Good (Mal)** vs **Service (Xidmət)** and, for every Good, assign the **HS commodity code** from the full EQM catalogue (**11,641 active codes**).

**Headline result:** Good/Service classification is about **99%**, and HS-code candidate retrieval reaches about **95% recall@k on hard items** after EQM index enrichment, query expansion, and a wider candidate pool. Exact 10-digit HS codes remain human-confirmed. ([details ↓](#results))

> New here? Read this README, then [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the code/process deep-dive and [`docs/SOLUTION_REPORT.md`](docs/SOLUTION_REPORT.md) for the methodology & results narrative.

---

## Table of contents
- [What it looks like](#what-it-looks-like)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [How it works — the processes](#how-it-works--the-processes)
- [Setup](#setup)
- [Running it](#running-it)
- [Testing](#testing)
- [Results](#results)
- [Tech stack & key decisions](#tech-stack--key-decisions)
- [Documentation index](#documentation-index)

---

## What it looks like

One FastAPI backend serves both tasks and a single web app with four tabs:

| Tab | What it does |
|---|---|
| **NL Query** | question (EN/AZ) → the generated **SQL** + the answer **table** |
| **Classify item** | item → **Good/Service** badge + **HS commodity code** + which **tier** handled it |
| **Evals** | model-comparison chart/table, the accuracy **journey**, HS-code retrieval evidence, confusion, misclassifications |
| **Report** | the in-app "how it was done" methodology write-up |

Launch with `scripts/run_demo.sh` → open **http://127.0.0.1:8642/**. Demo walkthrough: [`docs/DEMO.md`](docs/DEMO.md).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  WEB APP  (web/index.html)  ·  4 tabs  ·  vanilla JS, no build step    │
└───────────────────────────┬────────────────────────────────────────────┘
                            │  HTTP / JSON  (CORS)
┌───────────────────────────▼────────────────────────────────────────────┐
│  API  ·  FastAPI  (src/api.py)                                          │
│   /query   /classify   /evals   /catalog   /health   + static web/      │
└──────────┬──────────────────────────────────┬──────────────────────────┘
        TASK 1                              TASK 2
┌──────────▼───────────────┐   ┌───────────────▼──────────────────────────┐
│ NL→SQL engine (nlsql.py) │   │ Two-tier router (router.py)              │
│  catalog.py → grounded   │   │   classify_rag (rag.py)                   │
│  prompt → LLM            │   │     → BGE-M3 retrieval (train_index)      │
│  → sqlglot GUARD         │   │   → escalate local→cloud if low-conf      │
│  → read-only SQL exec    │   │   → EQM HS-code (eqm.py): enriched         │
│                          │   │       registry retrieval → LLM rerank      │
└──────────┬───────────────┘   └───────────────┬──────────────────────────┘
       ┌───▼────┐                      ┌────────▼─────────┐
       │Postgres│                      │ BGE-M3 vector    │
       │einvoice│                      │ indexes (.npy)   │
       │taxpayer│                      └────────┬─────────┘
       └───┬────┘                               │
┌──────────▼───────────────────────────────────▼─────────────────────────┐
│ MODEL / PROVIDER LAYER  (nlsql.call_llm — one interface, 4 providers)   │
│   LOCAL: Ollama (qwen3.5 LLM · bge-m3 embeddings)                       │
│   CLOUD: OpenRouter (qwen3.5-35b/-122b) · OpenAI · Anthropic            │
└─────────────────────────────────────────────────────────────────────────┘
```

The single `call_llm(system, user, provider, model)` function abstracts four providers, so the same code runs **fully offline** (Ollama) or **cloud-backed** (OpenRouter/OpenAI/Anthropic) by changing one setting.

---

## Repository layout

```
AZdata/
├── README.md                 ← you are here
├── db/schema.sql             Postgres DDL (einvoice + taxpayer), business concepts as COMMENTs
├── config/
│   ├── metadata_enrichment.yaml   business concepts + AZ/EN synonyms + roles (Task 1 catalog)
│   └── catalog.json               generated catalog (DDL + enrichment merged)
├── data/
│   ├── *.xlsx / *.xls        raw sample data (e-invoices, goods/services labels, EQM registry)
│   └── processed/            generated: labeled_items / train,dev,test / *_index.npy / eval_summary.json …
├── src/
│   ├── nlsql.py              Task 1: NL→SQL engine + the call_llm provider layer + sqlglot guard
│   ├── catalog.py            Task 1: DDL + enrichment → metadata catalog
│   ├── classify.py           Task 2: Good/Service classifier
│   ├── rag.py                Task 2: BGE-M3 retrieval + few-shot RAG classifier
│   ├── eqm.py                Task 2: EQM HS-code assignment (enriched registry retrieval → LLM rerank)
│   ├── router.py             Task 2: two-tier router + full classify→HS pipeline
│   └── api.py                FastAPI app: /query /classify /evals /catalog /health + serves web/
├── scripts/
│   ├── ingest.py             load raw e-invoices → Postgres (+ seed Scenario-1 demo taxpayer)
│   ├── prep_task2.py         build labeled Good/Service set + clean EQM registry
│   ├── make_splits.py        stratified train/dev/test split
│   ├── optimize_prompt.py    agentic prompt-optimisation loop (improves the classifier instruction)
│   ├── build_eval_summary.py consolidate eval results → eval_summary.json (for the Evals tab)
│   ├── demo_test.py          16-check end-to-end test suite
│   └── run_demo.sh           preflight checks + launch the app
├── web/
│   ├── index.html            single-page app (4 tabs)
│   └── report.html           styled render of the solution report (Report tab)
└── docs/                     ARCHITECTURE · SOLUTION_REPORT · DEMO · ROADMAP · CHANGELOG · HANDOFF · PROJECT_MEMORY
```

---

## How it works — the processes

### Data pipeline (one-time build)
1. **`scripts/ingest.py`** — reads the e-invoice Excel → Postgres tables `einvoice` (3,716 rows) + `taxpayer`; seeds a demo taxpayer (`1234567890`) so the brief's Scenario 1 is reproducible. Idempotent.
2. **`scripts/prep_task2.py`** — from the goods/services Excel builds `data/processed/labeled_items.csv` (8,643 Good/Service items) and cleans the EQM registry (`eqm_registry.csv`, 11,641 HS codes; leading zeros restored to 10-digit).
3. **`scripts/make_splits.py`** — stratified **train/dev/test** = 6,050 / 1,295 / 1,298 (tune on dev, report on the untouched test).
4. **`src/rag.py --build`** and **`src/eqm.py --build`** — embed the train items and the EQM descriptions with **BGE-M3** (via Ollama) into `*_index.npy` vector indexes.

### Task 1 — Natural-language → SQL (`src/nlsql.py`, `src/catalog.py`)
```
question → catalog-grounded prompt → LLM proposes SQL → sqlglot GUARD → read-only Postgres → rows
```
- **Catalog** (`catalog.py`): parses `db/schema.sql` (tables, columns, comments) and merges `config/metadata_enrichment.yaml` (business concepts + **Azerbaijani/English synonyms** + roles). This grounds the prompt so the model maps natural language to the right columns in either language.
- **Guard** (`nlsql.guard_sql`): the model only *proposes* SQL. A deterministic `sqlglot` check enforces **single read-only `SELECT`**, **table/column whitelist**, and a **forced `LIMIT`**; execution runs in a **read-only Postgres session**. Writes, DDL, multi-statements, and system-catalog reads are rejected — *security is structural, not prompt-trusted*.

### Task 2 — Classification (`src/rag.py`, `src/eqm.py`, `src/router.py`)
```
item → BGE-M3 retrieval (k similar solved items) → few-shot prompt (+ optimised instructions) → LLM → Good/Service
     → if Good: EQM HS-code retrieval over the enriched registry index → LLM rerank → human review confirms exact code
     → two-tier router decides local vs cloud
```
- **Retrieval (RAG)** — the single biggest accuracy lever (+14 points). For each item we retrieve the *k* most-similar **already-labelled** items and show them to the model as few-shot examples, grounding it on near-identical real cases.
- **Agentic prompt-optimisation** (`scripts/optimize_prompt.py`) — a loop that runs the classifier on dev, collects errors, and asks a stronger model to **rewrite the classifier's own instruction** to fix the error patterns; kept only if it improves on held-out data. The winning instruction is `data/processed/best_instructions.txt`.
- **EQM HS-code** (`eqm.py`) — pure embedding search can't reliably bridge product names → formal HS nomenclature. The current pipeline retrieves from the full enriched EQM registry index (description + brand/synonym keywords), expands the query, uses a wider candidate pool (`k=60`), and asks the LLM to rerank to the exact catalogue code. A Tier-2 auto-resolver can use LLM knowledge and optional privacy-gated web lookup, while the learning loop feeds confirmed decisions back into retrieval.
- **Two-tier router** (`router.py`) — the local model handles confident cases; only **low-confidence** items escalate to a stronger (cloud) model. Cost/quality control built in.

### Evaluation
- **`scripts/eval_task2.py`** / held-out evaluators measure Good/Service accuracy, confusion, HS-code retrieval recall, and HS heading/chapter agreement — on untouched splits and gold sets with no tuning leakage. **`scripts/build_eval_summary.py`** consolidates the numbers into `eval_summary.json`, which the **Evals** tab renders.

---

## Setup

**Prerequisites**
- **Python venv** at `/tmp/azx` (or any venv) with: `pandas openpyxl xlrd psycopg2-binary pyyaml sqlglot fastapi uvicorn httpx numpy openai`.
- **PostgreSQL 16** (Homebrew) with a database named `azdata`.
- **Ollama** running, with the **`bge-m3`** embedding model pulled (`ollama pull bge-m3`).
- **OpenRouter API key** for the strong/cloud model tier, stored at `~/.config/azdata/openrouter.key` (kept outside the repo; never committed).

**Build the data + indexes (one-time)**
```bash
PY=/tmp/azx/bin/python
$PY scripts/ingest.py            # raw e-invoices → Postgres (Task 1)
$PY scripts/prep_task2.py        # labeled set + EQM registry (Task 2)
$PY scripts/make_splits.py       # train/dev/test
$PY src/rag.py  --build          # BGE-M3 train index
$PY src/eqm.py  --build          # BGE-M3 EQM index (~15 min; long HS descriptions)
$PY src/catalog.py               # Task 1 catalog.json
```

---

## Running it

```bash
scripts/run_demo.sh              # preflight-checks everything, then starts the server
# → open http://127.0.0.1:8642/
```
Or directly:
```bash
export OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key)
/tmp/azx/bin/python src/api.py   # http://127.0.0.1:8642/
```

**API** (same origin as the UI):
- `POST /query`   `{question, provider?}` → `{sql, columns, rows, reference_date, …}`
- `POST /classify` `{text}` → `{label, hs_code, hs_description, tier, confidence, …}`
- `GET /evals` · `GET /catalog` · `GET /health`

---

## Testing

```bash
export OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key)
/tmp/azx/bin/python scripts/demo_test.py
```
16 checks: Task 1 NL→SQL (EN + AZ) + 5 SQL-guard security checks; Task 2 classification, EQM HS-code, two-tier router. All pass.

---

## Results

Held-out test set (1,298 items — never used for tuning):

| Metric | Result |
|---|---|
| **Good / Service** | **≈99%** |
| **HS-code retrieval recall@k on hard items** | **≈95%** after index enrichment + query expansion + `k=60` |
| **HS heading vs world-knowledge gold** | **≈68%** (`scripts/gold_build.py`) |
| **HS chapter vs world-knowledge gold** | **≈76%** (`scripts/gold_build.py`) |

*Accuracy was benchmarked via the OpenRouter API on the identical open weights; the model is deployable on-device (24 GB). See [`docs/SOLUTION_REPORT.md`](docs/SOLUTION_REPORT.md).*

> **Important caveat:** exact 10-digit HS-code assignment is harder than deciding Good/Service or retrieving a likely heading/chapter. The system narrows candidates from the full 11,641-code EQM catalogue, but human review confirms the exact catalogue number. Native Azerbaijani product terms classify well in the Good/Service stage.

---

## Limitations

- **Evaluation integrity — found & fixed.** A code audit ([`docs/AUDIT.md`](docs/AUDIT.md)) found the split wasn't deduplicated by text, so eval items leaked verbatim into the train RAG index. **Fixed:** `make_splits.py` now splits *unique texts* and asserts zero train↔eval overlap. Current reporting separates the Good/Service decision from HS-code retrieval and heading/chapter agreement.
- **Exact HS-code difficulty.** Candidate retrieval is strong on hard items (**≈95% recall@k** after enrichment + query expansion + `k=60`), but exact 10-digit catalogue assignment is still hard. The system proposes the code; human review confirms the exact EQM number.
- **Gold-set maturity.** Heading/chapter results are measured against a world-knowledge gold built by `scripts/gold_build.py` (**≈68% heading / ≈76% chapter**). A larger domain-expert-labelled gold set would give a stronger final accuracy claim.
- **Scenario-1 demo data is seeded** (the `1234567890` taxpayer) per the brief — synthetic and illustrative, not a performance claim.

---

## Tech stack & key decisions
- **Python** (FastAPI, psycopg2, sqlglot, numpy) + **vanilla-JS web app** (no build step).
- **Models:** Qwen3.5 family (open-weight) for the LLM; **BGE-M3** for embeddings. Served locally via **Ollama** or via **OpenRouter** (cloud), behind one provider interface.
- **PostgreSQL** for invoice data.
- **Decisions:** local-first for privacy (cloud is an optional escalation tier); retrieval (RAG) over raw model size; an agentic loop that improves its own prompt; safety-by-construction for SQL.

---

## Documentation index
| Doc | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Code & process deep-dive — every module and pipeline, and how to extend them |
| [`docs/SOLUTION_REPORT.md`](docs/SOLUTION_REPORT.md) | Methodology & results narrative (also the in-app Report tab) |
| [`docs/DEMO.md`](docs/DEMO.md) | 5-minute demo walkthrough + talking points |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phases, status, and decisions |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | What changed, newest first |
| [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) | Durable project knowledge (data profiles, environment, research) |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Resume guide / quick-start |
