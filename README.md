# AZdata — e-Invoice AI

An AI system for Azerbaijani tax-authority **electronic-invoice** data, delivering two capabilities through one web app:

1. **Natural-language → SQL** — ask questions about the invoice data in Azerbaijani or English; the system writes *safe, read-only* SQL and returns the answer.
2. **Invoice-item classification** — given an invoice line item, decide **Good (Mal)** vs **Service (Xidmət)**, assign one of **7 product groups**, and assign the **HS commodity code**.

**Headline result:** the classifier reaches **99.0% fully-correct / 99.4% label accuracy** on a held-out 1,298-item test set using a **24 GB open-weight model** (locally deployable), with the cloud used only as an optional escalation tier. ([details ↓](#results))

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
| **Classify item** | item → **Good/Service** badge + **7-group** + **HS code** + which **tier** handled it |
| **Evals** | model-comparison chart/table, the accuracy **journey**, per-group accuracy, confusion, misclassifications |
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
│  → read-only SQL exec    │   │   → EQM HS-code (eqm.py): LLM-first        │
│                          │   │       heading → registry filter → rerank  │
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
│   ├── classify.py           Task 2: Good/Service + 7-group classifier
│   ├── rag.py                Task 2: BGE-M3 retrieval + few-shot RAG classifier
│   ├── eqm.py                Task 2: EQM HS-code assignment (LLM-first heading → registry → rerank)
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
2. **`scripts/prep_task2.py`** — from the goods/services Excel builds `data/processed/labeled_items.csv` (8,643 items: Good/Service + 7-group) and cleans the EQM registry (`eqm_registry.csv`, 11,641 HS codes; leading zeros restored to 10-digit).
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
item → BGE-M3 retrieval (k similar solved items) → few-shot prompt (+ optimised instructions) → LLM → Good/Service + group
     → if Good: EQM HS-code  → two-tier router decides local vs cloud
```
- **Retrieval (RAG)** — the single biggest accuracy lever (+14 points). For each item we retrieve the *k* most-similar **already-labelled** items and show them to the model as few-shot examples, grounding it on near-identical real cases.
- **Agentic prompt-optimisation** (`scripts/optimize_prompt.py`) — a loop that runs the classifier on dev, collects errors, and asks a stronger model to **rewrite the classifier's own instruction** to fix the error patterns; kept only if it improves on held-out data. The winning instruction is `data/processed/best_instructions.txt`.
- **EQM HS-code** (`eqm.py`) — pure embedding search can't bridge product names → formal HS nomenclature, so the model first predicts the **HS heading** (where it has real knowledge: "syringe → 9018.31"); we filter the 9,957-code registry to that heading and rerank to the exact code.
- **Two-tier router** (`router.py`) — the local model handles confident cases; only **low-confidence** items escalate to a stronger (cloud) model. Cost/quality control built in.

### Evaluation
- **`scripts/eval_task2.py`** / a held-out evaluator measure label/group/fully-correct accuracy, confusion, and per-group recall — on the **untouched test split** (no tuning leakage). **`scripts/build_eval_summary.py`** consolidates the numbers into `eval_summary.json`, which the **Evals** tab renders.

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
- `POST /classify` `{text}` → `{label, group, hs_code, hs_description, tier, confidence, …}`
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

| Model | Where | Config | Label | Group | Fully correct |
|---|---|---|---|---|---|
| qwen3.5 9.7B | local | no retrieval | 74.1% | 48.4% | 60.8% |
| qwen3.5 35B | local | + RAG | 99.08% | 99.18% | 98.92% |
| **qwen3.5 35B** | **local** | **+ RAG + prompt-opt** | **99.38%** | **99.39%** | **99.00%** ★ |
| qwen3.5 122B | cloud/API | + RAG | 99.46% | 99.39% | 99.31% |

*Accuracy was benchmarked via the OpenRouter API on the identical open weights; the model is deployable on-device (24 GB). See [`docs/SOLUTION_REPORT.md`](docs/SOLUTION_REPORT.md).*

> **⚠️ Micro vs macro — read this.** The **group** figures above are **micro-averaged** (per-item) and are dominated by BAKERY (~73% of Goods). The **macro-F1** (per-class average) is **85.5%**, because **DENTAL MEDICINE** is data-starved (6 labelled items total, 1 in the test split) and scores **0%**. The other six groups are **99–100% F1**, and **Good/Service** is genuinely ~99% (macro-F1 99.2%). Read the 7-group result as **micro 99.4% / macro 85.5%**. Mitigations and the taxonomy-scalability caveat are in [Limitations](#limitations).

---

## Limitations

- **Class imbalance — micro vs macro.** The 7-group accuracy is **micro 99.4% / macro-F1 85.5%**. The gap is almost entirely **DENTAL MEDICINE** (n=6 total → unlearnable; 0% F1); the other six groups are 99–100% F1. Even the 122B model shows the same rare-class gap (macro ~82%) — it is the **data imbalance, not the model**. *Mitigations:* gather more labelled data for rare classes, and/or **abstain** ("uncertain / other") below a support/confidence threshold instead of guessing.
- **Fixed taxonomy in the prompt (scalability).** The 7 groups + their hints are injected into **every** classification prompt. Fine for a 7-class brief; it does **not scale to thousands of groups**. The scalable design is **retrieval-based label selection** — exactly what this repo already does for the 11,641 EQM HS codes (retrieve candidates → rerank). The group classifier should adopt the same pattern (the few-shot retrieval already surfaces the relevant candidate groups), making the taxonomy data-driven rather than hardcoded.
- **EQM HS-codes have no ground truth** in the sample data — accuracy is spot-checked, not measured.
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
