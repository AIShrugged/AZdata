# AZdata — e-Invoice AI · Roadmap

> Living document. Update on every change. Companion files: `CHANGELOG.md`, `PROJECT_MEMORY.md`.
> Last updated: 2026-06-28

## Overview
Two-task client deliverable for an **Azerbaijani tax-authority e-invoice** analytics system,
presented through a single web app.

- **Task 1** — Natural-language → SQL engine with a **metadata catalog** over e-invoice data in PostgreSQL.
- **Task 2** — **Two-tier** (local LLM + cloud escalation) classification: Good vs Service, then category.
- **Interface** — one web app, two tabs + a local-vs-cloud comparison panel.

## Working method
- **Codex (MCP) writes implementation code; Claude designs, manages, and tests.**
- Every change is recorded in `CHANGELOG.md`; this roadmap and `PROJECT_MEMORY.md` are kept current.
- Do **not** read `~/Downloads` or `~/Desktop`.

## Locked decisions (2026-06-28)
| Topic | Decision |
|------|----------|
| Cloud escalation tier | **Both** OpenAI + Anthropic, configurable router |
| Local model (start) | **Qwen3.5 (32B-class, 4-bit MLX)** + **BGE-M3** embeddings; scale to ~100B only if needed |
| Database | Local **Postgres 16 (Homebrew)** for build/demo — Docker not installed; **Neon** for shareable URL |
| Interface | Single web app, 2 tabs + comparison panel |
| Build order | Task 1 end-to-end first, then Task 2, then unify UI |

## Phases

### Phase 0 — Setup & data understanding — ✅ done
- venv for Excel parsing; Codex registered as MCP; all 5 data files profiled.
- Identified Scenario-1 gotcha (unique Supplier TINs → seed demo taxpayer).

### Phase 1 — Task 1: NL→SQL + metadata catalog — ✅ done
1. ✅ Postgres DDL `db/schema.sql` (einvoice 15 cols + taxpayer; concepts as COMMENTs) — applied to DB `azdata`.
2. ✅ Ingestion (`scripts/ingest.py`, idempotent): `FoodWholesale_sampleData.xlsx` → Postgres — **einvoice 3716, taxpayer 4123, FK-clean**; seeded demo taxpayer `1234567890` (June 1–4 turnover 25000/18000/31000/22000). Scenario-1 data verified (turnover last 4 days = 96000.00).
3. ✅ **SQL-DDL parser → metadata catalog** (`src/catalog.py` → `config/catalog.json`): parses `db/schema.sql` (structure + COMMENTs) + merges `config/metadata_enrichment.yaml`; 15+2 columns, roles/default_agg, **99-term normalized AZ/EN synonym index** (`dövriyyə`/`turnover`→`total_amount`).
4. ✅ **NL→SQL engine** (`src/nlsql.py`, LLM-backed): catalog-grounded prompt → LLM SQL → `sqlglot` safety guard (one read-only SELECT, table/column whitelist, forced LIMIT) + read-only `psycopg2` session → execute. Providers configurable (ollama/openai/anthropic); default local **qwen3.5** (`think:false`). AZ + EN.
5. ✅ Acceptance test: Scenario 1 reproduced in **EN + AZ** (daily turnover last 4 days, TIN 1234567890 → 25000/18000/31000/22000); real-data recipient query cross-checked vs psql (28 invoices / 59183.83).
6. ✅ **Backend API** (`src/api.py`, FastAPI): `POST /query` → {sql, rows, columns, reference_date}, `GET /health`, `GET /catalog`; CORS for the web app; provider selectable per request. Guard verified (DELETE/UPDATE/multi-statement DROP/system-catalog/hallucinated-column all blocked); live HTTP smoke-tested.

### Phase 2 — Task 2: two-tier classification — 🔄 in progress
1. ✅ Data prep (`scripts/prep_task2.py` → `data/processed/`): unified `labeled_items.csv` (8643: Good 6503 / Service 2140), stratified `eval_sample.csv` (166, all 7 groups + services), clean `eqm_registry.csv` (11,641 HS codes, leading zeros restored to 10-digit, 9957 active).
2. Local serving (MLX/Ollama) + prompts: Good/Service, then 7-group classification.
3. EQM HS-code assignment: BGE-M3 retrieval + LLM rerank.
4. Two-tier router: confidence threshold → cloud escalation, **token metering**.
5. Eval harness: accuracy / latency / cost; **local vs cloud** comparison.
6. Backend API endpoints.

### Phase 3 — Web app — ⬜
- Tab 1: NL query (Task 1) → table + chart.
- Tab 2: classify invoice item (Task 2) → Good/Service + group + EQM code + which tier handled it.
- Comparison panel: local vs cloud metrics.
- Optional Neon deployment.

### Phase 4 — Packaging & client demo — ⬜
- Demo script, seeded scenarios, short docs.

## Open items to revisit
- Final local SKU (32B vs ~100B) after first accuracy check.
- Services sub-category registry (data provides only Good/Service for services).
- Taxpayer names absent in sample (TIN-only) — synthesize a dim table for name lookups.
