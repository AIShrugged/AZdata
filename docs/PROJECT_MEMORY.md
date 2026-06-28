# AZdata — e-Invoice AI · Project Memory

> Durable project knowledge. Keep current alongside `ROADMAP.md` and `CHANGELOG.md`.
> Last updated: 2026-06-28

## 0. Current status (2026-06-28)
**Phase 1 (Task 1) in progress.** Done: research, decisions, all docs, Codex MCP registered, data profiled + copied to worktree `data/`, `db/schema.sql` applied to DB `azdata`, `config/metadata_enrichment.yaml`, **ingestion run (einvoice 3716 / taxpayer 4123, FK-clean; demo TIN `1234567890` Scenario-1 verified 96000.00), metadata catalog built (`src/catalog.py`), LLM-backed NL→SQL engine, and FastAPI backend (`src/api.py`) all built & verified — live HTTP, guard blocks writes/DDL, Scenario 1 EN+AZ. Phase 1 / Task 1 COMPLETE.** **Next:** Phase 2 (Task 2: Good/Service classification). Claude tests. Full resume in `docs/HANDOFF.md`.

## 1. Context
- Client deliverable: **Azerbaijani tax-authority e-invoice AI**, two tasks, presented via one web app.
- Language of item/product text: **Azerbaijani** (some Russian possible).
- Repo: `~/Dev/AZdata` (main). Active build runs in linked worktree `nettle-fragment`.
- Data files: `~/Dev/AZdata/docs/`.
- **Working method:** Codex (MCP) writes code; Claude designs/manages/tests.

## 2. Data profile

### Task 1 — `FoodWholesale_sampleData.xlsx` (e-invoice header, 3712 rows, 1 sheet)
Columns → catalog mapping:
| Column | Business concept |
|--------|------------------|
| `Supplier TIN` (e.g. A_00000001) | Submitting taxpayer (issuer TIN) |
| `Recipient TIN` (e.g. T_00000160) | Receiving taxpayer (recipient TIN) |
| `e-Invoice Date` | Invoice date |
| `e-Invoice Approval Date` | Approval date |
| `e-Invoice Series` (MT26xx) | Series |
| `e-Invoice Number` | Invoice number |
| `Excise Amount` | Excise |
| `Amount of VAT-Taxable Transactions` | VAT-taxable base |
| `Amount of Non-VAT-Taxable Transactions` | Non-VAT-taxable base |
| `Amount of VAT-Exempt Transactions` | VAT-exempt base |
| `Amount of Zero-Rated VAT Transactions` | Zero-rated base |
| `VAT Amount` | VAT |
| `Road Tax` | Road tax |
| `Total Amount` | **Turnover** |

- **GOTCHA:** Supplier TIN unique per row (3712 distinct); dates 2026-01-01…03-31; Recipient TIN repeats (934 distinct).
- **Scenario 1** needs TIN `1234567890` with June daily turnover 25000/18000/31000/22000 → **seed that demo taxpayer**; support real recipient-TIN queries too. Make "last N days" relative to a reference date (today or max(date)).
- No taxpayer-name columns → add a `taxpayer` dim (TIN→name) for "Submitter name"/"Recipient name".

### Task 2 — labeled samples + registry
- `e-invoice_Data_samples_goods.xlsx` (6503 rows): `MƏHSULUN ADI` (product name) → `GROUP` / `Aİ QRUP`. **7 classes**: BAKERY (4743), CANNED FISH (1092), WIPES (465), MED.SYRINGES (134), TOWELS (37), PUBLIC UTILITIES WATER (26), DENTAL MEDICINE (6). Imbalanced. All = Good (Mal).
- `e-invoice_Data_samples_services.xlsx` (2140 rows, header on row 2: `MAL_ADI`, `MAL/XİDMƏT`): all labeled **Xidmət** (Service). No sub-category provided for services.
- `eqm_mal_kodlari-v1.xls` (11,641 rows): the registry — 9–10 digit **HS-style commodity codes**. Cols `CODE`, `ADI` (AZ desc), `VAHID` (unit: –, əd, m2, l, m3…), `STATE` (~9957 active=1). Different taxonomy from the 7 GROUPs.
- Combined goods (Mal) + services (Xidmət) = labeled **Good-vs-Service** training/eval set.

## 3. Architecture (planned)

### Task 1
`SQL DDL → metadata catalog (concept↔data element, AZ/EN synonyms) → NL question → guarded read-only SQL → result (table + chart)`.

### Task 2
`item → [local LLM] Good/Service → if Good: 7-group + EQM HS-code (BGE-M3 retrieval + rerank); router escalates low-confidence to cloud LLM (token-metered) → comparison metrics`.

## 4. Local-model research (mid-2026)
- Qwen line: Qwen3 (Apr 2025, 119 langs) → **Qwen3.5 (Feb 2026)** → **Qwen3.6 (Apr 2026, e.g. 3.6-35B-A3B, Apache-2.0)**. Qwen3 dense ≈ Qwen2.5 of 2× size.
- Open-weight leaders mid-2026: DeepSeek V4 (~1T, too big), GLM-5.x, Kimi K2.x, Qwen3.5 397B.
- **M4 Max 128 GB fit:** ≥235B OOM-tight (DeepSeek V4-Flash 284B ≈140 GB @4-bit — risky); comfortable band 32B–~120B; 70B ≈ 15–18 tok/s via MLX.
- **Chosen start:** Qwen3.5 32B-class (4-bit MLX) + BGE-M3 embeddings; serving via MLX (Apple-native) or Ollama. Revisit size after first accuracy check.

## 5. Decisions log
See `CHANGELOG.md`. Key: cloud=both/configurable; UI=one web app; DB=local Postgres+Neon; method=Codex-writes/Claude-designs-tests.

## 6. Environment
- Codex CLI `0.134.0` at `~/.superset/bin/codex`, logged in (ChatGPT). MCP server: `codex mcp-server` (in `.mcp.json`).
- Python venv: `/tmp/azx` (**Python 3.9**; pandas, openpyxl, xlrd, psycopg2-binary, pyyaml, sqlglot, fastapi, uvicorn, httpx). NOTE: 3.9 → use `from __future__ import annotations` + `Optional`/`Union`, not runtime `X | None`.
- Local LLM box: **M4 Max MacBook Pro, 128 GB** unified memory.
- **Postgres 16.14 (Homebrew)**, `pg_ctl` at `/opt/homebrew/bin`; **Docker not installed** → Homebrew Postgres for local DB.
- **Node v23.10.0 / npm 11.4.2** (web app). **Ollama 0.30.7** with `qwen3.5:latest` (6.6 GB default tag — pull a 32B for the strong tier) and `deepseek-r1:70b` (42 GB) already present.
