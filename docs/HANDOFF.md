# AZdata — Resume / Handoff

> Read this first after reopening. Goal: continue with nothing lost.
> Last updated: 2026-06-28

## ▶ DEMO — start here
**To run the demo:** `~/Dev/AZdata/scripts/run_demo.sh` → open **http://127.0.0.1:8642/**.
The app has **4 tabs** (NL Query · Classify item · Evals · Report); every input has **one-click example chips**. Full walkthrough + talking points in **`docs/DEMO.md`**.
Re-run all checks: `OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key) /tmp/azx/bin/python scripts/demo_test.py` (16 checks — all pass). "How it was done" = the **Report** tab + `docs/SOLUTION_REPORT.md`.
*The launcher verifies prereqs: Postgres `azdata`, Ollama + `bge-m3`, OpenRouter key at `~/.config/azdata/openrouter.key`.*

## Where we are
**Phase 1 (Task 1) — ✅ COMPLETE** (NL→SQL engine + catalog + FastAPI; Scenario 1 EN+AZ; guard blocks writes/DDL). **Phase 2 (Task 2) — backend ✅ COMPLETE:** Good/Service + 7-group classifier hits **99.0% fully / 99.4% label on the held-out 1298-item test** (local `qwen3.5-35b-a3b` + BGE-M3 RAG k=16 + agentic prompt-opt; `122b`+RAG = 99.31%); **EQM HS-code** assignment (LLM-first heading → registry filter → rerank; medical items → correct 9018 codes); **two-tier router** + full pipeline working. **Product ✅ COMPLETE & demo-ready:** unified FastAPI (`/query` `/classify` `/evals`) + 4-tab web app (NL Query · Classify · Evals · Report) on **:8642**, with a demo launcher (`scripts/run_demo.sh`), guide (`docs/DEMO.md`), and a 16-check e2e suite (`scripts/demo_test.py`, all pass). **Remaining: Phase 4 packaging polish / optional Neon deploy.**

## Key locations
- **Repo (cwd):** `~/Dev/AZdata` on branch **`main`**. *(The `nettle-fragment` build branch + worktree were merged into `main` and dropped on 2026-06-28 — all code + docs now live on `main`.)*
- **Durable docs:** `~/Dev/AZdata/docs/` → `ROADMAP.md`, `CHANGELOG.md`, `PROJECT_MEMORY.md`, this `HANDOFF.md`.
- **Raw data:** tracked in `~/Dev/AZdata/data/*.xlsx|*.xls` (also copies in `docs/`).
- **DB:** local Postgres 16 (Homebrew, running). Database **`azdata`** — connect: `psql -d azdata`.
- **Excel venv:** `/tmp/azx/bin/python` (pandas/openpyxl/xlrd). *Note: `/tmp` may clear on reboot — recreate with `python3 -m venv /tmp/azx && /tmp/azx/bin/pip install pandas openpyxl xlrd psycopg2-binary pyyaml sqlglot fastapi uvicorn httpx` if missing.*
- **Internal memory:** `~/.claude/projects/-Users-frodobaggins-Dev-AZdata/memory/`.

## Done so far
- Decisions: cloud tier = **both OpenAI+Anthropic (configurable)**; UI = **one web app, 2 tabs + compare**; local model = **start Qwen3.5 32B-class (4-bit MLX) + BGE-M3**; DB = **Homebrew Postgres** (+ Neon later). Method = **Codex writes code; Claude designs/manages/tests.**
- Research recorded in `PROJECT_MEMORY.md` §4 (Qwen3.5/3.6, M4 Max 128 GB fit, embeddings).
- Data profiled (see `PROJECT_MEMORY.md` §2); Scenario-1 unique-TIN gotcha → seed demo taxpayer.
- `db/schema.sql` written + **applied to `azdata`** (tables `einvoice`, `taxpayer`).
- `config/metadata_enrichment.yaml` written (concepts + AZ/EN synonyms + roles).
- Codex registered as MCP server (`codex mcp-server`) in `.mcp.json`; logged in via ChatGPT.

## FIRST thing after reopen
1. **Codex MCP tools**: `.mcp.json` on `main` (`~/Dev/AZdata`) now registers `codex` (+ ruflo). Launching Claude Code from `~/Dev/AZdata` should load `mcp__codex__*` after a restart — check via `ToolSearch("codex")`. Reliable fallback either way: `codex exec -C ~/Dev/AZdata "<spec>"` (CLI authenticated).
2. Quick sanity: `psql -d azdata -c '\dt'` should list `einvoice`, `taxpayer`.

## Next steps (Phase 1 remaining) — Codex implements, Claude tests
1. ✅ **Ingestion** (`scripts/ingest.py`, Codex-authored, idempotent): loaded `data/FoodWholesale_sampleData.xlsx` → `taxpayer` (4123) + `einvoice` (3716); seeded demo taxpayer `1234567890` (June 1–4 totals 25000/18000/31000/22000). Verified: Scenario-1 turnover last 4 days = 96000.00. Re-run: `/tmp/azx/bin/python scripts/ingest.py` (needs `psycopg2-binary` in the venv).
2. ✅ **DDL parser → metadata catalog** (`src/catalog.py`): parses `db/schema.sql` + merges `config/metadata_enrichment.yaml` → `config/catalog.json` (catalog dict + 99-term normalized AZ/EN `synonym_index` + `by_role`). Build/inspect: `/tmp/azx/bin/python src/catalog.py` (needs `pyyaml`).
3. ✅ **NL→SQL engine** (`src/nlsql.py`, LLM-backed): catalog-grounded prompt → LLM SQL → `sqlglot` guard (read-only SELECT, table/col whitelist, LIMIT) + read-only psycopg2 session → rows. Providers configurable (ollama/openai/anthropic); default local **qwen3.5** (`think:false` for speed; needs Ollama up). AZ+EN. Run: `/tmp/azx/bin/python src/nlsql.py "<q>"` (needs `sqlglot`). Ref date = max(einvoice_date) = 2026-06-04.
4. ✅ **API** (`src/api.py`, FastAPI): `POST /query` → {sql, rows, columns, reference_date}, `GET /health`, `GET /catalog`; CORS on; provider selectable per request. Run: `OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key) /tmp/azx/bin/python src/api.py` → **http://127.0.0.1:8642/** (dedicated default port — 8000=gemhunter, 8137=Agentic-OS are other projects on this machine). Serves `/query` `/classify` `/evals` + the web UI (3 tabs). Live-tested.
5. ✅ **Acceptance test:** Scenario 1 reproduced via the engine in **EN + AZ** (4 rows 25000/18000/31000/22000); real-data recipient query cross-checked vs psql (28 / 59183.83).

## Phase 2 (Task 2) — built, how to run
- **Models:** local `qwen3.5:latest`(9.7B) + **bge-m3** embeddings via Ollama; strong models via **OpenRouter** (`qwen/qwen3.5-35b-a3b`, `qwen/qwen3.5-122b-a10b`). Key at `~/.config/azdata/openrouter.key` (chmod 600, outside repo). `src/nlsql.py` has an `openrouter` provider; export `OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key)` before runs.
- **Splits/data:** `scripts/make_splits.py` → `data/processed/{train,dev,test}.csv` (6050/1295/1298). Large derived CSVs + `*_index.npy` are untracked — regenerate.
- **Classifier + RAG:** `src/classify.py`, `src/rag.py`. Build index: `python src/rag.py --build` (→ `train_index.*`). `classify_rag(text, emb, meta, k, provider, model, instructions)`. Optimized prompt: `data/processed/best_instructions.txt`.
- **EQM HS-code:** `src/eqm.py` — `python src/eqm.py --build` (→ `eqm_index.*`, ~15 min). LLM-first: predict heading → filter the 9957-code registry → rerank.
- **Prompt-opt loop:** `scripts/optimize_prompt.py` (agentic; 122B rewrites instructions from dev errors; ~99% on dev).
- **Router / full pipeline:** `python src/router.py --demo` (local 35B → escalate to 122B if conf<0.9 → HS code for Goods).
- **Held-out test (1298):** 35B+RAG+opt = **99.0% fully / 99.4% label**; 122B+RAG = 99.31%. RAG = big lever (+14pts); prompt-opt closed the rest. Venv also needs `numpy openai`.

## Then
- Task 2 **API** (extend `src/api.py` with `/classify`), Phase 3 **web app**, Phase 4 packaging — per `ROADMAP.md`.

## Constraints / standing rules
- Do **not** read `~/Downloads` or `~/Desktop`.
- **Codex writes code; Claude designs/manages/tests.**
- Update `ROADMAP.md` + `CHANGELOG.md` on every change; keep `PROJECT_MEMORY.md` current.
- No `Co-Authored-By` trailer on commits unless project settings enable it.
