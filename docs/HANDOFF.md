# AZdata — Resume / Handoff

> Read this first after reopening. Goal: continue with nothing lost.
> Last updated: 2026-06-28

## Where we are
**Phase 1 (Task 1: NL→SQL + metadata catalog) — ✅ COMPLETE.** Ingestion, metadata catalog, the LLM-backed NL→SQL engine, and the FastAPI backend are all built + verified — Scenario 1 passes in EN + AZ, the guard blocks writes/DDL, and a live uvicorn server serves `/query`. **Next: Phase 2 (Task 2 — Good vs Service classification).**

## Key locations
- **Build worktree (cwd):** `~/.superset/worktrees/5846a0d3-f32a-4c9a-8c54-617d348faf0f/nettle-fragment` (git branch `nettle-fragment`).
- **Durable docs:** `~/Dev/AZdata/docs/` → `ROADMAP.md`, `CHANGELOG.md`, `PROJECT_MEMORY.md`, this `HANDOFF.md`.
- **Raw data:** `~/Dev/AZdata/docs/*.xlsx|*.xls`; copies in worktree `data/`.
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
1. **Codex MCP tools** load ONLY when Claude Code is launched from the **`nettle-fragment` worktree** (its `.mcp.json` registers `codex`). From the `main` worktree (`~/Dev/AZdata`, whose `.mcp.json` has only ruflo) they will NOT appear — confirmed 2026-06-28. Reliable fallback (in use): `codex exec -C <nettle-fragment> "<spec>"` (CLI authenticated). `ToolSearch("codex")` to check.
2. Quick sanity: `psql -d azdata -c '\dt'` should list `einvoice`, `taxpayer`.

## Next steps (Phase 1 remaining) — Codex implements, Claude tests
1. ✅ **Ingestion** (`scripts/ingest.py`, Codex-authored, idempotent): loaded `data/FoodWholesale_sampleData.xlsx` → `taxpayer` (4123) + `einvoice` (3716); seeded demo taxpayer `1234567890` (June 1–4 totals 25000/18000/31000/22000). Verified: Scenario-1 turnover last 4 days = 96000.00. Re-run: `/tmp/azx/bin/python scripts/ingest.py` (needs `psycopg2-binary` in the venv).
2. ✅ **DDL parser → metadata catalog** (`src/catalog.py`): parses `db/schema.sql` + merges `config/metadata_enrichment.yaml` → `config/catalog.json` (catalog dict + 99-term normalized AZ/EN `synonym_index` + `by_role`). Build/inspect: `/tmp/azx/bin/python src/catalog.py` (needs `pyyaml`).
3. ✅ **NL→SQL engine** (`src/nlsql.py`, LLM-backed): catalog-grounded prompt → LLM SQL → `sqlglot` guard (read-only SELECT, table/col whitelist, LIMIT) + read-only psycopg2 session → rows. Providers configurable (ollama/openai/anthropic); default local **qwen3.5** (`think:false` for speed; needs Ollama up). AZ+EN. Run: `/tmp/azx/bin/python src/nlsql.py "<q>"` (needs `sqlglot`). Ref date = max(einvoice_date) = 2026-06-04.
4. ✅ **API** (`src/api.py`, FastAPI): `POST /query` → {sql, rows, columns, reference_date}, `GET /health`, `GET /catalog`; CORS on; provider selectable per request. Run: `/tmp/azx/bin/python src/api.py` (port 8000; needs `fastapi uvicorn`). Live-tested.
5. ✅ **Acceptance test:** Scenario 1 reproduced via the engine in **EN + AZ** (4 rows 25000/18000/31000/22000); real-data recipient query cross-checked vs psql (28 / 59183.83).

## Then
- Phase 2 (Task 2 classification, two-tier) and Phase 3 (web app) per `ROADMAP.md`.

## Constraints / standing rules
- Do **not** read `~/Downloads` or `~/Desktop`.
- **Codex writes code; Claude designs/manages/tests.**
- Update `ROADMAP.md` + `CHANGELOG.md` on every change; keep `PROJECT_MEMORY.md` current.
- No `Co-Authored-By` trailer on commits unless project settings enable it.
