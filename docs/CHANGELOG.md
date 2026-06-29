# Changelog — AZdata e-Invoice AI

All notable decisions and changes. Newest first. Dates absolute.

## 2026-06-29
### Shared & hardened
- **Published to GitHub:** `AIShrugged/AZdata` (private) — full code + docs. Added comprehensive `README.md` and `docs/ARCHITECTURE.md` (every module + process explained), plus `.gitignore` and `requirements.txt`; untracked local AI-tooling config.
- **Performance:** cached the catalog (`nlsql.cached_catalog()` — was rebuilt on every `/query`); bounded LRU caches for `/query` + `/classify` (temperature-0 deterministic) → repeated requests in ~ms (`/query` 1.07s → 0.004s on cache hit).
- **Reliability:** `call_llm` wraps every provider call in `_with_retries` (exponential backoff) for transient failures.
- **Observability:** structured per-request logging (latency, cache hit/miss, tier/escalated).
- **CI & tests:** `.github/workflows/ci.yml` runs **12 deterministic unit tests** (`tests/test_unit.py`: SQL guard, catalog parsing, classifier normalize/extract, EQM clean, RAG numerics — no DB/LLM) on every push; CI green.
- **Code audit:** multi-agent audit workflow over the whole repo (6 dimensions, every finding adversarially verified) → `docs/AUDIT.md`.

## 2026-06-28
### Built (Product & demo)
- **Unified API + web app:** `src/api.py` serves `/query` (Task 1), `/classify` (Task 2 full pipeline), `/evals`, `/catalog`, `/health`, and the `web/` UI. `web/index.html` — **4 tabs** (NL Query, Classify, Evals, Report) with one-click example chips, a Copy-SQL button, and an SVG model-comparison chart. Dedicated port **8642** (8000=gemhunter, 8137=Agentic-OS are other projects on the machine).
- **In-app report:** `docs/SOLUTION_REPORT.md` + `web/report.html` (Report tab) — architecture, methods, results, and how results are presented. `scripts/build_eval_summary.py` → `data/processed/eval_summary.json` feeds the Evals dashboard.
- **Demo kit:** `scripts/run_demo.sh` (preflight checks Postgres/Ollama/bge-m3/key/indexes, then launches), `docs/DEMO.md` (5-min walkthrough + talking points).
- **End-to-end test:** `scripts/demo_test.py` — **16 checks all pass** (Task 1 NL→SQL EN+AZ + 5 SQL-guard security checks; Task 2 classification, EQM HS-code, two-tier router).

### Built (Phase 2 — Task 2 backend COMPLETE — classifier 99%, EQM HS-codes, two-tier router)
- **Classifier hits 99% on the held-out test.** `src/classify.py` (Good/Service + 7-group) + `src/rag.py` (BGE-M3 few-shot retrieval over the train split). Local `qwen3.5-35b-a3b` + RAG(k=16) + `scripts/optimize_prompt.py` (agentic prompt-opt, 122B optimizer rewriting instructions from dev errors) = **99.0% fully / 99.4% label** on the full 1298-item held-out test; `qwen3.5-122b-a10b`+RAG = **99.31%**. RAG lifted the deployable 35B from ~85% → 98.9% (+14 pts); the prompt-opt loop closed the rest. Baselines: 9.7B no-RAG 60.8%, gpt-5.5 (batched) 98.2%.
- `src/eqm.py`: EQM HS-code assignment — **LLM-first** (predict HS heading → filter the 9957-code registry to that heading → rerank). Fixes the product↔HS semantic gap that pure embedding retrieval couldn't: şpris→9018.31, kateter→9018.39.
- `src/router.py`: two-tier router (local 35B → escalate to 122B when confidence < 0.9) + full pipeline (classify → HS code for Goods).
- `src/nlsql.py`: added **openrouter** provider (qwen3.5 35B/122B via API; key at `~/.config/azdata/openrouter.key`, gitignored) + reasoning toggle + request timeout.
- `scripts/make_splits.py`: stratified train/dev/test (6050/1295/1298).
- **Headline:** RAG + an agentic prompt-opt loop made a **24 GB local model hit 99%** → fully-local/private deployment is viable; cloud (122B/gpt-5.5) reserved as the escalation tier.

### Built (Phase 2 — Task 2 start)
- `scripts/prep_task2.py` (Codex-authored, Claude-tested) — Task-2 data prep → `data/processed/`: `labeled_items.csv` (8643: Good 6503 / Service 2140), stratified `eval_sample.csv` (166, all 7 groups + services), `eqm_registry.csv` (11,641 HS codes, leading zeros restored to 10-digit, 9957 active).

### Repo
- Merged branch `nettle-fragment` (Task 1 implementation) into `main` and **dropped the branch + its worktree**; all code + docs now live on `main` in `~/Dev/AZdata`. `.mcp.json` (ruflo + **codex**) is now tracked on `main`, so Codex MCP tools load when launching from `~/Dev/AZdata`.

### Built (Phase 1 — Task 1 COMPLETE: ingestion → catalog → NL→SQL → API)
- `src/api.py` (Codex-authored, Claude-tested) — **FastAPI backend** wrapping the engine: `POST /query` → {sql, rows, columns, reference_date}, `GET /health`, `GET /catalog`; CORS for the Phase-3 web app. Verified via TestClient + a live uvicorn server (port 8077): Scenario 1 → 96000 / the 4 daily rows; **guard blocks** DELETE / UPDATE / multi-statement DROP / system-catalog / hallucinated-column, allows legit SELECT (+injected LIMIT). Compat fix (Claude): venv is Python 3.9 → `from __future__ import annotations` + `Optional`/`Any` (avoid 3.10 `X | None` runtime eval).
- `src/nlsql.py` (Codex-authored, Claude-tested + tuned) — **LLM-backed NL→SQL engine**: catalog-grounded prompt → model SQL → `sqlglot` guard (single read-only SELECT, table/column whitelist, LIMIT enforcement) → read-only `psycopg2` session execution. Providers configurable (ollama/openai/anthropic), default local **qwen3.5**. Verified on local Ollama: **Scenario 1 passes in EN + AZ** (daily turnover last 4 days, TIN 1234567890 = 25000/18000/31000/22000); real-data recipient query matches psql (28 / 59183.83). Tuning (Claude): default `think:false` for reasoning models (155s→3.5s) + configurable `AZDATA_LLM_TIMEOUT`.
- `src/catalog.py` (Codex-authored, Claude-tested) — parses `db/schema.sql` (structure + COMMENTs) + merges `config/metadata_enrichment.yaml` → `config/catalog.json`: 2 tables (15+2 cols, roles/default_agg), **99-term normalized AZ/EN synonym index** (e.g. `dövriyyə`/`turnover`→`total_amount`), `by_role`. Stable contract for the NL→SQL engine. Note: bare `VÖEN`/`TIN` resolves to issuer + taxpayer dir; recipient needs the `alıcı` qualifier (ambiguity for the engine to resolve).
- `scripts/ingest.py` (Codex-authored, Claude-tested) — loads `FoodWholesale_sampleData.xlsx` → Postgres `azdata`: **einvoice 3716** (3712 real + 4 demo), **taxpayer 4123**, FK-clean, idempotent (TRUNCATE+reload via `psycopg2`). Seeds demo taxpayer `1234567890` (June 1–4 turnover 25000/18000/31000/22000). **Scenario-1 verified**: turnover last 4 days = 96000.00. Note: 525 TINs act as both supplier & recipient (A_/T_ namespaces overlap).
- `db/schema.sql` (einvoice + taxpayer; business concepts as COMMENTs) **applied to local DB `azdata`**.
- `config/metadata_enrichment.yaml` (business concepts + AZ/EN synonyms + roles).
- Sample data copied into worktree `data/`.
- `docs/HANDOFF.md` resume guide.

### Decided
- Working method: **Codex (MCP) writes code; Claude designs/manages/tests.**
- Cloud escalation tier: **both OpenAI + Anthropic**, configurable.
- Interface: **single web app**, 2 tabs + local-vs-cloud comparison panel.
- Local model: **start with strongest practical multilingual — Qwen3.5 (32B-class, 4-bit MLX)** + **BGE-M3** embeddings; option to scale to ~100B.
- Database: **local Postgres (Docker)** + **Neon** for a shareable demo URL.
- Build order: Task 1 → Task 2 → unified UI.

### Added
- Registered **Codex** as MCP server (`codex mcp-server`) in project `.mcp.json` (logged in via ChatGPT). Its `mcp__codex__*` tools load after a Claude Code restart; until then driven via `codex exec`.
- Python venv (`/tmp/azx`: pandas, openpyxl, xlrd) for Excel parsing.
- Project files: `docs/ROADMAP.md`, `docs/CHANGELOG.md`, `docs/PROJECT_MEMORY.md`.

### Environment (probed)
- Postgres **16.14 (Homebrew)** present; **Docker not installed** → use Homebrew Postgres for local DB (supersedes the earlier Docker note).
- Node v23.10.0 / npm 11.4.2. **Ollama 0.30.7** already has `qwen3.5:latest` (6.6 GB, default tag) and `deepseek-r1:70b` (42 GB). No LM Studio / mlx_lm yet.

### Researched
- Mid-2026 local-LLM landscape: Qwen3.5 (Feb 2026) / Qwen3.6 (Apr 2026), DeepSeek V4, GLM-5.x, Kimi K2.x. M4 Max 128 GB fit: ≥235B too tight; 32B–~120B is the comfortable band. Embeddings: BGE-M3, Qwen3-Embedding.

### Analyzed
- Profiled all 5 data files (see `PROJECT_MEMORY.md`).
- Found Scenario-1 gotcha: Supplier TIN unique per row + Jan–Mar 2026 dates → seed brief's demo taxpayer.

### Constraint
- Do not read `~/Downloads` or `~/Desktop`.
