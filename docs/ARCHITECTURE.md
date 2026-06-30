# Architecture & Processes

A developer-level guide to how AZdata is built: every module, every pipeline, the configuration, and how to extend each part. Pairs with the higher-level [`../README.md`](../README.md) and the results narrative in [`SOLUTION_REPORT.md`](SOLUTION_REPORT.md).

---

## 1. Design philosophy

- **One provider interface, four backends.** All LLM calls go through `nlsql.call_llm(system, user, provider, model)`. Swapping local↔cloud is a config change, not a code change.
- **Safety by construction.** The model is never trusted to "be safe". For SQL, a deterministic guard + read-only DB session make unsafe queries impossible regardless of model output.
- **Small model + smart context.** Retrieval (RAG) and an agentic prompt-optimisation loop do more than raw model size — a 24 GB model matches a 122 B one.
- **Reproducible & honest evaluation.** Strict train/dev/test split; tuning only on dev; final numbers on an untouched test set; misclassifications kept visible.

---

## 2. The model / provider layer  (`src/nlsql.py`)

A single function abstracts every LLM call:

```python
call_llm(system: str, user: str, provider: str, model: str) -> str
```

| provider | transport | notes |
|---|---|---|
| `ollama` | HTTP `…/api/chat` (local) | thinking disabled by default (`think:false`) for speed |
| `openrouter` | OpenAI-compatible, `base_url=https://openrouter.ai/api/v1` | reads `OPENROUTER_API_KEY`; `reasoning.enabled` follows the think flag; request timeout |
| `openai` | OpenAI SDK | reads `OPENAI_API_KEY` |
| `anthropic` | Anthropic SDK | reads `ANTHROPIC_API_KEY` |

`DEFAULT_MODELS` maps each provider to a default model. Embeddings are separate (`rag.embed_texts` → Ollama `bge-m3`).

### Configuration (environment variables)
| var | default | meaning |
|---|---|---|
| `AZDATA_LLM_PROVIDER` | `ollama` | default LLM provider |
| `AZDATA_LLM_MODEL` | per-provider | override the model |
| `AZDATA_LLM_THINK` | `false` | enable model "thinking" (slower; off for classification/SQL) |
| `AZDATA_LLM_TIMEOUT` | `120` | per-request timeout (s) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `AZDATA_EMBED_MODEL` | `bge-m3` | embedding model |
| `OPENROUTER_API_KEY` | — | cloud tier (file: `~/.config/azdata/openrouter.key`) |
| `PGHOST/PGPORT/PGDATABASE` | `/tmp` / `5432` / `azdata` | Postgres connection |
| `AZDATA_ROW_LIMIT` | `1000` | enforced SQL LIMIT |
| `AZDATA_STATEMENT_TIMEOUT_MS` | `5000` | Postgres `statement_timeout` |
| `AZDATA_REFERENCE_DATE` | `max(einvoice_date)` | "today" for relative date phrases |
| `AZDATA_API_HOST/PORT` | `127.0.0.1` / `8642` | API bind address |

---

## 3. Task 1 — Natural-language → SQL

### 3.1 Catalog (`src/catalog.py`)
`build_catalog()` produces the structure that grounds the NL→SQL prompt:
1. Parse `db/schema.sql` — `CREATE TABLE` blocks (paren-depth-aware splitting so `numeric(18,2)` isn't broken), column types/nullability/defaults/PKs, and `COMMENT ON …` business descriptions.
2. Merge `config/metadata_enrichment.yaml` — per column: `concept_en`, `concept_az`, `role` (measure/dimension/time/id), `default_agg`, and **synonyms**.
3. Emit: `tables` (full column metadata), a normalised **`synonym_index`** (`casefold`ed term → [{table, column}], 99 terms — the AZ/EN mapping), and `by_role`.
Validation rejects YAML columns that don't exist in the DDL (catches drift). Output cached to `config/catalog.json`.

### 3.2 Engine (`src/nlsql.py`)
`answer(question, provider, model, ref_date) -> dict` runs the pipeline:
1. `reference_date(conn)` — `AZDATA_REFERENCE_DATE` or `max(einvoice_date)`; used so "last N days" resolves against the data.
2. `build_prompt(question, catalog, ref_date)` — a system prompt embedding the catalog (each column: type, EN/AZ concept, role, synonyms), the safety rules, the issuer-vs-recipient default, the relative-date convention, and one worked example.
3. `call_llm(...)` → `extract_sql(text)` strips fences / `<think>` / prose to the bare statement.
4. **`guard_sql(sql, catalog, row_limit)`** (raises `GuardError`):
   - parse with `sqlglot` (postgres); require exactly **one** statement, a `SELECT` (optionally `WITH`);
   - reject any DDL/DML node (Insert/Update/Delete/Drop/Alter/Create/Truncate/Copy/Command/Grant/Set/Transaction);
   - every referenced **table** must be in the catalog whitelist (blocks `pg_catalog`, `information_schema`, …);
   - best-effort **column** whitelist (catches hallucinated columns);
   - **enforce LIMIT** (inject/cap).
5. `execute_readonly(dsn, sql, timeout_ms)` — `conn.set_session(readonly=True)` + `statement_timeout`; returns JSON-safe columns/rows. Defense in depth: even if the guard missed something, the session can't write.
Returns `{question, provider, model, reference_date, sql, raw_sql, columns, rows, row_count, error?}` — never raises.

**Extending:** add columns/synonyms in `metadata_enrichment.yaml` (+ DDL COMMENTs) and rebuild the catalog; the engine immediately understands the new vocabulary in both languages.

---

## 4. Task 2 — Classification

### 4.1 Retrieval + RAG classifier (`src/rag.py`)
- `embed_texts(texts)` → BGE-M3 vectors via Ollama `/api/embed` (batched).
- `build_index(rows, prefix)` / `load_index(prefix)` → `*_index.npy` (float32 matrix, L2-normalised on load) + `*.meta.json`.
- `retrieve(text, emb, meta, k)` → top-k by cosine (float64 matmul, FP flags suppressed).
- `build_rag_prompt(text, examples, instructions=None)` → the classifier instruction (default `DEFAULT_INSTRUCTIONS`, or an optimised override) + the retrieved few-shot examples.
- `classify_rag(text, emb, meta, k, provider, model, instructions) -> {label, confidence, ok}`.

### 4.2 Base classifier (`src/classify.py`)
`classify()` (no retrieval) + shared `extract_json` / `normalize` helpers used by RAG. Useful as a baseline/ablation.

### 4.3 EQM HS-code (`src/eqm.py`) — *retrieval + LLM rerank*
Pure embedding search is not enough because product names often do not resemble formal HS text. The current pipeline:
1. Retrieves candidates from the full enriched EQM registry index (**11,641 active codes**) built from descriptions plus brand/synonym keywords.
2. Expands the query and uses a wider candidate pool (`k=60`) so hard items keep the likely HS code in the candidate set.
3. Reranks candidates with the LLM → the exact 10-digit `code` + confidence. A Tier-2 auto-resolver can use LLM knowledge and optional privacy-gated web lookup, and the learning loop feeds confirmed codes back into retrieval.

### 4.4 Two-tier router & full pipeline (`src/router.py`)
- `classify_route(...)` — run the **local** model; if `ok` and `confidence ≥ threshold` keep it (`tier="local"`), else re-run on the **strong** model (`tier="strong"`, `escalated=True`).
- `classify_item(...)` — `classify_route` then, for Goods, `eqm.assign_code` → `{item, label, hs_code, hs_description, tier, escalated, confidence}`. This is exactly what `POST /classify` returns.
Defaults: `LOCAL=qwen/qwen3.5-35b-a3b`, `STRONG=qwen/qwen3.5-122b-a10b`, `THRESHOLD=0.9`, `K=16`, provider `openrouter`.

---

## 5. The agentic prompt-optimisation loop (`scripts/optimize_prompt.py`)

How the classifier improved from ~94% to 99% **without changing the model**:
```
best = DEFAULT_INSTRUCTIONS
for round in 1..N:
    acc, errors = evaluate(best)                # classify a fixed dev sample, collect misclassifications
    candidate   = propose(best, errors)         # a strong model rewrites the instruction to fix the patterns
    if evaluate(candidate).acc > best.acc: best = candidate   # keep only if it helps
write best → data/processed/best_instructions.txt
```
`propose()` is a meta-prompt: *"here is the current instruction and the items it got wrong — rewrite it to fix the general patterns, don't memorise examples."* The optimiser model is the 122 B; the classifier under test is the 35 B. The winning instruction (medical-kit/utility/repair-vs-part disambiguation rules) is loaded by the API at startup.

---

## 6. Evaluation

- `scripts/eval_task2.py` — runs the classifier over a labelled CSV, prints Good/Service accuracy and the confusion matrix, and saves misclassifications.
- HS-code evaluation reports retrieval recall@k on hard items plus heading/chapter agreement against the world-knowledge gold built by `scripts/gold_build.py`.
- `scripts/build_eval_summary.py` consolidates everything into `data/processed/eval_summary.json` (model comparison, journey, HS-code retrieval evidence, confusion, misclassified examples) which `GET /evals` serves to the **Evals** tab.

---

## 7. The API (`src/api.py`)

FastAPI app. At import it loads the catalog and the RAG + EQM indexes + the optimised instructions into module globals (each in a `try/except` so a missing piece disables only that feature; `/health` reports `task2_ready`). CORS is open (local demo). Endpoints:
| method | path | calls |
|---|---|---|
| POST | `/query` | `nlsql.answer(...)` |
| POST | `/classify` | `router.classify_item(...)` |
| GET | `/evals` | serves `eval_summary.json` |
| GET | `/catalog` | the catalog |
| GET | `/health` | status + `task2_ready` |
| — | `/` and static | serves `web/` (`html=True`) |

---

## 8. The web app (`web/index.html`, `web/report.html`)

A single self-contained page — inline CSS + vanilla JS, **no build step, no external libraries**. Tabs toggle by `data-tab` ↔ pane `id`. Each fetch handler (`postJSON`/`getJSON`) HTML-escapes output and surfaces errors. Features: one-click **example chips** (fill input + submit), **Copy SQL**, an inline-**SVG** model-comparison chart, and the **Evals** dashboard rendered from `/evals`. The **Report** tab embeds `report.html` (a styled render of `SOLUTION_REPORT.md`) in an iframe.

---

## 9. Data & artifacts

| path | tracked? | how to (re)generate |
|---|---|---|
| `data/*.xlsx`, `*.xls` | yes | raw source data |
| `data/processed/labeled_items.csv`, `eqm_registry.csv` | no | `scripts/prep_task2.py` |
| `data/processed/{train,dev,test}.csv` | partial | `scripts/make_splits.py` |
| `data/processed/train_index.*`, `eqm_index.*` | no (large) | `src/rag.py --build`, `src/eqm.py --build` |
| `data/processed/best_instructions.txt` | yes | `scripts/optimize_prompt.py` |
| `data/processed/eval_summary.json` | yes | `scripts/build_eval_summary.py` |
| `config/catalog.json` | yes | `src/catalog.py` |
| Postgres `azdata` (einvoice, taxpayer) | n/a | `scripts/ingest.py` |

Large derived artifacts (vector indexes, big CSVs) are git-ignored and rebuilt locally.

---

## 10. How to extend

- **New invoice query vocabulary (Task 1):** edit `config/metadata_enrichment.yaml` (concepts/synonyms) and/or `db/schema.sql` COMMENTs → `python src/catalog.py`.
- **New HS catalogue coverage (Task 2):** add confirmed examples and synonyms → rebuild `labeled_items.csv`, splits, and the RAG/EQM indexes; optionally re-run `optimize_prompt.py`.
- **New / different model:** set `AZDATA_LLM_PROVIDER` + `AZDATA_LLM_MODEL`, or pass `local_model`/`strong_model` to the router / `provider` to `/query`.
- **Swap the cloud provider:** add a branch in `nlsql.call_llm` (the OpenRouter branch is the template) or just point `openrouter`'s model id elsewhere.
- **Tune cost/quality:** raise/lower the router `threshold` (more escalation = more accuracy + more cost).

---

## 11. Operational notes

- **Ports:** the app defaults to **8642** (other local projects use 8000/8137). Override with `AZDATA_API_PORT`.
- **Services required at run time:** Postgres (`azdata`), Ollama (for BGE-M3 embeddings; and for the `ollama` LLM provider), and the OpenRouter key (for the cloud tier). `scripts/run_demo.sh` verifies all of these.
- **Secrets:** the OpenRouter key lives at `~/.config/azdata/openrouter.key` — **outside** the repo, never committed.
- **The `local` tier in the demo** calls the 35 B via OpenRouter (the same open weights). For a fully-offline deployment, `ollama pull` the 35 B model and point the local tier at it.

## 12. Security & reliability config (audit remediation)

See [`AUDIT.md`](AUDIT.md) for the findings these address. The defaults keep the local demo working with no env set; production sets the security vars.

| var | default | purpose |
|---|---|---|
| `AZDATA_API_KEY` | `""` (off) | if set, `/query` + `/classify` require a matching `X-API-Key` header |
| `AZDATA_RATE_LIMIT` | `60` | requests / 60 s per client (key or IP) → 429 over limit |
| `AZDATA_CORS_ORIGINS` | `127.0.0.1:8642,localhost:8642` | allowed browser origins (was `*`) |
| `AZDATA_ALLOWED_PROVIDERS` / `AZDATA_ALLOWED_MODELS` | openrouter,ollama / the 3 known models | server-side allowlist so clients can't force arbitrary paid models |
| `AZDATA_DEBUG` | `false` | when false, responses omit `raw_sql` and replace internal error text with a correlation id (logged); `run_demo.sh` sets it true |
| `AZDATA_DB_ROLE` | `azdata_ro` | least-privilege role the executor SET ROLEs into (`db/readonly_role.sql`); `""` disables |
| `AZDATA_EMBED_TIMEOUT` / `AZDATA_LLM_RETRIES` | 120 / 3 | embedding request timeout; transient-only retry budget |

- **SQL safety is now layered:** the `guard_sql` function allowlist + schema/system-column rejection, *and* the `azdata_ro` DB role (SELECT on the two tables only, non-superuser) — so a guard bypass still cannot write, read other tables, or use superuser functions.
- **Eval is reproducible & leak-free:** `scripts/make_splits.py` asserts zero train↔eval text overlap; `scripts/eval_task2.py` reports Good/Service accuracy; HS-code checks report retrieval recall@k plus heading/chapter agreement against the world-knowledge gold from `scripts/gold_build.py`.
