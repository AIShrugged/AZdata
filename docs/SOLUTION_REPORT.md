# AZdata — e-Invoice AI · Solution Report

*How the two tasks were solved, and how the results are presented.*

---

## 1. What was built

A single web application that delivers two capabilities for Azerbaijani tax-authority e-invoice data:

- **Task 1 — Natural-language → SQL.** Ask a question in Azerbaijani or English about the e-invoice data; the system writes safe SQL against a metadata catalog and returns the answer.
- **Task 2 — Invoice-item classification.** Given an invoice line item, decide **Good (Mal)** vs **Service (Xidmət)** and, for every Good, assign the **HS commodity code** from the full EQM catalogue (**11,641 active codes**).

Both are served by one backend (FastAPI) and one web app (three working tabs + this report).

---

## 2. Architecture

```
Web app (NL Query · Classify · Evals · Report)
        │  HTTP/JSON
FastAPI  /query  /classify  /evals  /health  /catalog
   │ Task 1                         │ Task 2
NL→SQL engine                  Two-tier router
  catalog-grounded prompt        RAG classify (BGE-M3 few-shot)
  → sqlglot GUARD                → escalate local→cloud if unsure
  → read-only SQL                → EQM HS-code (retrieval + rerank)
   │                                │
Postgres (einvoice, taxpayer)   BGE-M3 vector indexes
        └──────── Model/provider layer ────────┘
        local: Ollama (qwen3.5, bge-m3)
        cloud: OpenRouter (qwen3.5-35b / -122b), OpenAI, Anthropic — configurable
```

A single `call_llm(provider, model, …)` abstracts four providers, so the same code runs **fully offline** or **cloud-backed** by changing one setting.

---

## 3. Task 1 — Natural-language → SQL

**Approach.** A **metadata catalog** is parsed from the database DDL (tables, columns, types, comments) and enriched with **business concepts + Azerbaijani/English synonyms + roles** (measure / dimension / time / id). The catalog grounds the prompt, so the model maps natural language to the right columns. The model only *proposes* SQL.

**Safety is structural, not prompt-trusted.** Every generated query passes a deterministic **`sqlglot` guard** — it must be a single **read-only `SELECT`**, may reference **only whitelisted tables/columns**, and a `LIMIT` is enforced — and it executes inside a **read-only Postgres session**. Writes, DDL, multi-statements, and access to system catalogs are rejected even if the model misbehaves.

**Result.** The brief's Scenario 1 — *"turnover for the last 4 days, taxpayer 1234567890"* — returns the four daily figures (25000 / 18000 / 31000 / 22000 = **96000**), verified in **both English and Azerbaijani**, and cross-checked against the database. The guard was tested to block `DELETE`/`UPDATE`/`DROP`/system-catalog/hallucinated-column inputs.

**In the UI:** the *NL Query* tab shows the generated SQL and the result table.

---

## 4. Task 2 — Invoice-item classification

The headline result: a **24 GB open-weight model** (`qwen3.5-35b-a3b`) reaches about **99% accuracy** on the Good/Service decision. For Goods, HS-code candidate retrieval reaches about **95% recall@k on hard items** after EQM index enrichment, query expansion, and a wider candidate pool. It is **deployable fully on-device** — 24 GB fits the target 128 GB machine — so the solution can run **private**, with the cloud used only as an optional escalation tier.

> *Note on how this was measured:* we benchmarked the model via the **OpenRouter API** for fast iteration. The weights are identical to a local Ollama deployment, so accuracy is a property of the model + method, not of where it runs. The embedding model (BGE-M3) already runs locally. For a fully-offline deployment, the 35B model is downloaded to Ollama (a one-time 24 GB pull); the 9.7B variant is already local.

**How we got there (the journey):**

1. **Baseline.** A small local model alone classified poorly (~60–74% on early held-out runs) — it confused goods with services and struggled with cryptic product SKUs.
2. **+ Retrieval (RAG).** We index every labelled example with **BGE-M3 embeddings**; at classification time we retrieve the *k* most-similar **already-solved** items and show them to the model as few-shot examples. This grounds the model on near-identical real cases and was the single biggest lever.
3. **+ Agentic prompt-optimisation.** An automated loop runs the classifier on a dev set, collects the errors, and a stronger model **rewrites the classifier's own instructions** to fix the error patterns. Kept only when it improved on held-out data. This closed the Good/Service decision to about **99%**.

**HS commodity code (EQM).** Pure embedding search couldn't reliably bridge product names → formal HS nomenclature. The current approach retrieves from the full enriched EQM registry index (**11,641 active codes**, with description + brand/synonym keywords), expands the query, uses a wider candidate pool (`k=60`), and asks the LLM to rerank candidates to the exact catalogue code. A Tier-2 auto-resolver can use LLM knowledge and optional privacy-gated web lookup, while the learning loop feeds confirmed decisions back into retrieval. HS-code retrieval recall@k is about **95% on hard items**; against the world-knowledge gold from `scripts/gold_build.py`, heading agreement is about **68%** and chapter agreement about **76%**. The exact 10-digit code remains hard, so human review confirms the final HS code.

**Two-tier router.** The local model handles confident cases; only **low-confidence** items escalate to a stronger (cloud) model — a built-in cost/quality control.

**In the UI:** the *Classify item* tab shows label + HS code + which tier handled it; the *Evals* tab shows the full evidence.

---

## 5. Results

Current Task 2 validation separates the Good/Service decision from HS-code retrieval and heading/chapter agreement:

| Metric | Result |
|---|---|
| **Good / Service** | **≈99%** |
| **HS-code retrieval recall@k on hard items** | **≈95%** after index enrichment + query expansion + `k=60` |
| **HS heading vs world-knowledge gold** | **≈68%** (`scripts/gold_build.py`) |
| **HS chapter vs world-knowledge gold** | **≈76%** (`scripts/gold_build.py`) |

> **Honest limitation:** exact 10-digit HS-code assignment is the hard part. The system retrieves and reranks likely EQM candidates from the full catalogue, but human review confirms the exact catalogue number. Native Azerbaijani product terms classify well in the Good/Service stage.

---

## 6. Methodology & key decisions

- **Local-first for privacy** — the chosen models are **open-weight and deployable on-device** (tax data need never leave the machine); cloud is an optional escalation tier, not a hard dependency. Accuracy was benchmarked via API on the identical weights.
- **Small model + smart context beats raw size** — retrieval (RAG) did most of the work; we did not need a giant model.
- **The model improves itself** — the agentic prompt-optimisation loop writes better instructions from its own mistakes.
- **Rigorous evaluation** — train/dev/test separation for Good/Service, HS retrieval checks on hard items, and heading/chapter comparison against the `scripts/gold_build.py` gold set, with errors kept visible.
- **Safety by construction** — the SQL guard is deterministic; the model is never trusted to "be safe".

---

## 7. How to run it

```bash
export OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key)   # cloud tier
/tmp/azx/bin/python ~/Dev/AZdata/src/api.py                        # → http://127.0.0.1:8642/
```

The web app opens with four tabs: **NL Query**, **Classify item**, **Evals**, and this **Report**.
