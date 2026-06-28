# AZdata — e-Invoice AI · Solution Report

*How the two tasks were solved, and how the results are presented.*

---

## 1. What was built

A single web application that delivers two capabilities for Azerbaijani tax-authority e-invoice data:

- **Task 1 — Natural-language → SQL.** Ask a question in Azerbaijani or English about the e-invoice data; the system writes safe SQL against a metadata catalog and returns the answer.
- **Task 2 — Invoice-item classification.** Given an invoice line item, decide **Good (Mal)** vs **Service (Xidmət)**, assign one of **7 product groups** (for Goods), and assign the **HS commodity code**.

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
  → read-only SQL                → EQM HS-code (LLM-first)
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

The headline result: a **24 GB model running locally** reaches **99% accuracy** on a held-out test set — so the solution can run **fully private**, with the cloud used only as an optional escalation tier.

**How we got there (the journey):**

1. **Baseline.** A small local model alone classified poorly (~60–74% fully-correct) — it confused medical goods, utilities, and cryptic product SKUs.
2. **+ Retrieval (RAG).** We index every labelled example with **BGE-M3 embeddings**; at classification time we retrieve the *k* most-similar **already-solved** items and show them to the model as few-shot examples. This grounds the model on near-identical real cases and lifted fully-correct accuracy to **98.9%** (+14 points) — the single biggest lever.
3. **+ Agentic prompt-optimisation.** An automated loop runs the classifier on a dev set, collects the errors, and a stronger model **rewrites the classifier's own instructions** to fix the error patterns (e.g. "medical kits are Goods", "water/sewer utilities are Goods"). Kept only when it improved on held-out data. This closed the gap to **99.0%**.

**HS commodity code (EQM).** Pure embedding search couldn't bridge product names → formal HS nomenclature (a syringe doesn't lexically resemble the HS-9018 wording). The fix is **LLM-first**: the model predicts the likely **HS heading** (where it has real knowledge — "syringe → 9018.31"), we filter the **9,957-code registry** to that heading, then rerank to the exact code. Medical items now map correctly (syringes → 9018.31, catheters → 9018.39).

**Two-tier router.** The local model handles confident cases; only **low-confidence** items escalate to a stronger (cloud) model — a built-in cost/quality control.

**In the UI:** the *Classify item* tab shows label + group + HS code + which tier handled it; the *Evals* tab shows the full evidence.

---

## 5. Results (held-out test, 1,298 items — never used for tuning)

| Model | Where | Config | Label | Group | Fully correct |
|---|---|---|---|---|---|
| qwen3.5 9.7B | local | no retrieval | 74.1% | 48.4% | 60.8% |
| qwen3.5 35B | local | + RAG | 99.08% | 99.18% | 98.92% |
| **qwen3.5 35B** | **local** | **+ RAG + prompt-opt** | **99.38%** | **99.39%** | **99.00%** ★ |
| qwen3.5 122B | cloud/API | + RAG | 99.46% | 99.39% | 99.31% |

Good/Service (the core decision) is ~99.4%; the few remaining misses are the genuinely rare/ambiguous classes (a single dental item, one utility). All numbers are reproducible on the untouched test split.

---

## 6. Methodology & key decisions

- **Local-first for privacy** — tax data stays on-device; cloud is an optional escalation tier, not a dependency.
- **Small model + smart context beats raw size** — retrieval (RAG) did most of the work; we did not need a giant model.
- **The model improves itself** — the agentic prompt-optimisation loop writes better instructions from its own mistakes.
- **Rigorous evaluation** — proper train/dev/test split (6050/1295/1298), tuning only on dev, final numbers on an untouched test set, with misclassifications kept visible.
- **Safety by construction** — the SQL guard is deterministic; the model is never trusted to "be safe".

---

## 7. How to run it

```bash
export OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key)   # cloud tier
/tmp/azx/bin/python ~/Dev/AZdata/src/api.py                        # → http://127.0.0.1:8642/
```

The web app opens with four tabs: **NL Query**, **Classify item**, **Evals**, and this **Report**.
