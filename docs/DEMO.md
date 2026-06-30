# AZdata — Demo Guide

A 5-minute walkthrough of the e-Invoice AI. Everything runs from one web app.

---

## 0. Start it (one command)

```bash
~/Dev/AZdata/scripts/run_demo.sh
```

This checks the prerequisites, starts the server, and prints the URL. Then open:

# → http://127.0.0.1:8642/

**Prerequisites** (the script verifies these):
- **Postgres** running with the `azdata` database (Homebrew Postgres).
- **Ollama** running with `bge-m3` pulled (embeddings) — `ollama serve` + `ollama pull bge-m3`.
- **OpenRouter key** at `~/.config/azdata/openrouter.key` (for the strong/cloud model tier).

If something's off, the script says exactly what.

---

## 1. The story (30-second framing)

> "Two deliverables for the tax authority, in one app: ask the invoice data questions in plain Azerbaijani or English, and automatically classify invoice items as Good vs Service; for every Good, propose the HS commodity code from the full EQM catalogue. Good/Service classification is about **99%**, with exact HS codes confirmed by human review. The open-weight model runs on a single machine — private by default; the cloud is only an optional 'second opinion'."

---

## 2. Walkthrough (click the example chips — no typing needed)

### Tab 1 — **NL Query** (Task 1)
1. Click the **"Turnover · last 4 days (EN)"** chip → it runs *"turnover for the last 4 days for taxpayer 1234567890"*.
   - Point out: the **generated SQL** (transparent/auditable) and the **answer** = **96000**.
2. Click **"Dövriyyə · son 4 gün (AZ)"** → the *same question in Azerbaijani* returns the four daily figures (25000 / 18000 / 31000 / 22000).
   - Talking point: *same engine, both languages; and every query is checked by a safety guard — it can only run read-only SELECTs on whitelisted columns, so it can never modify the tax database.*

### Tab 2 — **Classify item** (Task 2)
3. Click **"Syringe"** → **Good / HS 9018** candidate family (medical instruments), handled by the **local** tier.
4. Click **"Bread"** → **Good / HS 1905**; **"Canned fish"** → **Good / HS 1604**; **"Service"** → **Service** (no HS code).
   - Talking points: the **HS commodity code**, the **full 11,641-code EQM catalogue**, and **which tier** handled it (local vs cloud) — the cost/quality routing is visible.

### Tab 3 — **Evals**
5. Show the **model-comparison chart + table**: a small local model with retrieval hits **99%** (★ best local), matching the big cloud model.
6. Show **the journey** (no-RAG → +RAG → +prompt-opt) and the **misclassified examples** — *we don't hide the errors.*

### Tab 4 — **Report**
7. The full **"how it was done"** write-up — architecture, methods, and results — for anyone who wants the detail.

---

## 3. Key talking points
- **99% accuracy, private:** open-weight 24 GB model + retrieval; no data needs to leave the machine.
- **Retrieval was the unlock** (+14 points) and an **agentic loop** tuned the prompt automatically.
- **Safety is structural:** the SQL guard is deterministic — the model is never trusted to "be safe".
- **Two-tier routing:** cheap local model first, escalate only hard cases to the cloud.

---

## 4. If something fails mid-demo
- App won't load → re-run `scripts/run_demo.sh` (it restarts the server on :8642).
- Classify/NL-query errors → check Ollama is running (`ollama ps`) and the OpenRouter key is valid.
- Switch the NL-query provider dropdown to **Local (Ollama)** to show it works offline (note: the local 9.7B model is weaker than the cloud tier).
- Re-run the full check anytime: `OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key) /tmp/azx/bin/python ~/Dev/AZdata/scripts/demo_test.py`
