#!/usr/bin/env bash
# AZdata demo launcher: preflight-check prerequisites, then start the API + web app.
set -u  # NOT pipefail: `... | grep -q` would SIGPIPE the left side and falsely fail checks
ROOT="$HOME/Dev/AZdata"
PY="/tmp/azx/bin/python"
PORT="${AZDATA_API_PORT:-8642}"
KEY="$HOME/.config/azdata/openrouter.key"
ok(){  printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad(){ printf "  \033[31m✗\033[0m %s\n" "$1"; }

echo "AZdata demo — preflight checks"

[ -x "$PY" ] || { bad "venv python missing at $PY"; echo "    recreate: python3 -m venv /tmp/azx && /tmp/azx/bin/pip install pandas openpyxl xlrd psycopg2-binary pyyaml sqlglot fastapi uvicorn httpx numpy openai"; exit 1; }
ok "python venv ($PY)"

if psql -d azdata -tAc "select 1" >/dev/null 2>&1; then ok "Postgres database 'azdata' reachable"
else bad "Postgres/azdata not reachable — start Homebrew Postgres and ensure DB 'azdata' exists"; exit 1; fi

if curl -s --max-time 4 http://localhost:11434/api/tags >/dev/null 2>&1; then ok "Ollama running"
else bad "Ollama not running — run: ollama serve"; exit 1; fi

if ollama list 2>/dev/null | grep -qi 'bge-m3'; then ok "bge-m3 embedding model present"
else bad "bge-m3 missing — run: ollama pull bge-m3"; exit 1; fi

if [ -s "$KEY" ]; then ok "OpenRouter key present"; export OPENROUTER_API_KEY="$(cat "$KEY")"
else bad "OpenRouter key missing at $KEY"; exit 1; fi
# demo: show SQL + detailed errors (production leaves this false)
export AZDATA_DEBUG=true

[ -f "$ROOT/data/processed/train_index.npy" ] || { bad "RAG index missing — run: $PY $ROOT/src/rag.py --build"; exit 1; }
[ -f "$ROOT/data/processed/eqm_index.npy" ]   || { bad "EQM index missing — run: $PY $ROOT/src/eqm.py --build"; exit 1; }
ok "RAG + EQM indexes present"

# stop a previous AZdata server on this port (only if it IS AZdata)
PID="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
if [ -n "${PID:-}" ]; then
  if curl -s --max-time 3 "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q "AZdata"; then
    kill -9 "$PID" 2>/dev/null || true; ok "stopped previous AZdata server on :$PORT"
  else
    bad "port $PORT is used by another app — re-run with: AZDATA_API_PORT=<free port> $0"; exit 1
  fi
fi

cd "$ROOT"
AZDATA_API_PORT="$PORT" nohup "$PY" src/api.py > /tmp/azdata_server.log 2>&1 &
if curl -s --retry 60 --retry-connrefused --retry-delay 1 --max-time 90 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
  ok "server healthy"
  echo ""
  echo "  ▶  Open the demo:   http://127.0.0.1:$PORT/"
  echo "     logs: /tmp/azdata_server.log   stop: pkill -f 'src/api.py'"
  exit 0
else
  bad "server did not become healthy — see /tmp/azdata_server.log"; exit 1
fi
