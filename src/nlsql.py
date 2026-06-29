from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg2
import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import build_catalog

PROVIDER = os.environ.get("AZDATA_LLM_PROVIDER", "ollama").lower()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ROW_LIMIT = int(os.environ.get("AZDATA_ROW_LIMIT", "1000"))
STMT_TIMEOUT_MS = int(os.environ.get("AZDATA_STATEMENT_TIMEOUT_MS", "5000"))
LLM_TIMEOUT = int(os.environ.get("AZDATA_LLM_TIMEOUT", "120"))
AZDATA_LLM_RETRIES = int(os.environ.get("AZDATA_LLM_RETRIES", "3"))
# Reasoning-capable local models (e.g. qwen3.5) burn minutes "thinking" for simple SQL; off by default.
OLLAMA_THINK = os.environ.get("AZDATA_LLM_THINK", "false").strip().lower() in ("1", "true", "yes", "on")
DSN = f"host={os.environ.get('PGHOST') or '/tmp'} port={os.environ.get('PGPORT') or '5432'} dbname={os.environ.get('PGDATABASE') or 'azdata'}"
DEFAULT_MODELS = {"ollama": "qwen3.5:latest", "openai": "gpt-5.5", "anthropic": "claude-opus-4-8", "openrouter": "qwen/qwen3.5-122b-a10b"}
MODEL_ENV = os.environ.get("AZDATA_LLM_MODEL")
MODEL = MODEL_ENV or DEFAULT_MODELS.get(PROVIDER, DEFAULT_MODELS["ollama"])
_CATALOG_CACHE: dict[str, Any] = {}


class GuardError(Exception):
    pass


def cached_catalog() -> dict[str, Any]:
    if "catalog" not in _CATALOG_CACHE:
        _CATALOG_CACHE["catalog"] = build_catalog()
    return _CATALOG_CACHE["catalog"]


def reference_date(conn: Any) -> dt.date:
    override = os.environ.get("AZDATA_REFERENCE_DATE")
    if override:
        return dt.date.fromisoformat(override)
    with conn.cursor() as cur:
        cur.execute("SELECT max(einvoice_date) FROM einvoice")
        value = cur.fetchone()[0]
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    raise ValueError("Could not determine reference date from einvoice.einvoice_date")


def _schema_block(catalog: dict[str, Any]) -> str:
    lines: list[str] = []
    for table in sorted(catalog["tables"]):
        columns = catalog["tables"][table]["columns"]
        for column in sorted(columns):
            meta = columns[column]
            concept_en = meta.get("concept_en") or "-"
            concept_az = meta.get("concept_az") or "-"
            role = meta.get("role") or "-"
            agg = meta.get("default_agg") or "-"
            line = f"{table}.{column} : {meta.get('type') or '-'} — {concept_en} / {concept_az} [{role}, {agg}]"
            synonyms = [str(s) for s in (meta.get("synonyms") or []) if s]
            if synonyms:
                line += " synonyms: " + ", ".join(synonyms)
            lines.append(line)
    return "\n".join(lines)


def build_prompt(question: str, catalog: dict[str, Any], ref_date: dt.date) -> tuple[str, str]:
    ref = ref_date.isoformat()
    schema = _schema_block(catalog)
    system = f"""You are a careful PostgreSQL analyst for an Azerbaijani e-invoice database.
Output EXACTLY ONE read-only SQL statement and NOTHING else: a single SELECT, optionally with a leading WITH. No prose, no markdown fences, no comments, no semicolon.

Rules:
- Use ONLY the tables and columns listed. Never invent columns. Never write/modify data.
- Questions may be Azerbaijani or English; map words to columns using the provided concepts/synonyms.
- Money/turnover/dövriyyə = einvoice.total_amount. Aggregate measures with their default aggregation (usually SUM) unless the question implies otherwise.
- A bare taxpayer/TIN/VÖEN reference means the ISSUER -> filter supplier_tin. Only filter recipient_tin when the question explicitly says recipient/buyer/customer/alıcı/müştəri.
- Treat the date DATE '{ref}' as "today". For "last N days"/"son N gün": einvoice_date > DATE '{ref}' - INTERVAL 'N days'. For a per-day breakdown, GROUP BY einvoice_date ORDER BY einvoice_date.
- Always include a LIMIT {ROW_LIMIT} or smaller.

Schema:
{schema}

Worked example:
Q: "total VAT collected by issuer A_00000001 in the last 7 days"
SQL: SELECT sum(vat_amount) AS total_vat FROM einvoice WHERE supplier_tin = 'A_00000001' AND einvoice_date > DATE '{ref}' - INTERVAL '7 days' LIMIT {ROW_LIMIT}"""
    return system, question


def _with_retries(fn, attempts=None):
    attempts = attempts or AZDATA_LLM_RETRIES
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # transient: timeouts, 5xx, rate limits, conn errors
            last = exc
            if i == attempts - 1:
                raise
            time.sleep(min(2 ** i, 8))
    raise last


def call_llm(system: str, user: str, provider: str, model: str) -> str:
    provider = provider.lower()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if provider == "ollama":
        payload = {"model": model, "messages": messages, "stream": False, "think": OLLAMA_THINK, "options": {"temperature": 0}}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST.rstrip('/')}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        def request() -> dict[str, Any]:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))

        body = _with_retries(request)
        return str(body["message"]["content"])
    if provider == "openai":
        import openai

        client = openai.OpenAI()
        resp = _with_retries(lambda: client.chat.completions.create(model=model, messages=messages, temperature=0))
        return str(resp.choices[0].message.content)
    if provider == "openrouter":
        import openai

        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"), timeout=LLM_TIMEOUT)
        resp = _with_retries(lambda: client.chat.completions.create(
            model=model, messages=messages, temperature=0,
            extra_body={"reasoning": {"enabled": OLLAMA_THINK}},  # OLLAMA_THINK defaults false -> no runaway thinking
        ))
        return str(resp.choices[0].message.content)
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        resp = _with_retries(lambda: client.messages.create(model=model, max_tokens=1024, system=system, messages=[{"role": "user", "content": user}], temperature=0))
        return str(resp.content[0].text)
    raise ValueError(f"Unsupported provider: {provider}")


def extract_sql(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    if fence:
        cleaned = fence.group(1).strip()
    else:
        match = re.search(r"\b(select|with)\b", cleaned, flags=re.I)
        if match:
            cleaned = cleaned[match.start() :].strip()
    return cleaned.rstrip(" \t\r\n;")


def _select_root(expression: exp.Expression) -> exp.Select | None:
    if isinstance(expression, exp.Select):
        return expression
    if isinstance(expression, exp.With):
        body = expression.this
        return body if isinstance(body, exp.Select) else None
    return None


def _forbidden_classes() -> tuple[type[exp.Expression], ...]:
    names = ("Insert", "Update", "Delete", "Drop", "Alter", "Create", "TruncateTable", "Truncate", "Copy", "Command", "Grant", "Set", "Transaction")
    classes: list[type[exp.Expression]] = []
    for name in names:
        cls = getattr(exp, name, None)
        if isinstance(cls, type):
            classes.append(cls)
    return tuple(classes)


def _alias_names(expression: exp.Expression) -> set[str]:
    aliases: set[str] = set()
    for alias in expression.find_all(exp.Alias):
        if alias.alias:
            aliases.add(alias.alias.casefold())
    for cte in expression.find_all(exp.CTE):
        if cte.alias:
            aliases.add(cte.alias.casefold())
    return aliases


def _catalog_columns(catalog: dict[str, Any]) -> set[str]:
    return {column.casefold() for table in catalog["tables"].values() for column in table["columns"]}


def _literal_limit(select: exp.Select) -> int | None:
    limit = select.args.get("limit")
    if not isinstance(limit, exp.Limit):
        return None
    value = limit.expression
    if isinstance(value, exp.Literal) and value.is_int:
        return int(value.this)
    return None


def guard_sql(sql: str, catalog: dict[str, Any], row_limit: int) -> str:
    try:
        statements = [stmt for stmt in sqlglot.parse(sql, dialect="postgres") if stmt is not None]
    except Exception as exc:
        raise GuardError(f"SQL parse failed: {exc}") from exc
    if len(statements) != 1:
        raise GuardError("Expected exactly one SQL statement")

    expression = statements[0]
    select = _select_root(expression)
    if select is None:
        raise GuardError("Only SELECT statements are allowed")

    forbidden = _forbidden_classes()
    for node in expression.walk():
        if isinstance(node, forbidden):
            raise GuardError(f"Forbidden SQL node: {node.__class__.__name__}")

    allowed_tables = {name.casefold() for name in catalog["tables"]}
    cte_names = {cte.alias.casefold() for cte in expression.find_all(exp.CTE) if cte.alias}
    for table in expression.find_all(exp.Table):
        name = table.name.casefold()
        if name in cte_names:
            continue
        if name not in allowed_tables:
            raise GuardError(f"Table is not allowed: {table.name}")

    known_columns = _catalog_columns(catalog)
    aliases = _alias_names(expression)
    for column in expression.find_all(exp.Column):
        if column.name == "*":
            continue
        name = column.name.casefold()
        if name not in known_columns and name not in aliases and not column.table:
            raise GuardError(f"Column is not in catalog: {column.name}")
        if name not in known_columns and column.table and column.table.casefold() not in aliases:
            raise GuardError(f"Column is not in catalog: {column.sql(dialect='postgres')}")

    current_limit = _literal_limit(select)
    if current_limit is None or current_limit > row_limit:
        select.limit(row_limit, copy=False)
    return expression.sql(dialect="postgres")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def execute_readonly(dsn: str, sql: str, timeout_ms: int) -> tuple[list[str], list[list[Any]]]:
    conn = psycopg2.connect(dsn)
    try:
        conn.set_session(readonly=True, autocommit=False)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = %s", (timeout_ms,))
            cur.execute(sql)
            columns = [desc.name for desc in cur.description]
            rows = [[_json_safe(value) for value in row] for row in cur.fetchall()]
        return columns, rows
    finally:
        try:
            conn.rollback()
        finally:
            conn.close()


def _base_result(question: str, provider: str, model: str, ref_date: dt.date | None) -> dict[str, Any]:
    return {
        "question": question,
        "provider": provider,
        "model": model,
        "reference_date": ref_date.isoformat() if ref_date else None,
        "sql": None,
        "raw_sql": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
    }


def answer(
    question: str,
    provider: str = PROVIDER,
    model: str | None = None,
    ref_date: dt.date | str | None = None,
) -> dict[str, Any]:
    provider = provider.lower()
    chosen_model = model or MODEL_ENV or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["ollama"])
    parsed_ref = dt.date.fromisoformat(ref_date) if isinstance(ref_date, str) else ref_date
    result = _base_result(question, provider, chosen_model, parsed_ref)
    try:
        catalog = cached_catalog()
        if parsed_ref is None:
            conn = psycopg2.connect(DSN)
            try:
                parsed_ref = reference_date(conn)
            finally:
                conn.close()
            result["reference_date"] = parsed_ref.isoformat()
        system, user = build_prompt(question, catalog, parsed_ref)
        raw_text = call_llm(system, user, provider, chosen_model)
        raw_sql = extract_sql(raw_text)
        result["raw_sql"] = raw_sql
        guarded_sql = guard_sql(raw_sql, catalog, ROW_LIMIT)
        result["sql"] = guarded_sql
        columns, rows = execute_readonly(DSN, guarded_sql, STMT_TIMEOUT_MS)
        result["columns"] = columns
        result["rows"] = rows
        result["row_count"] = len(rows)
    except Exception as exc:
        result["error"] = f"{exc.__class__.__name__}: {exc}"
    return result


def _print_table(columns: list[str], rows: list[list[Any]]) -> None:
    if not columns:
        print("(no columns)")
        return
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    print(" | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural-language SQL for the AZ e-invoice database")
    parser.add_argument("question")
    parser.add_argument("--provider", default=PROVIDER, choices=sorted(DEFAULT_MODELS))
    parser.add_argument("--model")
    parser.add_argument("--ref-date")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = answer(args.question, provider=args.provider, model=args.model, ref_date=args.ref_date)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"provider/model: {result['provider']} / {result['model']}")
    print(f"reference_date: {result['reference_date']}")
    if result.get("sql"):
        print("sql:")
        print(result["sql"])
    if result.get("error"):
        print(f"error: {result['error']}")
        return
    _print_table(result["columns"], result["rows"])


if __name__ == "__main__":
    main()
