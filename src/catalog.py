from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "db" / "schema.sql"
ENRICH = ROOT / "config" / "metadata_enrichment.yaml"
OUT = ROOT / "config" / "catalog.json"

TABLE_CONSTRAINTS = {"CONSTRAINT", "PRIMARY", "FOREIGN", "UNIQUE", "CHECK"}
COLUMN_CONSTRAINTS = TABLE_CONSTRAINTS | {"REFERENCES", "NOT", "NULL", "DEFAULT"}
ROLES = ("measure", "dimension", "time", "id")


def normalize_term(term: str) -> str:
    return term.strip().casefold()


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = depth = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            if part := text[start:i].strip():
                parts.append(part)
            start = i + 1
    if part := text[start:].strip():
        parts.append(part)
    return parts


def extract_create_blocks(sql: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*\(", re.I)
    blocks: list[tuple[str, str]] = []
    for match in pattern.finditer(sql):
        table, start = match.group(1), match.end()
        depth, i = 1, start
        while i < len(sql) and depth:
            if sql[i] == "(":
                depth += 1
            elif sql[i] == ")":
                depth -= 1
            i += 1
        if depth:
            raise ValueError(f"Unclosed CREATE TABLE body for {table}")
        blocks.append((table, sql[start : i - 1]))
    return blocks


def parenthesized_columns(definition: str) -> list[str]:
    match = re.search(r"\(([^)]*)\)", definition)
    return [] if not match else [c.strip() for c in match.group(1).split(",") if c.strip()]


def parse_default(definition: str) -> str | None:
    match = re.search(r"\bDEFAULT\b\s+(.+)$", definition, re.I | re.S)
    if not match:
        return None
    value = match.group(1).strip()
    stop = re.search(r"\s+\b(?:PRIMARY|FOREIGN|UNIQUE|CHECK|REFERENCES|NOT|NULL|CONSTRAINT)\b", value, re.I)
    value = value[: stop.start()].strip() if stop else value
    return value or None


def parse_column_definition(definition: str) -> tuple[str, dict[str, Any], bool]:
    tokens = definition.split()
    if not tokens:
        raise ValueError("Empty column definition")
    type_tokens: list[str] = []
    for token in tokens[1:]:
        if token.upper() in COLUMN_CONSTRAINTS:
            break
        type_tokens.append(token)
    upper = definition.upper()
    column = {"type": " ".join(type_tokens), "nullable": "NOT NULL" not in upper}
    column.update({"default": parse_default(definition), "comment": None})
    return tokens[0], column, "PRIMARY KEY" in upper


def parse_tables(sql: str) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for table, body in extract_create_blocks(sql):
        columns: dict[str, dict[str, Any]] = {}
        primary_key: list[str] = []
        for definition in split_top_level_commas(body):
            first = definition.split(maxsplit=1)[0].upper()
            if first in TABLE_CONSTRAINTS:
                if re.search(r"\bPRIMARY\s+KEY\b", definition, re.I):
                    primary_key.extend(parenthesized_columns(definition))
                continue
            name, column, inline_pk = parse_column_definition(definition)
            columns[name] = column
            if inline_pk:
                primary_key.append(name)
        tables[table] = {
            "comment": None, "description_en": None, "description_az": None,
            "primary_key": primary_key,
            "columns": columns,
        }
    return tables


def parse_comments(sql: str) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    table_re = r"COMMENT\s+ON\s+TABLE\s+(\w+)\s+IS\s+'([^']*)'\s*;"
    column_re = r"COMMENT\s+ON\s+COLUMN\s+(\w+)\.(\w+)\s+IS\s+'([^']*)'\s*;"
    table_comments = {m.group(1): m.group(2) for m in re.finditer(table_re, sql, re.I)}
    column_comments = {(m.group(1), m.group(2)): m.group(3) for m in re.finditer(column_re, sql, re.I)}
    return table_comments, column_comments


def attach_comments(tables: dict[str, dict[str, Any]], sql: str) -> None:
    table_comments, column_comments = parse_comments(sql)
    for table, comment in table_comments.items():
        if table in tables:
            tables[table]["comment"] = comment
    for (table, column), comment in column_comments.items():
        if table in tables and column in tables[table]["columns"]:
            tables[table]["columns"][column]["comment"] = comment


def merge_enrichment(tables: dict[str, dict[str, Any]], enrichment: dict[str, Any] | None) -> None:
    enrichment = enrichment or {}
    offenders = [
        f"{table}.{column}"
        for table, data in enrichment.items()
        for column in ((data or {}).get("columns") or {})
        if table not in tables or column not in tables[table]["columns"]
    ]
    if offenders:
        raise ValueError("Enrichment columns missing from DDL: " + ", ".join(offenders))

    empty = {"concept_en": None, "concept_az": None, "role": None, "default_agg": None, "synonyms": []}
    for table_data in tables.values():
        for column_data in table_data["columns"].values():
            column_data.update(empty.copy())

    for table, table_enrichment in enrichment.items():
        if table not in tables:
            continue
        tables[table]["description_en"] = table_enrichment.get("description_en")
        tables[table]["description_az"] = table_enrichment.get("description_az")
        for column, column_enrichment in (table_enrichment.get("columns") or {}).items():
            tables[table]["columns"][column].update({
                "concept_en": column_enrichment.get("concept_en"),
                "concept_az": column_enrichment.get("concept_az"),
                "role": column_enrichment.get("role"),
                "default_agg": column_enrichment.get("default_agg"),
                "synonyms": list(column_enrichment.get("synonyms") or []),
            })


def build_synonym_index(tables: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    for table, table_data in tables.items():
        for column, column_data in table_data["columns"].items():
            terms = column_data["synonyms"] + [column_data["concept_en"], column_data["concept_az"], column]
            for term in terms:
                normalized = normalize_term(str(term)) if term else ""
                pair = (table, column)
                if not normalized or pair in seen.setdefault(normalized, set()):
                    continue
                seen[normalized].add(pair)
                index.setdefault(normalized, []).append({"table": table, "column": column})
    return {term: index[term] for term in sorted(index)}


def build_by_role(tables: dict[str, dict[str, Any]]) -> dict[str, list[list[str]]]:
    by_role: dict[str, list[list[str]]] = {role: [] for role in ROLES}
    for table, table_data in tables.items():
        for column, column_data in table_data["columns"].items():
            if column_data.get("role") in by_role:
                by_role[column_data["role"]].append([table, column])
    return by_role


def build_catalog(schema_path: Path = SCHEMA, enrich_path: Path = ENRICH) -> dict[str, Any]:
    sql = schema_path.read_text(encoding="utf-8")
    tables = parse_tables(sql)
    attach_comments(tables, sql)
    merge_enrichment(tables, yaml.safe_load(enrich_path.read_text(encoding="utf-8")))
    return {
        "tables": tables, "synonym_index": build_synonym_index(tables), "by_role": build_by_role(tables),
    }


def main() -> None:
    catalog = build_catalog()
    OUT.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tables = catalog["tables"]
    print("tables: " + ", ".join(tables))
    for table, data in tables.items():
        print(f"{table}: {len(data['columns'])} columns, pk={data['primary_key']}")
    index = catalog["synonym_index"]
    print(f"synonym_index terms: {len(index)}")
    for term in ("turnover", "dövriyyə", "alıcı", "VÖEN", "vat", "invoice date"):
        normalized = normalize_term(term)
        print(f"{normalized}: {index.get(normalized, [])}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
