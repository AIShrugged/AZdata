"""End-to-end test / demo rehearsal for AZdata.
Exercises Task 1 (NL->SQL + safety guard) and Task 2 (classify + EQM HS-code + two-tier router)
through the core modules. Requires: Postgres `azdata`, Ollama (bge-m3), OPENROUTER_API_KEY.
Run:  OPENROUTER_API_KEY=$(cat ~/.config/azdata/openrouter.key) /tmp/azx/bin/python scripts/demo_test.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OPENROUTER_API_KEY", (Path.home() / ".config/azdata/openrouter.key").read_text(encoding="utf-8").strip())

import nlsql, rag, eqm, router
from nlsql import guard_sql, GuardError, build_catalog

PROVIDER = "openrouter"
STRONG = "qwen/qwen3.5-122b-a10b"
LOCAL = "qwen/qwen3.5-35b-a3b"

passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


def numbers(rows) -> list:
    return [x for row in (rows or []) for x in row if isinstance(x, (int, float))]


print("== Task 1 · NL->SQL ==")
r_en = nlsql.answer("turnover for the last 4 days for taxpayer 1234567890", provider=PROVIDER, model=STRONG)
check("Scenario 1 (EN) totals 96000", (96000 in numbers(r_en.get("rows"))) or (round(sum(numbers(r_en.get("rows"))), 2) == 96000.0), f"sql={r_en.get('sql')}")
check("Scenario 1 (EN) no error", not r_en.get("error"), str(r_en.get("error")))
r_az = nlsql.answer("son 4 günün gündəlik dövriyyəsi, vergi ödəyicisi 1234567890", provider=PROVIDER, model=STRONG)
check("Scenario 1 (AZ) returns 4 daily rows", len(r_az.get("rows") or []) == 4, f"rows={r_az.get('rows')}")

print("== Task 1 · SQL guard (security) ==")
cat = build_catalog()
def blocked(sql: str) -> bool:
    try:
        guard_sql(sql, cat, 1000); return False
    except GuardError:
        return True
check("blocks DELETE", blocked("DELETE FROM einvoice"))
check("blocks UPDATE", blocked("UPDATE einvoice SET total_amount = 0"))
check("blocks multi-statement DROP", blocked("SELECT 1 FROM einvoice; DROP TABLE taxpayer"))
check("blocks system-catalog read", blocked("SELECT * FROM pg_user"))
check("allows a legit SELECT", not blocked("SELECT total_amount FROM einvoice LIMIT 5"))

print("== Task 2 · classification (RAG) ==")
emb, meta = rag.load_index(ROOT / "data/processed/train_index")
instr = (ROOT / "data/processed/best_instructions.txt").read_text(encoding="utf-8")
cases = [
    ("Syringe -> Good", "Şpris 10 ml 21GХ38mm 3H rezin porşenli (Şpris) (BandBraumann)", "Good"),
    ("Bread -> Good", "Çörək tam buğda 500q", "Good"),
    ("Canned fish -> Good", "Konserv tunes balığı 185q", "Good"),
    ("Repair service -> Service", "Texniki təmir xidmətləri", "Service"),
    ("Transport service -> Service", "Yüklərin nəqliyyatla daşınması 30 km", "Service"),
]
for name, text, expected in cases:
    res = rag.classify_rag(text, emb, meta, k=16, provider=PROVIDER, model=LOCAL, instructions=instr)
    check(name, res.get("label") == expected, f"got {res.get('label')}")

print("== Task 2 · EQM HS-code ==")
e_emb, e_meta = eqm.load_eqm_index()
hs = eqm.assign_code("Şpris 10 ml rezin porşenli", e_emb, e_meta, k=20, provider=PROVIDER, model=STRONG)
check("syringe -> HS heading 9018", str(hs.get("code")).startswith("9018"), f"got {hs.get('code')}")

print("== Task 2 · two-tier router (full pipeline) ==")
res = router.classify_item("Konserv balıq 185q", emb, meta, e_emb, e_meta, instr, assign_hs=True)
check("router: Good + HS code assigned", res.get("label") == "Good" and bool(res.get("hs_code")), str(res))
check("router: reports a tier", res.get("tier") in ("local", "strong"), f"tier={res.get('tier')}")

print(f"\n=== RESULT: {passed} passed, {failed} failed ===")
sys.exit(0 if failed == 0 else 1)
