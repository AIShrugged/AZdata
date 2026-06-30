import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

import eqm
import rag
from catalog import build_catalog
from classify import extract_json, normalize
from nlsql import GuardError, extract_sql, guard_sql


CAT = build_catalog()


def test_guard_blocks_delete():
    with pytest.raises(GuardError):
        guard_sql("DELETE FROM einvoice", CAT, 1000)


def test_guard_blocks_update():
    with pytest.raises(GuardError):
        guard_sql("UPDATE einvoice SET total_amount = 0", CAT, 1000)


def test_guard_blocks_multistatement():
    with pytest.raises(GuardError):
        guard_sql("SELECT 1 FROM einvoice; DROP TABLE taxpayer", CAT, 1000)


def test_guard_blocks_system_catalog():
    with pytest.raises(GuardError):
        guard_sql("SELECT * FROM pg_user", CAT, 1000)


def test_guard_allows_select_and_injects_limit():
    sql = guard_sql("SELECT total_amount FROM einvoice", CAT, 1000)
    assert "LIMIT" in sql.upper()


@pytest.mark.parametrize("attack", [
    "SELECT database_to_xml(true, false, '')",                       # whole-DB exfiltration
    "SELECT query_to_xml('SELECT * FROM taxpayer', true, false, '')",
    "SELECT pg_read_file('/etc/passwd')",                            # arbitrary file read
    "SELECT lo_import('/etc/passwd')",
    "SELECT total_amount FROM einvoice WHERE pg_sleep(10) IS NULL",  # DoS
    "SELECT total_amount FROM einvoice WHERE supplier_tin = dblink('h', 'SELECT 1')",
    "SELECT total_amount FROM secret_schema.einvoice",               # schema-qualifier bypass
    "SELECT ctid FROM einvoice",                                     # system column
])
def test_guard_blocks_dangerous(attack):
    with pytest.raises(GuardError):
        guard_sql(attack, CAT, 1000)


def test_guard_allows_aggregates_and_dates():
    # Legitimate analytics SQL (aggregates, date_trunc, date arithmetic) must NOT be blocked.
    guard_sql(
        "SELECT date_trunc('month', einvoice_date) m, sum(total_amount) FROM einvoice "
        "WHERE supplier_tin = '1' AND einvoice_date > DATE '2026-01-04' - INTERVAL '4 days' GROUP BY m",
        CAT, 1000,
    )
    guard_sql("SELECT recipient_tin, round(avg(total_amount), 2) FROM einvoice GROUP BY recipient_tin", CAT, 1000)


def test_catalog_tables():
    assert set(CAT["tables"]) == {"einvoice", "taxpayer"}
    assert len(CAT["tables"]["einvoice"]["columns"]) == 15
    assert len(CAT["tables"]["taxpayer"]["columns"]) == 2


def test_catalog_synonyms():
    for term in ("turnover", "dövriyyə"):
        matches = CAT["synonym_index"][term.strip().casefold()]
        assert {"table": "einvoice", "column": "total_amount"} in matches


def test_normalize_label_aliases():
    assert normalize({"label": "Mal", "confidence": 0.9})["label"] == "Good"
    assert normalize({"label": "Xidmət"})["label"] == "Service"


def test_extract_json_fenced():
    data = extract_json('```json\n{"label":"Good","confidence":0.8}\n```')
    assert data["label"] == "Good"


def test_extract_sql_fenced():
    assert extract_sql("```sql\nSELECT 1 FROM einvoice\n```") == "SELECT 1 FROM einvoice"


def test_extract_sql_strips_think_block():
    assert extract_sql("<think>reasoning</think>\nSELECT 1 FROM einvoice") == "SELECT 1 FROM einvoice"


def test_extract_sql_strips_leading_prose():
    assert extract_sql("Here is the query:\nSELECT 1 FROM einvoice").startswith("SELECT")


def test_eqm_clean_query():
    cleaned = eqm._clean_query("Şpris 10 ml 21GХ38mm (Şpris) (B.Braun)")
    assert "Şpris" in cleaned
    assert "(" not in cleaned
    assert "10" not in cleaned.split()


def test_l2norm_handles_zero_rows():
    normalized = rag._l2norm(np.zeros((2, 4), dtype=np.float32))
    assert normalized.shape == (2, 4)
    assert np.isfinite(normalized).all()
