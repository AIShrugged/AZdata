import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

import eqm
import rag
from catalog import build_catalog
from classify import GROUPS, extract_json, normalize
from nlsql import GuardError, guard_sql


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


def test_catalog_tables():
    assert set(CAT["tables"]) == {"einvoice", "taxpayer"}
    assert len(CAT["tables"]["einvoice"]["columns"]) == 15
    assert len(CAT["tables"]["taxpayer"]["columns"]) == 2


def test_catalog_synonyms():
    for term in ("turnover", "dövriyyə"):
        matches = CAT["synonym_index"][term.strip().casefold()]
        assert {"table": "einvoice", "column": "total_amount"} in matches


def test_normalize_label_aliases():
    assert normalize({"label": "Mal", "group": "BAKERY", "confidence": 0.9})["label"] == "Good"
    assert normalize({"label": "Xidmət"})["label"] == "Service"


def test_normalize_group_guard():
    assert normalize({"label": "Service", "group": "BAKERY"})["group"] is None
    assert normalize({"label": "Good", "group": "NOT_A_GROUP"})["group"] is None
    assert "BAKERY" in GROUPS


def test_extract_json_fenced():
    data = extract_json('```json\n{"label":"Good","group":"BAKERY","confidence":0.8}\n```')
    assert data["label"] == "Good"


def test_eqm_clean_query():
    cleaned = eqm._clean_query("Şpris 10 ml 21GХ38mm (Şpris) (B.Braun)")
    assert "Şpris" in cleaned
    assert "(" not in cleaned
    assert "10" not in cleaned.split()


def test_l2norm_handles_zero_rows():
    normalized = rag._l2norm(np.zeros((2, 4), dtype=np.float32))
    assert normalized.shape == (2, 4)
    assert np.isfinite(normalized).all()
