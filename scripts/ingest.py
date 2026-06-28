import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DATA = Path(__file__).resolve().parents[1] / "data" / "FoodWholesale_sampleData.xlsx"
MAP = {
    "Supplier TIN": "supplier_tin",
    "Recipient TIN": "recipient_tin",
    "e-Invoice Date": "einvoice_date",
    "e-Invoice Approval Date": "approval_date",
    "e-Invoice Series": "series",
    "e-Invoice Number": "number",
    "Excise Amount": "excise_amount",
    "Amount of VAT-Taxable Transactions": "vat_taxable_amount",
    "Amount of Non-VAT-Taxable Transactions": "non_vat_taxable_amount",
    "Amount of VAT-Exempt Transactions": "vat_exempt_amount",
    "Amount of Zero-Rated VAT Transactions": "zero_rated_amount",
    "VAT Amount": "vat_amount",
    "Road Tax": "road_tax",
    "Total Amount": "total_amount",
}
INVOICE_COLS = list(MAP.values())
TAXPAYER_SQL = "INSERT INTO taxpayer (tin, name) VALUES %s ON CONFLICT (tin) DO NOTHING"
DEMO_TAXPAYERS = [("1234567890", "Demo Wholesale LLC"), ("0987654321", "Demo Buyer LLC")]


def dsn():
    host, port = os.environ.get("PGHOST", "/tmp"), os.environ.get("PGPORT", "5432")
    return f"host={host} port={port} dbname={os.environ.get('PGDATABASE', 'azdata')}"


def taxpayer_name(tin):
    prefix = "Supplier" if tin.startswith("A_") else "Recipient" if tin.startswith("T_") else "Taxpayer"
    return f"{prefix} {tin}"


def read_data():
    df = pd.read_excel(DATA, sheet_name="Sheet1", engine="openpyxl")
    missing = [header for header in MAP if header not in df.columns]
    if missing:
        print("missing headers:", *missing, sep="\n")
        sys.exit(1)
    return df


def taxpayer_rows(df):
    tins = pd.concat([df["Supplier TIN"], df["Recipient TIN"]]).dropna()
    return [(tin, taxpayer_name(tin)) for tin in sorted({str(tin) for tin in tins})]


def invoice_rows(df):
    out = df[list(MAP)].rename(columns=MAP)
    for col in ("supplier_tin", "recipient_tin", "series"):
        out[col] = out[col].map(lambda v: None if pd.isna(v) else str(v))
    out["einvoice_date"] = pd.to_datetime(out["einvoice_date"]).dt.date
    out["approval_date"] = pd.to_datetime(out["approval_date"]).dt.date
    out["number"] = out["number"].map(lambda v: None if pd.isna(v) else int(v))
    for col in INVOICE_COLS[6:]:
        out[col] = out[col].fillna(0).map(float)
    return [
        tuple(None if pd.isna(value) else value for value in row)
        for row in out[INVOICE_COLS].itertuples(index=False, name=None)
    ]


def demo_rows():
    raw = [("2026-06-01", 90000001, 25000.00), ("2026-06-02", 90000002, 18000.00),
           ("2026-06-03", 90000003, 31000.00), ("2026-06-04", 90000004, 22000.00)]
    zeros = (0.00,) * 7
    dated = ((date.fromisoformat(day), num, total) for day, num, total in raw)
    return [("1234567890", "0987654321", d, d, "MT2606", n, *zeros, t) for d, n, t in dated]


def insert_invoices(cur, rows):
    sql = f"INSERT INTO einvoice ({', '.join(INVOICE_COLS)}) VALUES %s"
    execute_values(cur, sql, rows)


def print_summary(cur):
    for table in ("taxpayer", "einvoice"):
        cur.execute(f"SELECT count(*) FROM {table}")
        print(f"{table} rows: {cur.fetchone()[0]}")
    cur.execute(
        "SELECT einvoice_date, total_amount FROM einvoice "
        "WHERE supplier_tin='1234567890' ORDER BY einvoice_date"
    )
    for invoice_date, total in cur.fetchall():
        print(f"{invoice_date}: {total}")
    cur.execute(
        "SELECT COALESCE(sum(total_amount), 0) FROM einvoice "
        "WHERE supplier_tin='1234567890'"
    )
    print(f"demo turnover (sum total_amount): {cur.fetchone()[0]}")


def main():
    conn = None
    try:
        df = read_data()
        conn = psycopg2.connect(dsn())
        with conn.cursor() as cur:
            cur.execute("TRUNCATE einvoice, taxpayer RESTART IDENTITY CASCADE")
            execute_values(cur, TAXPAYER_SQL, taxpayer_rows(df))
            insert_invoices(cur, invoice_rows(df))
            execute_values(cur, TAXPAYER_SQL, DEMO_TAXPAYERS)
            insert_invoices(cur, demo_rows())
        conn.commit()
        with conn.cursor() as cur:
            print_summary(cur)
    except Exception as exc:
        if conn is not None:
            conn.rollback()
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
