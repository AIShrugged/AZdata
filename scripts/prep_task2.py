import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
GOODS = DATA_DIR / "e-invoice_Data_samples_goods.xlsx"
SERVICES = DATA_DIR / "e-invoice_Data_samples_services.xlsx"
EQM = DATA_DIR / "eqm_mal_kodlari-v1.xls"
ITEM_COLUMNS = ["text", "label", "group", "group_verbose", "source"]


def require_columns(df: pd.DataFrame, expected: list[str], source: str) -> None:
    missing = [column for column in expected if column not in df.columns]
    if missing:
        print(f"{source} missing columns:")
        print(*missing, sep="\n")
        sys.exit(1)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_goods() -> pd.DataFrame:
    df = pd.read_excel(GOODS, sheet_name="Sheet1", header=0, engine="openpyxl")
    require_columns(df, ["MƏHSULUN ADI", "GROUP", "Aİ QRUP"], "GOODS")
    return pd.DataFrame(
        {
            "text": df["MƏHSULUN ADI"].map(clean_text),
            "label": "Good",
            "group": df["Aİ QRUP"].map(clean_text),
            "group_verbose": df["GROUP"].map(clean_text),
            "source": "goods",
        },
        columns=ITEM_COLUMNS,
    )


def load_services() -> pd.DataFrame:
    df = pd.read_excel(SERVICES, sheet_name="Sheet1", header=1, engine="openpyxl")
    require_columns(df, ["MAL_ADI", "MAL/XİDMƏT"], "SERVICES")
    labels = sorted({clean_text(value) for value in df["MAL/XİDMƏT"].dropna()})
    unexpected = [value for value in labels if value != "Xidmət"]
    if unexpected:
        print("SERVICES unexpected MAL/XİDMƏT values:")
        print(*unexpected, sep="\n")
    return pd.DataFrame(
        {
            "text": df["MAL_ADI"].map(clean_text),
            "label": "Service",
            "group": "",
            "group_verbose": "",
            "source": "services",
        },
        columns=ITEM_COLUMNS,
    )


def load_eqm() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(EQM, sheet_name="Sheet 1", engine="xlrd")
    require_columns(df, ["CODE", "ADI", "VAHID", "STATE"], "EQM")
    rows: list[dict[str, object]] = []
    nine_digit_examples: list[str] = []
    for record in df[["CODE", "ADI", "VAHID", "STATE"]].to_dict("records"):
        raw_code = record["CODE"]
        if pd.isna(raw_code):
            continue
        code_int = int(raw_code)
        code = f"{code_int:010d}"
        if 100_000_000 <= code_int <= 999_999_999 and len(nine_digit_examples) < 3:
            nine_digit_examples.append(code)
        rows.append(
            {
                "code": code,
                "description": clean_text(record["ADI"]),
                "unit": clean_text(record["VAHID"]),
                "active": record["STATE"] == 1,
            }
        )
    return pd.DataFrame(rows, columns=["code", "description", "unit", "active"]), nine_digit_examples


def build_items() -> pd.DataFrame:
    items = pd.concat([load_goods(), load_services()], ignore_index=True)
    return items[items["text"].map(clean_text) != ""].reset_index(drop=True)


def build_eval_sample(items: pd.DataFrame) -> pd.DataFrame:
    goods = items[items["label"] == "Good"]
    services = items[items["label"] == "Service"]
    goods_sample = goods.groupby("group", sort=True).head(20)
    services_sample = services.head(40)
    return pd.concat([goods_sample, services_sample], ignore_index=True)[ITEM_COLUMNS]


def print_distribution(title: str, values: pd.Series) -> None:
    print(title)
    counts = values.value_counts()
    print("(empty)" if counts.empty else counts.to_string())


def print_summary(items: pd.DataFrame, eval_sample: pd.DataFrame, eqm: pd.DataFrame, examples: list[str]) -> None:
    item_counts = items["label"].value_counts()
    print(
        f"labeled_items: {len(items)}  "
        f"(Good {int(item_counts.get('Good', 0))}, Service {int(item_counts.get('Service', 0))})"
    )
    print_distribution("goods group distribution:", items.loc[items["label"] == "Good", "group"])
    eval_counts = eval_sample["label"].value_counts()
    print(
        f"eval_sample: {len(eval_sample)} rows  "
        f"(Good {int(eval_counts.get('Good', 0))}, Service {int(eval_counts.get('Service', 0))})"
    )
    print_distribution("eval_sample group distribution:", eval_sample.loc[eval_sample["label"] == "Good", "group"])
    print(f"eqm_registry: {len(eqm)} rows, active {int(eqm['active'].sum())}")
    print("zero-pad check:", ", ".join(examples))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = build_items()
    items.to_csv(OUT_DIR / "labeled_items.csv", index=False, encoding="utf-8")
    eval_sample = build_eval_sample(items)
    eval_sample.to_csv(OUT_DIR / "eval_sample.csv", index=False, encoding="utf-8")
    eqm, examples = load_eqm()
    eqm.to_csv(OUT_DIR / "eqm_registry.csv", index=False, encoding="utf-8")
    print_summary(items, eval_sample, eqm, examples)


if __name__ == "__main__":
    main()
