from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/labeled_items.csv"
COLUMNS = ["text", "label", "group", "group_verbose", "source"]
SPLITS = ["train", "dev", "test"]
SEED = 42


def _normtext(value: object) -> str:
    """Whitespace-collapsed, case-folded key used to detect duplicate items."""
    return " ".join(str(value).split()).casefold()


def _split_counts(size: int) -> tuple[int, int, int]:
    if size < 3:
        return size, 0, 0
    train = int(round(size * 0.70))
    train = max(1, min(train, size - 2))
    remaining = size - train
    dev = max(1, remaining // 2)
    test = remaining - dev
    if test < 1:
        test = 1
        dev = remaining - test
    return train, dev, test


def main() -> None:
    df = pd.read_csv(INPUT, usecols=COLUMNS)
    n_rows = len(df)

    # --- Deduplicate to UNIQUE item text BEFORE splitting. -------------------
    # The previous version split raw rows, so identical item strings scattered
    # across train/dev/test and ~20% of eval items leaked verbatim (with their
    # gold labels) into the train-based RAG index. Splitting unique texts makes
    # train/dev/test disjoint by construction, so retrieval can never hand an
    # eval item its own answer.
    df["_norm"] = df["text"].map(_normtext)
    inconsistent = int((df.groupby("_norm")["label"].nunique() > 1).sum())
    uniq = df.drop_duplicates("_norm", keep="first").reset_index(drop=True)

    # --- Stratified split of the UNIQUE texts by (label, group). -------------
    split_rows: dict[str, list[int]] = {s: [] for s in SPLITS}
    rng = random.Random(SEED)
    for _, stratum in uniq.groupby(["label", "group"], sort=True, dropna=False):
        idx = list(stratum.index)
        rng.shuffle(idx)
        tr, dv, te = _split_counts(len(idx))
        split_rows["train"].extend(idx[:tr])
        split_rows["dev"].extend(idx[tr : tr + dv])
        split_rows["test"].extend(idx[tr + dv : tr + dv + te])

    frames = {s: uniq.loc[sorted(split_rows[s]), COLUMNS].reset_index(drop=True) for s in SPLITS}

    # --- Hard assert: ZERO cross-split text overlap (the whole point). -------
    norms = {s: set(frames[s]["text"].map(_normtext)) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                shared = norms[a] & norms[b]
                assert not shared, f"LEAKAGE: {len(shared)} texts shared between {a} and {b}"

    for s in SPLITS:
        frames[s].to_csv(ROOT / f"data/processed/{s}.csv", index=False)

    # --- Report. -------------------------------------------------------------
    total = sum(len(frames[s]) for s in SPLITS)
    print(f"input rows: {n_rows} | unique texts: {len(uniq)} | written: {total}")
    if inconsistent:
        print(f"note: {inconsistent} texts had >1 distinct label in the source; kept first occurrence")
    for s in SPLITS:
        print(f"  {s}: {len(frames[s])}")
    print("OK — asserted zero train/dev/test text overlap")
    print("\nper-(label,group)  [train/dev/test]:")
    strata = uniq[["label", "group"]].drop_duplicates().sort_values(["label", "group"], na_position="last")
    for _, st in strata.iterrows():
        counts = []
        for s in SPLITS:
            f = frames[s]
            gm = f["group"].isna() if pd.isna(st["group"]) else f["group"] == st["group"]
            counts.append(int(((f["label"] == st["label"]) & gm).sum()))
        print(f"  {str(st['label']):8} {str(st['group']):24} {counts[0]}/{counts[1]}/{counts[2]}")


if __name__ == "__main__":
    main()
