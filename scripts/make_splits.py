from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/labeled_items.csv"
COLUMNS = ["text", "label", "group", "group_verbose", "source"]
SPLITS = ["train", "dev", "test"]


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


def _indices_for_split(indices: list[int]) -> dict[str, list[int]]:
    shuffled = list(indices)
    random.Random(42).shuffle(shuffled)
    train_n, dev_n, test_n = _split_counts(len(shuffled))
    return {
        "train": shuffled[:train_n],
        "dev": shuffled[train_n : train_n + dev_n],
        "test": shuffled[train_n + dev_n : train_n + dev_n + test_n],
    }


def _print_split_counts(frames: dict[str, pd.DataFrame]) -> None:
    total = sum(len(frame) for frame in frames.values())
    print(f"total: {total}")
    for split in SPLITS:
        print(f"{split}: {len(frames[split])}")


def _split_table(df: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    strata = df[["label", "group"]].drop_duplicates().sort_values(["label", "group"])
    rows: list[dict[str, object]] = []
    for _, stratum in strata.iterrows():
        row: dict[str, object] = {"label": stratum["label"], "group": stratum["group"]}
        for split in SPLITS:
            frame = frames[split]
            group_mask = frame["group"].isna() if pd.isna(stratum["group"]) else frame["group"] == stratum["group"]
            mask = (frame["label"] == stratum["label"]) & group_mask
            row[split] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows, columns=["label", "group"] + SPLITS)


def _label_balance(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    labels: Iterable[object] = sorted({label for frame in frames.values() for label in frame["label"].dropna().unique()})
    rows: list[dict[str, object]] = []
    for label in labels:
        row: dict[str, object] = {"label": label}
        for split in SPLITS:
            row[split] = int((frames[split]["label"] == label).sum())
        rows.append(row)
    return pd.DataFrame(rows, columns=["label"] + SPLITS)


def main() -> None:
    df = pd.read_csv(INPUT, usecols=COLUMNS)
    split_indices: dict[str, list[int]] = {split: [] for split in SPLITS}

    for _, group_df in df.groupby(["label", "group"], sort=True, dropna=False):
        assigned = _indices_for_split(list(group_df.index))
        for split in SPLITS:
            split_indices[split].extend(assigned[split])

    frames = {
        split: df.loc[sorted(split_indices[split]), COLUMNS].reset_index(drop=True)
        for split in SPLITS
    }
    for split, frame in frames.items():
        frame.to_csv(ROOT / f"data/processed/{split}.csv", index=False)

    _print_split_counts(frames)
    print("\nper-(label,group) counts:")
    print(_split_table(df, frames).to_string(index=False))
    print("\noverall label balance:")
    print(_label_balance(frames).to_string(index=False))


if __name__ == "__main__":
    main()
