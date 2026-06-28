from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from classify import GROUPS, classify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task 2 invoice item classifier.")
    parser.add_argument("--input", default=str(ROOT / "data" / "processed" / "eval_sample.csv"))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "data" / "processed" / "eval_results.csv"))
    return parser.parse_args()


def load_rows(path: str, limit: int) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit > 0 else rows


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def label_is_good(label: str) -> bool:
    return label.strip().casefold() == "good"


def truncate(text: str, width: int = 70) -> str:
    clean = " ".join(text.split())
    if len(clean) <= width:
        return clean
    return clean[: width - 3] + "..."


def evaluate_row(row: dict[str, str], provider: Optional[str], model: Optional[str]) -> dict[str, Any]:
    text = row.get("text", "")
    started = time.time()
    pred = classify(text, provider, model)
    latency = time.time() - started
    true_label = row.get("label", "")
    pred_label = str(pred.get("label") or "")
    true_group = row.get("group", "") or ""
    pred_group = str(pred.get("group") or "")
    label_correct = pred_label == true_label
    group_correct = (not label_is_good(true_label)) or pred_group == true_group
    return {
        "text": text,
        "true_label": true_label,
        "pred_label": pred_label,
        "true_group": true_group,
        "pred_group": pred_group,
        "confidence": pred.get("confidence", 0.0),
        "label_correct": label_correct,
        "group_correct": group_correct,
        "ok": bool(pred.get("ok")),
        "latency_s": latency,
        "provider": pred.get("provider") or provider or "",
        "model": pred.get("model") or model or "",
    }


def write_results(path: str, rows: list[dict[str, Any]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fields = [
        "text",
        "true_label",
        "pred_label",
        "true_group",
        "pred_group",
        "confidence",
        "label_correct",
        "group_correct",
        "ok",
        "latency_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def print_confusion(results: list[dict[str, Any]]) -> None:
    labels = ["Good", "Service"]
    counts: dict[tuple[str, str], int] = collections.Counter()
    for row in results:
        true_label = row["true_label"] if row["true_label"] in labels else "Service"
        pred_label = row["pred_label"] if row["pred_label"] in labels else "Service"
        counts[(true_label, pred_label)] += 1
    print("\nLabel confusion (rows=true, cols=pred)")
    print("             pred Good  pred Service")
    for true_label in labels:
        print(f"true {true_label:<7} {counts[(true_label, 'Good')]:>9}  {counts[(true_label, 'Service')]:>12}")


def print_report(results: list[dict[str, Any]], total_time: float, out_path: str) -> None:
    total = len(results)
    parse_failures = sum(1 for row in results if not row["ok"])
    label_correct = sum(1 for row in results if row["label_correct"])
    true_good = [row for row in results if label_is_good(row["true_label"])]
    pred_good_true_good = [row for row in true_good if label_is_good(row["pred_label"])]
    strict_group_correct = sum(1 for row in true_good if row["pred_group"] == row["true_group"])
    pred_good_group_correct = sum(1 for row in pred_good_true_good if row["pred_group"] == row["true_group"])
    fully_correct = sum(
        1
        for row in results
        if row["label_correct"] and (not label_is_good(row["true_label"]) or row["pred_group"] == row["true_group"])
    )
    provider = results[0]["provider"] if results else ""
    model = results[0]["model"] if results else ""
    mean_latency = sum(float(row["latency_s"]) for row in results) / total if total else 0.0

    print(f"items: {total} | parse_failures: {parse_failures} | {provider}/{model} | total_time {total_time:.2f}s | mean_latency {mean_latency:.2f}s")
    print(f"\n*** LABEL ACCURACY: {pct(label_correct, total)} ({label_correct}/{total}) ***")
    print(f"*** FULLY CORRECT:  {pct(fully_correct, total)} ({fully_correct}/{total}) ***")
    print(
        f"*** GROUP ACCURACY true Good strict: {pct(strict_group_correct, len(true_good))} "
        f"({strict_group_correct}/{len(true_good)}) ***"
    )
    print(
        f"*** GROUP ACCURACY true Good predicted Good: "
        f"{pct(pred_good_group_correct, len(pred_good_true_good))} "
        f"({pred_good_group_correct}/{len(pred_good_true_good)}) ***"
    )

    print_confusion(results)
    print("\nPer-true-group recall")
    for group in GROUPS:
        group_rows = [row for row in true_good if row["true_group"] == group]
        correct = sum(1 for row in group_rows if row["pred_group"] == group)
        print(f"{group:<24} {pct(correct, len(group_rows)):>8} ({correct}/{len(group_rows)})")

    mistakes = [
        row
        for row in results
        if not row["label_correct"] or (label_is_good(row["true_label"]) and row["pred_group"] != row["true_group"])
    ]
    print("\nMisclassified examples")
    for row in mistakes[:12]:
        true_value = row["true_label"] if not label_is_good(row["true_label"]) else f"{row['true_label']}/{row['true_group']}"
        pred_value = row["pred_label"] if not label_is_good(row["pred_label"]) else f"{row['pred_label']}/{row['pred_group']}"
        print(f"- {truncate(row['text'])} | true={true_value} pred={pred_value} conf={float(row['confidence']):.2f}")
    print(f"\nWrote per-row results to {out_path}")


def main() -> None:
    args = parse_args()
    source_rows = load_rows(args.input, args.limit)
    started = time.time()
    results = [evaluate_row(row, args.provider, args.model) for row in source_rows]
    total_time = time.time() - started
    write_results(args.out, results)
    print_report(results, total_time, args.out)


if __name__ == "__main__":
    main()
