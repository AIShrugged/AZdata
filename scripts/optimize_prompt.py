from __future__ import annotations

import sys, csv, json, time, argparse, os, collections, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("OPENROUTER_API_KEY", (Path.home()/".config/azdata/openrouter.key").read_text().strip())
import rag as R
from nlsql import call_llm


Row = dict[str, object]
Error = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize RAG classifier instructions on a fixed dev sample.")
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--optimizer-model", default="qwen/qwen3.5-122b-a10b")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--sample", type=int, default=250)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def clean(value: object) -> object:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def load_dev_sample(sample_size: int) -> list[Row]:
    rows: list[Row] = []
    with open(ROOT / "data/processed/dev.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "text": clean(row.get("text")),
                    "label": clean(row.get("label")),
                    "group": clean(row.get("group")),
                }
            )

    strata: dict[tuple[object, object], list[Row]] = collections.defaultdict(list)
    for row in rows:
        strata[(row["label"], row["group"])].append(row)

    rng = random.Random(13)
    per = sample_size // len(strata) if strata else 0
    sample_rows: list[Row] = []
    for key in sorted(strata, key=lambda item: (item[0] or "", item[1] or "")):
        bucket = strata[key]
        rng.shuffle(bucket)
        sample_rows.extend(bucket[:per])
    return sample_rows


def is_fully_correct(row: Row, pred: dict[str, object]) -> bool:
    true_label = row["label"]
    pred_label = pred.get("label")
    if pred_label != true_label:
        return False
    return true_label == "Service" or pred.get("group") == row["group"]


def format_error(error: Error) -> str:
    return (
        f'- "{error["text"]}" TRUE={error["true_label"]}/{error["true_group"]} '
        f'PRED={error["pred_label"]}/{error["pred_group"]}'
    )


class Optimizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sample_rows = load_dev_sample(args.sample)
        self.emb, self.meta = R.load_index(ROOT / "data/processed/train_index")

    def classify_one(self, row: Row, instructions: str) -> dict[str, object]:
        return R.classify_rag(
            str(row["text"] or ""),
            self.emb,
            self.meta,
            k=self.args.k,
            provider=self.args.provider,
            model=self.args.model,
            instructions=instructions,
        )

    def evaluate(self, instructions: str) -> tuple[float, list[Error]]:
        if not self.sample_rows:
            return 0.0, []

        predictions: list[object] = [None] * len(self.sample_rows)
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = {
                pool.submit(self.classify_one, row, instructions): idx
                for idx, row in enumerate(self.sample_rows)
            }
            for future in as_completed(futures):
                predictions[futures[future]] = future.result()

        correct = 0
        errors: list[Error] = []
        for row, pred in zip(self.sample_rows, predictions):
            predicted = pred if isinstance(pred, dict) else {"label": None, "group": None}
            if is_fully_correct(row, predicted):
                correct += 1
            else:
                errors.append(
                    {
                        "text": row["text"],
                        "true_label": row["label"],
                        "true_group": row["group"],
                        "pred_label": str(predicted.get("label")) if predicted.get("label") is not None else None,
                        "pred_group": str(predicted.get("group")) if predicted.get("group") is not None else None,
                    }
                )
        return 100.0 * correct / len(self.sample_rows), errors

    def propose(self, current_instructions: str, errors: list[Error]) -> str:
        system = (
            "You are optimizing the INSTRUCTION text of an Azerbaijani invoice-item classifier "
            "(Good/Mal vs Service/Xidmət, plus a 7-group label for Goods). You will see the "
            "current instruction and items it got wrong. Rewrite the instruction to fix the "
            "GENERAL error patterns. Rules: keep it concise; keep the required JSON output "
            "format intact; you MAY add clarifying rules to disambiguate confusable categories "
            "(e.g. PUBLIC UTILITIES WATER vs services, DENTAL MEDICINE); do NOT hard-code "
            "specific item texts or memorize examples; output ONLY the new instruction text, "
            "nothing else."
        )
        user = (
            "CURRENT INSTRUCTION:\n"
            + current_instructions
            + "\n\nMISCLASSIFIED (true vs predicted):\n"
            + "\n".join(format_error(error) for error in errors[:40])
        )
        rewritten = call_llm(system, user, provider=self.args.provider, model=self.args.optimizer_model).strip()
        if len(rewritten) < 50:
            return current_instructions
        return rewritten


def main() -> None:
    args = parse_args()
    optimizer = Optimizer(args)

    started = time.time()
    best_instr = R.DEFAULT_INSTRUCTIONS
    best_acc, best_errors = optimizer.evaluate(best_instr)
    print(f"baseline: {best_acc:.1f}%")

    history: list[tuple[str, float]] = [("baseline", best_acc)]
    for round_no in range(1, args.rounds + 1):
        cand = optimizer.propose(best_instr, best_errors)
        cand_acc, cand_errors = optimizer.evaluate(cand)
        print(f"round {round_no}: {cand_acc:.1f}%  (best {best_acc:.1f}%)")
        if cand_acc > best_acc:
            best_instr, best_acc, best_errors = cand, cand_acc, cand_errors
            print("  -> kept")
        else:
            print("  -> reverted")
        history.append((f"round{round_no}", cand_acc))

    print("history: " + json.dumps(history, ensure_ascii=False))
    print(f"best: {best_acc:.1f}%")
    print(f"elapsed: {time.time() - started:.1f}s")
    with open(ROOT / "data/processed/best_instructions.txt", "w", encoding="utf-8") as fh:
        fh.write(best_instr)
    print("saved best instruction")


if __name__ == "__main__":
    main()
