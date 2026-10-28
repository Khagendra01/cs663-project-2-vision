#!/usr/bin/env python3
"""Score predictions.jsonl against reference labels.

Usage:
    python3 evaluate.py --predictions predictions.jsonl

Input format (one JSON object per line, written by test_tuned_model.py):
    {"expected": <label>, "tuned": <label>, "base": <label>}

Output: overall accuracy for tuned vs base, per-class accuracy for the
tuned model, and a confusion matrix (rows = expected, columns = predicted).
Matching is exact after lowercasing and stripping whitespace; a tuned
answer that adds extra words counts as wrong, which is why the prompt
asks for exactly one word.
"""

import argparse
import json
from collections import Counter, defaultdict


def normalize(text):
    return " ".join((text or "").strip().lower().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("no predictions found")
        return

    for key in ("tuned", "base"):
        correct = sum(1 for r in rows if normalize(r[key]) == normalize(r["expected"]))
        print(f"{key}: {correct}/{len(rows)} = {correct / len(rows):.3f} accuracy")

    labels = sorted({normalize(r["expected"]) for r in rows})
    print("\nper-class tuned accuracy:")
    for label in labels:
        subset = [r for r in rows if normalize(r["expected"]) == label]
        correct = sum(1 for r in subset if normalize(r["tuned"]) == label)
        print(f"  {label}: {correct}/{len(subset)} = {correct / len(subset):.3f}")

    print("\nconfusion matrix (tuned), rows = expected, cols = predicted:")
    predicted_labels = sorted({normalize(r["tuned"]) for r in rows})
    counts = Counter(
        (normalize(r["expected"]), normalize(r["tuned"])) for r in rows)
    header = "expected \\ predicted".ljust(22) + "".join(p[:12].ljust(14) for p in predicted_labels)
    print(header)
    for expected in labels:
        row = expected[:20].ljust(22)
        row += "".join(str(counts[(expected, p)]).ljust(14) for p in predicted_labels)
        print(row)


if __name__ == "__main__":
    main()
