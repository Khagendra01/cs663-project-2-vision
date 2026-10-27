#!/usr/bin/env python3
"""Build train/validation/test JSONL files from a labelled frame folder.

Input layout (see dataset-collection-guide.md):
    dog-data/
      <dog-name>/
        <behaviour>/
          <dog>_<behaviour>_<nnn>.jpg

Splitting is BY DOG, never by random shuffle: each dog's folders go
entirely into one split. Dogs are assigned deterministically in sorted
order: the last dog(s) go to test, the second-to-last to validation,
the rest to train. Override with --test-dogs / --val-dogs.

Usage:
    python3 build_jsonl.py --data dog-data --bucket YOUR_BUCKET \
        --out jsonl --test-dogs bella --val-dogs milo

Output: train.jsonl, validation.jsonl, test.jsonl in --out. Each line is
one JSON object: user message (image fileData gs:// URI + fixed prompt)
and model message (exactly one label word = the behaviour folder name,
with underscores converted to spaces, e.g. lying_down -> "lying down").

Validation performed before writing: every file is a .jpg/.jpeg/.png,
every label is in --labels, and no dog appears in more than one split.
Exits non-zero on any violation.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_LABELS = ["sitting", "standing", "lying down", "walking", "rolling over"]

PROMPT_TEMPLATE = ("Look at the image of the dog. What is the dog doing? "
                   "Answer with exactly one of: {labels}.")


def build_prompt(labels):
    return PROMPT_TEMPLATE.format(labels=", ".join(labels))


def label_from_folder(name):
    return name.replace("_", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dog-data root folder")
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs://)")
    parser.add_argument("--out", required=True, help="Output folder for JSONL files")
    parser.add_argument("--test-dogs", nargs="*", default=[],
                        help="Dog names forced into the test split")
    parser.add_argument("--val-dogs", nargs="*", default=[],
                        help="Dog names forced into the validation split")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS,
                        help="Allowed label words")
    args = parser.parse_args()

    data = Path(args.data)
    if not data.is_dir():
        print(f"error: data folder not found: {data}", file=sys.stderr)
        return 1

    dogs = sorted(p.name for p in data.iterdir() if p.is_dir())
    if not dogs:
        print("error: no dog folders found", file=sys.stderr)
        return 1

    split_of = {}
    for dog in dogs:
        if dog in args.test_dogs:
            split_of[dog] = "test"
        elif dog in args.val_dogs:
            split_of[dog] = "validation"
        else:
            split_of[dog] = "train"
    # Default deterministic split when no overrides: last dog -> test,
    # second-to-last -> validation, rest -> train.
    if not args.test_dogs and not args.val_dogs:
        if len(dogs) >= 2:
            split_of[dogs[-1]] = "test"
        if len(dogs) >= 3:
            split_of[dogs[-2]] = "validation"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(args.labels)
    writers = {name: open(out / f"{name}.jsonl", "w", encoding="utf-8")
               for name in ("train", "validation", "test")}
    counts = {name: 0 for name in writers}
    errors = []

    for dog in dogs:
        for behaviour_dir in sorted((data / dog).iterdir()):
            if not behaviour_dir.is_dir():
                continue
            label = label_from_folder(behaviour_dir.name)
            if label not in args.labels:
                errors.append(f"{dog}/{behaviour_dir.name}: label '{label}' "
                              f"not in allowed set {args.labels}")
                continue
            for image in sorted(behaviour_dir.iterdir()):
                if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
                record = {
                    "contents": [
                        {"role": "user", "parts": [
                            {"fileData": {
                                "mimeType": mime,
                                "fileUri": f"gs://{args.bucket}/images/{image.name}"}},
                            {"text": prompt}]},
                        {"role": "model", "parts": [{"text": label}]},
                    ]
                }
                split = split_of[dog]
                writers[split].write(json.dumps(record) + "\n")
                counts[split] += 1

    for w in writers.values():
        w.close()

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"dogs: {dogs}")
    print(f"splits: {split_of}")
    print(f"records: train={counts['train']}, "
          f"validation={counts['validation']}, test={counts['test']}")
    if counts["test"] == 0:
        print("warning: test split is empty — need at least 2 dogs "
              "(1 for test) to evaluate honestly", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
