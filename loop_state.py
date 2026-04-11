#!/usr/bin/env python3
"""
Helpers for autonomous loop bookkeeping.

This keeps results schema, duplicate detection, plateau tracking, and
workspace/summary reconciliation in Python so the shell runner only has to
orchestrate the loop.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import subprocess
import sys
from pathlib import Path

RESULTS_HEADER = ["commit", "val_loglik", "status", "features", "description"]
OLD_RESULTS_HEADER = ["commit", "val_loglik", "status", "description"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ensure-schema")
    subparsers.add_parser("best")
    subparsers.add_parser("plateau")
    subparsers.add_parser("current-features")
    subparsers.add_parser("sync-current")

    has_features = subparsers.add_parser("has-features")
    has_features.add_argument("--features", required=True)

    log_cmd = subparsers.add_parser("log")
    log_cmd.add_argument("--commit", required=True)
    log_cmd.add_argument("--val-loglik", required=True)
    log_cmd.add_argument("--status", required=True)
    log_cmd.add_argument("--features", required=True)
    log_cmd.add_argument("--description", required=True)

    return parser.parse_args()


def ensure_results_schema(results_path: Path) -> None:
    if not results_path.exists():
        write_rows(results_path, [])
        return

    rows = read_rows_raw(results_path)
    if not rows:
        write_rows(results_path, [])
        return

    header = rows[0]
    body = rows[1:]
    if header == RESULTS_HEADER:
        return
    if header != OLD_RESULTS_HEADER:
        return

    upgraded: list[dict[str, str]] = []
    for row in body:
        if not row:
            continue
        record = {
            "commit": row[0] if len(row) > 0 else "",
            "val_loglik": row[1] if len(row) > 1 else "",
            "status": row[2] if len(row) > 2 else "",
            "features": "",
            "description": "\t".join(row[3:]) if len(row) > 3 else "",
        }
        upgraded.append(record)

    write_rows(results_path, upgraded)


def read_rows_raw(results_path: Path) -> list[list[str]]:
    with results_path.open(newline="") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def read_records(results_path: Path) -> list[dict[str, str]]:
    ensure_results_schema(results_path)
    rows = read_rows_raw(results_path)
    if not rows:
        return []

    records: list[dict[str, str]] = []
    for row in rows[1:]:
        if not row:
            continue
        padded = row + [""] * (len(RESULTS_HEADER) - len(row))
        records.append(dict(zip(RESULTS_HEADER, padded[: len(RESULTS_HEADER)])))
    return records


def write_rows(results_path: Path, records: list[dict[str, str]]) -> None:
    with results_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(RESULTS_HEADER)
        for record in records:
            writer.writerow([record.get(field, "") for field in RESULTS_HEADER])


def append_record(results_path: Path, record: dict[str, str]) -> None:
    ensure_results_schema(results_path)
    with results_path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([record.get(field, "") for field in RESULTS_HEADER])


def best_loglik(records: list[dict[str, str]]) -> str:
    keep_values = [float(row["val_loglik"]) for row in records if row["status"] == "keep" and row["val_loglik"]]
    if not keep_values:
        return ""
    return f"{max(keep_values):.4f}"


def plateau_count(records: list[dict[str, str]]) -> int:
    plateau = 0
    for row in records:
        if row["status"] == "keep":
            plateau = 0
        elif row["status"]:
            plateau += 1
    return plateau


def current_features(train_path: Path) -> str:
    source = train_path.read_text()
    match = re.search(r"^STATIC_FEATURE_NAMES:\s*list\[str\]\s*=\s*(\[[^\n]*\])", source, re.MULTILINE)
    if not match:
        raise SystemExit("Could not locate STATIC_FEATURE_NAMES in train.py")

    values = ast.literal_eval(match.group(1))
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise SystemExit("STATIC_FEATURE_NAMES must be a list[str]")

    return "[" + ",".join(sorted(values)) + "]"


def latest_summary_loglik(summary_path: Path) -> str:
    if not summary_path.exists():
        return ""

    for line in summary_path.read_text().splitlines():
        if line.startswith("val_loglik:"):
            return line.split(":", 1)[1].strip()
    return ""


def has_features(records: list[dict[str, str]], features: str) -> bool:
    return any(row["features"] == features for row in records)


def git_label_for_current_state(train_path: Path) -> str:
    head_hash = run_git(["rev-parse", "--short", "HEAD"]).strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "--", str(train_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return head_hash if dirty.returncode == 0 else f"{head_hash}+dirty"


def run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def sync_current_state(results_path: Path, train_path: Path, summary_path: Path) -> int:
    records = read_records(results_path)
    features = current_features(train_path)
    if has_features(records, features):
        return 0

    summary_loglik = latest_summary_loglik(summary_path)
    if not summary_loglik:
        return 0

    best = best_loglik(records)
    if best and float(summary_loglik) <= float(best):
        print("Current workspace state is ahead of results.tsv but does not improve on the logged best.", file=sys.stderr)
        print(f"Current features: {features}", file=sys.stderr)
        print(f"Current val_loglik: {summary_loglik}", file=sys.stderr)
        print(f"Logged best val_loglik: {best}", file=sys.stderr)
        return 1

    append_record(
        results_path,
        {
            "commit": git_label_for_current_state(train_path),
            "val_loglik": summary_loglik,
            "status": "keep",
            "features": features,
            "description": "sync current workspace state",
        },
    )
    print(f"Synced current workspace state to results.tsv: {features} -> {summary_loglik}")
    return 0


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    results_path = repo_root / "results.tsv"
    train_path = repo_root / "train.py"
    summary_path = repo_root / "latest_model_summary.txt"

    if args.command == "ensure-schema":
        ensure_results_schema(results_path)
        return 0

    if args.command == "best":
        print(best_loglik(read_records(results_path)))
        return 0

    if args.command == "plateau":
        print(plateau_count(read_records(results_path)))
        return 0

    if args.command == "current-features":
        print(current_features(train_path).strip("[]"))
        return 0

    if args.command == "has-features":
        return 0 if has_features(read_records(results_path), args.features) else 1

    if args.command == "log":
        append_record(
            results_path,
            {
                "commit": args.commit,
                "val_loglik": args.val_loglik,
                "status": args.status,
                "features": args.features,
                "description": args.description,
            },
        )
        return 0

    if args.command == "sync-current":
        return sync_current_state(results_path, train_path, summary_path)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
