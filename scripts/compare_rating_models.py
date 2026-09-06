#!/usr/bin/env python3
"""Compare official JUPR with the private Bayesian doubles shadow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jupr_app.domain.rating_model_comparison import (  # noqa: E402
    compare_jupr_with_bayesian_shadow,
)


def _load_json(handle: TextIO) -> list[dict[str, Any]]:
    payload = json.load(handle)
    rows = payload.get("matches") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Input must be a JSON list or an object with a matches list")
    return rows


def _load_rows(path_value: str) -> list[dict[str, Any]]:
    if path_value == "-":
        return _load_json(sys.stdin)
    path = Path(path_value)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        return _load_json(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select the Bayesian shadow on a chronological validation window, "
            "then compare it with official JUPR on an untouched holdout."
        )
    )
    parser.add_argument("--input", required=True, help="Match export path, or - for JSON stdin")
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--holdout-start", required=True)
    parser.add_argument("--holdout-end")
    parser.add_argument(
        "--exclude-league",
        action="append",
        default=[],
        help="Exact league name to exclude; repeat for multiple leagues",
    )
    parser.add_argument("--output", type=Path, help="Optional aggregate JSON report path")
    args = parser.parse_args(argv)

    try:
        rows = _load_rows(args.input)
        excluded = {value.strip().casefold() for value in args.exclude_league if value.strip()}
        if excluded:
            rows = [
                row
                for row in rows
                if str(row.get("league") or "").strip().casefold() not in excluded
            ]
        report = compare_jupr_with_bayesian_shadow(
            rows,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            holdout_start=args.holdout_start,
            holdout_end=args.holdout_end,
        )
        report["input"] = {
            "rows_received": len(rows),
            "excluded_leagues": sorted(excluded),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[rating-shadow] ERROR: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
