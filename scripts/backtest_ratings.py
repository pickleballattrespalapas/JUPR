#!/usr/bin/env python3
"""Run the frozen JUPR formula as a chronological predictive backtest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jupr_app.domain.rating_backtest import run_chronological_backtest  # noqa: E402


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("matches") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Input must be a JSON list of matches or an object with a matches list.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate JUPR in strict match chronology without future-data leakage."
    )
    parser.add_argument("--input", required=True, type=Path, help="Match export (.json, .jsonl, or .csv).")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    try:
        report = run_chronological_backtest(_load_rows(args.input))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[rating-backtest] ERROR: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
