"""Validate the badge activation result after Fly's SSH status messages."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ACTIVATION = "tres_operations_badges_20260924"


def parse_activation_output(output: str) -> dict:
    results = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            results.append(value)
    if len(results) != 1:
        raise ValueError("Expected exactly one badge activation result from Fly SSH.")
    result = results[0]
    if result.get("activation") != ACTIVATION or result.get("ok") is not True:
        raise ValueError("The reviewed badge activation did not report success.")
    if result.get("already_applied") is not True and result.get("existing_awards_unchanged") is not True:
        raise ValueError("Badge activation did not confirm existing award preservation.")
    return result


if __name__ == "__main__":
    result = parse_activation_output(Path(sys.argv[1]).read_text())
    serialized = json.dumps(result, sort_keys=True) + "\n"
    Path(sys.argv[2]).write_text(serialized)
    print(serialized, end="")
