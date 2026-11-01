"""Select a reachable existing production machine using read-only SSH probes."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def select_machine(rows: list[dict]) -> str:
    candidates = sorted({
        row["id"] for row in rows
        if row.get("state") == "started"
        and re.fullmatch(r"[0-9a-f]{14}", str(row.get("id", "")))
        and (row.get("config", {}).get("metadata", {}).get("fly_process_group") == "app")
    })
    for machine_id in candidates:
        try:
            result = subprocess.run(
                ["flyctl", "ssh", "console", "--quiet", "--app", "juprleagues-api",
                 "--machine", machine_id, "--command", "/bin/true"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=30, check=False,
            )
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0:
            return machine_id
    raise RuntimeError("No running production app machine accepted the read-only SSH probe.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machines-json", required=True, type=Path)
    args = parser.parse_args()
    print(select_machine(json.loads(args.machines_json.read_text())))


if __name__ == "__main__":
    main()
