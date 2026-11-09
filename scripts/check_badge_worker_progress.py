"""Accept current worker cycle evidence, never raw provider logs or payloads."""
import argparse
import json
from pathlib import Path


def verify(text: str, *, sha: str, machine_id: str) -> dict:
    last = None
    for line in text.splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict) and isinstance(item.get("message"), str):
                item = json.loads(item["message"])
            if not isinstance(item, dict):
                continue
        except (ValueError, TypeError):
            continue
        if (item.get("deployment_sha") == sha
                and str(item.get("worker_id", "")).startswith(machine_id + ":")
                and item.get("event") in {"badge.worker.cycle_completed", "badge.worker.cycle_failed"}):
            last = item
    if last is None or last.get("event") != "badge.worker.cycle_completed":
        raise ValueError("No successful current worker cycle is verified.")
    if last.get("program_errors") != 0 or last.get("queue_errors") != 0:
        raise ValueError("Current worker cycle reported errors.")
    return {key: last.get(key) for key in ("event", "deployment_sha", "worker_id",
             "program_processed", "queue_processed", "elapsed_ms", "cpu_ms")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--machine-id", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.logs.read_text(), sha=args.sha, machine_id=args.machine_id)))
    except ValueError:
        raise SystemExit("Current worker-cycle verification is incomplete.")
