"""Validate staged worker topology without exposing machine environment values."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

APP = "juprleagues-api-staging"
WORKER_COMMAND = "env JUPR_BADGE_QUEUE_LEASES=1 python -m jupr_app.workers.isolated_badge_worker --club-id tres_palapas"


def verify(machines: list[dict], *, phase: str) -> dict:
    if not isinstance(machines, list) or phase not in {"before", "api", "final"}:
        raise ValueError("Invalid staging topology input.")
    groups: dict[str, list] = {"app": [], "badge_worker": []}
    digests = set()
    seen = set()
    for machine in machines:
        if machine.get("state") in {"destroyed", "replaced", "migrated"}:
            continue
        config = machine.get("config") or {}
        group = (config.get("metadata") or {}).get("fly_process_group", "app")
        machine_id = machine.get("id", "")
        if group not in groups or not re.fullmatch(r"[0-9a-f]+", machine_id) or machine_id in seen:
            raise ValueError("Unexpected staging machine identity or process group.")
        seen.add(machine_id)
        if machine.get("region") != "dfw":
            raise ValueError("Worker rehearsal requires Dallas machines.")
        image = machine.get("image_ref") or {}
        if image.get("registry") != "registry.fly.io" or image.get("repository") != APP:
            raise ValueError("Foreign machine image in staging inventory.")
        groups[group].append(machine)
        if phase == "before":
            continue
        if group == "badge_worker" and phase == "api":
            if machine.get("state") != "stopped":
                raise ValueError("The previous worker must be stopped during API rollout.")
            continue
        if machine.get("state") != "started":
            raise ValueError("A required staging machine is not started.")
        guest = config.get("guest") or {}
        if (guest.get("cpu_kind"), guest.get("cpus"), guest.get("memory_mb")) != (
            "shared", 1, 1024 if group == "app" else 2048
        ):
            raise ValueError("Staging machine capacity differs from the reviewed topology.")
        env = config.get("env") or {}
        expected = {"JUPR_ENV": "staging", "JUPR_EMAIL_MODE": "dry_run",
                    "JUPR_BADGE_WORKER_MODE": "external",
                    "JUPR_BADGE_WORKER_CLUB_ID": "tres_palapas",
                    "JUPR_BADGE_QUEUE_LEASES": "0"}
        if any(env.get(key) != value for key, value in expected.items()):
            raise ValueError("Staging worker routing flags do not match the release.")
        if group == "badge_worker":
            if config.get("services"):
                raise ValueError("The background worker must not receive HTTP traffic.")
            command = (config.get("init") or {}).get("cmd") or []
            if command != WORKER_COMMAND.split():
                raise ValueError("The background worker command is not lease-enabled and scoped.")
        elif not config.get("services"):
            raise ValueError("API machine is missing its HTTP service.")
        digest = str(image.get("digest", "")).removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Missing immutable staging image digest.")
        digests.add(digest)
    if len(groups["app"]) != 2 or len(groups["badge_worker"]) > 1:
        raise ValueError("Expected two API machines and at most one worker.")
    if phase == "final" and len(groups["badge_worker"]) != 1:
        raise ValueError("Exactly one dedicated worker is required.")
    if phase != "before" and len(digests) != 1:
        raise ValueError("All updated machines must use the same immutable image.")
    result = {"app": APP, "phase": phase,
              "api_ids": [m["id"] for m in groups["app"]],
              "worker_ids": [m["id"] for m in groups["badge_worker"]]}
    if digests:
        result["image"] = f"registry.fly.io/{APP}@sha256:{next(iter(digests))}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--phase", choices=["before", "api", "final"], required=True)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.inventory.read_text()), phase=args.phase)))


if __name__ == "__main__":
    main()
