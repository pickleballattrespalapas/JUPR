"""Replace one reviewed unreachable, volume-free API replica before release."""
from __future__ import annotations

import copy
import json
import os
import time
from urllib.request import Request, urlopen

APP = "juprleagues-api"
SOURCE = "2872960c342798"
UNREACHABLE = "e820e37c790718"
SHA = "9bf5ae0994e6d0ad886f9e2704ddfae82d46b574"
IMAGE = "registry.fly.io/juprleagues-api:deployment-01M530T898FJ0B67W6TBRMSHYN"
DIGEST = "sha256:6a73f6c54e3f9d926c039b2b71e3a080987cda3f7910e97c3af08e528b84d193"


def validate_source(machine):
    assert machine["host_status"] == "ok" and machine["state"] == "started"
    assert machine["config"]["image"] == IMAGE
    assert not machine["config"].get("mounts")
    assert machine["image_ref"]["digest"] == DIGEST
    assert machine["image_ref"]["labels"]["org.opencontainers.image.revision"] == SHA


def recover(api, health, *, pause=time.sleep):
    machines = {m["id"]: m for m in api("GET", "/machines")}
    volumes = api("GET", "/volumes")
    print(json.dumps({"machines": [
        {"id": m["id"], "state": m.get("state"), "host_status": m.get("host_status"),
         "image": (m.get("config") or {}).get("image"),
         "digest": (m.get("image_ref") or {}).get("digest")}
        for m in machines.values()],
        "volumes": [{"id": v.get("id"), "attached_machine_id": v.get("attached_machine_id"),
                     "state": v.get("state")} for v in volumes]}), flush=True)
    assert set(machines) == {SOURCE, UNREACHABLE}, "Unexpected machine inventory"
    assert volumes == [], "Recovery requires a volume-free app"
    source, failed = machines[SOURCE], machines[UNREACHABLE]
    validate_source(source)
    assert failed["host_status"] == "unreachable", "Target host is no longer unavailable"
    assert failed["state"] == "started"
    assert health()["git_commit_sha"] == SHA
    config = copy.deepcopy(source["config"])
    replacement = api("POST", "/machines", {
        "name": "weekly-report-recovery", "region": source["region"], "config": config,
    })
    new_id = replacement["id"]
    assert new_id not in {SOURCE, UNREACHABLE}
    print(json.dumps({"replacement_created": new_id}), flush=True)
    for _ in range(24):
        current = {m["id"]: m for m in api("GET", "/machines")}
        new = current.get(new_id, {})
        checks = new.get("checks") or []
        if (new.get("state") == "started" and new.get("host_status") == "ok"
                and checks and all(c.get("status") == "passing" for c in checks)):
            validate_source(new)
            break
        pause(5)
    else:
        raise RuntimeError("Replacement did not pass health checks; original machines preserved")
    # Recheck every destructive-action prerequisite immediately before retirement.
    assert set(current) == {SOURCE, UNREACHABLE, new_id}
    validate_source(current[SOURCE])
    assert current[UNREACHABLE]["host_status"] == "unreachable"
    assert api("GET", "/volumes") == []
    live = health()
    assert live["ok"] and live["git_commit_sha"] == live["image_build_git_sha"] == SHA
    api("DELETE", f"/machines/{UNREACHABLE}?kill=true")
    final = {m["id"]: m for m in api("GET", "/machines")
             if m["state"] not in {"destroyed", "replaced", "migrated"}}
    assert set(final) == {SOURCE, new_id}
    for machine in final.values():
        validate_source(machine)
    print(json.dumps({"ok": True, "healthy_replicas": len(final),
                      "retired_unreachable_machine": UNREACHABLE}), flush=True)


def main():
    def api(method, path, payload=None):
        req = Request(f"https://api.machines.dev/v1/apps/{APP}{path}",
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Authorization": "Bearer " + os.environ["FLY_API_TOKEN"],
                               "Content-Type": "application/json"}, method=method)
        with urlopen(req, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    def health():
        with urlopen("https://api.juprleagues.com/health", timeout=20) as response:
            return json.load(response)

    try:
        recover(api, health)
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "condition": str(exc) if isinstance(exc, AssertionError) else None,
                          "http_status": getattr(exc, "code", None)}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
