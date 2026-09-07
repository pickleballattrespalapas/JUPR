"""Check registration signing readiness; explicitly initialize a missing key.

Run only in the protected production deployment environment after owner approval.
Secret material stays in process memory and flyctl stdin. No email or database
write is performed. Existing keys, including the effective confirmation signer,
are preserved. A failed/uncertain import is never automatically retried.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys

APP = "juprleagues-api"
PROJECT_URL = "https://dnoockbwfenunhcibwfn.supabase.co"
EDIT_KEY = "JUPR_REGISTRATION_EDIT_SECRET"
CONFIRMATION_KEY = "JUPR_REGISTRATION_CONFIRMATION_SECRET"

# Only booleans and a one-way fingerprint leave the machine. Never return
# credentials or actual signing keys from this probe.
REMOTE_PROBE = '''
import hashlib, json, os, sys
from jupr_app.config import get_explicit_registration_edit_token_secret, get_registration_confirmation_token_secret
if (os.getenv("JUPR_ENV") != "production"
    or os.getenv("FLY_APP_NAME") != "juprleagues-api"
    or os.getenv("SUPABASE_URL", "").rstrip("/") != "https://dnoockbwfenunhcibwfn.supabase.co"):
    sys.exit(2)
try:
    confirmation = get_registration_confirmation_token_secret()
    try:
        get_explicit_registration_edit_token_secret()
        edit_ready = True
    except ValueError:
        edit_ready = False
    result = {
        "edit_ready": edit_ready,
        "edit_env_present": bool(os.getenv("JUPR_REGISTRATION_EDIT_SECRET")),
        "confirmation_env_present": bool(os.getenv("JUPR_REGISTRATION_CONFIRMATION_SECRET")),
        "confirmation_fingerprint": hashlib.sha256(confirmation.encode()).hexdigest(),
    }
except Exception:
    sys.exit(3)
print("JUPR_REGISTRATION_READINESS=" + json.dumps(result))
'''


class SetupError(RuntimeError):
    """Only constant, safe operator messages belong in this exception."""


def fly(*args: str, payload: str | None = None, ssh: bool = False) -> str:
    env = dict(os.environ)
    if ssh:
        env["FLY_API_TOKEN"] = env.get("FLY_SSH_TOKEN") or env.get("FLY_API_TOKEN", "")
    try:
        result = subprocess.run(
            ["flyctl", *args, "--app", APP], input=payload,
            text=True, capture_output=True, timeout=300, env=env, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise SetupError("Fly command failed or timed out; inspect readiness before retrying.") from None
    if result.returncode:
        # Provider output can contain input values. Do not propagate it, even
        # on errors, and do not interpolate CalledProcessError/TimeoutExpired.
        raise SetupError("Fly command failed; its output was withheld. Inspect readiness before retrying.")
    return result.stdout


def inventory() -> set[str]:
    try:
        rows = json.loads(fly("secrets", "list", "--json"))
    except ValueError:
        raise SetupError("Fly returned an invalid secret inventory.") from None
    if not isinstance(rows, list) or not rows:
        raise SetupError("Fly secret inventory is missing or unrecognized.")
    names = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SetupError("Fly returned an invalid secret inventory entry.")
        name = row.get("Name") or row.get("name")
        # Pinned flyctl 0.4.49 emits lowercase name/digest/status.
        status = row.get("status") or row.get("Status") or row.get("DeploymentStatus") or row.get("deployment_status")
        if not isinstance(name, str) or str(status).lower() != "deployed":
            raise SetupError("All Fly secrets must be fully deployed before initialization.")
        names.add(name)
    return names


def probe() -> dict:
    encoded = base64.b64encode(REMOTE_PROBE.encode()).decode()
    command = f"python -c 'import base64;exec(compile(base64.b64decode(\"{encoded}\"),\"registration_readiness\",\"exec\"))'"
    try:
        output = fly("ssh", "console", "--quiet", "--command", command, ssh=True)
        # flyctl can emit SSH connection notices even with --quiet. Only the
        # explicitly framed remote result is data; never log the other output.
        prefix = "JUPR_REGISTRATION_READINESS="
        frames = [line.strip()[len(prefix):] for line in output.splitlines() if line.strip().startswith(prefix)]
        if len(frames) != 1:
            raise ValueError("Missing or duplicate readiness frame")
        result = json.loads(frames[0])
    except ValueError:
        raise SetupError("Registration readiness probe returned an invalid response.") from None
    if not isinstance(result, dict) or any(
        not isinstance(result.get(key), bool)
        for key in ("edit_ready", "edit_env_present", "confirmation_env_present")
    ) or not isinstance(result.get("confirmation_fingerprint"), str) or len(result["confirmation_fingerprint"]) != 64:
        raise SetupError("Registration readiness probe returned an incomplete response.")
    return result


def prepare_missing_keys(names: set[str], before: dict, service_role_key: str) -> dict[str, str]:
    if EDIT_KEY in names or before["edit_env_present"] or before["edit_ready"]:
        raise SetupError("An edit signing key already exists; initialization never replaces an existing key.")
    updates = {}
    if CONFIRMATION_KEY in names:
        if not before["confirmation_env_present"]:
            raise SetupError("The confirmation signing key is not active on the checked machine.")
    else:
        if before["confirmation_env_present"] or not service_role_key.strip():
            raise SetupError("Cannot safely preserve the existing confirmation signing key.")
        # Production currently uses this server-only legacy fallback. Pin the
        # exact same value before introducing the higher-priority edit secret,
        # so outstanding confirmation URLs remain valid. This is NOT an edit
        # key fallback and does not change the signing algorithm.
        confirmation = "registration-confirmation-token:" + service_role_key.strip()
        fingerprint = hashlib.sha256(confirmation.encode()).hexdigest()
        if not hmac.compare_digest(fingerprint, before["confirmation_fingerprint"]):
            raise SetupError("The current confirmation signer differs from the protected credential; stop for operator review.")
        updates[CONFIRMATION_KEY] = confirmation
    updates[EDIT_KEY] = secrets.token_urlsafe(48)
    return updates


def run(*, initialize_missing: bool = False) -> dict:
    if (os.getenv("JUPR_ENV") != "production"
        or os.getenv("FLY_APP_NAME") != APP
        or os.getenv("SUPABASE_URL", "").rstrip("/") != PROJECT_URL
        or not os.getenv("FLY_API_TOKEN")):
        raise SetupError("Run only with protected production credentials and the exact production app identity.")
    names = inventory()
    before = probe()
    if before["edit_ready"] and before["edit_env_present"] and EDIT_KEY in names:
        return {"registration_edit_ready": True, "changed": False}
    if not initialize_missing:
        raise SetupError("Production registration edit links require a stable signing secret; initialization needs owner approval.")
    updates = prepare_missing_keys(names, before, os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))
    if inventory() != names or probe() != before:
        raise SetupError("Production configuration changed during preparation; no secrets were imported.")
    # Keep all values out of argv, files, logs and artifacts. Import the two
    # settings together so no machine observes a different confirmation signer.
    fly("secrets", "import", payload="".join(f"{name}={value}\n" for name, value in updates.items()))
    after_names = inventory()
    after = probe()
    if (not after["edit_ready"] or not after["edit_env_present"]
        or not after["confirmation_env_present"]
        or not {EDIT_KEY, CONFIRMATION_KEY}.issubset(after_names)
        or not hmac.compare_digest(before["confirmation_fingerprint"], after["confirmation_fingerprint"])):
        raise SetupError("Post-import signing verification failed. Preserve stored keys and inspect production; do not rotate or retry blindly.")
    return {"registration_edit_ready": True, "confirmation_signer_preserved": True, "changed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-missing", action="store_true", help="After owner approval, initialize only an absent edit key.")
    args = parser.parse_args()
    try:
        result = run(initialize_missing=args.initialize_missing)
    except SetupError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
