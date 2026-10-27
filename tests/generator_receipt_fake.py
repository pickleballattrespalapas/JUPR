"""Local transactional double for generator receipt RPCs; SQL tests cover real SQL."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace


class ReceiptConflict(RuntimeError):
    code = "PT409"


class ReceiptRejected(ValueError):
    code = "22023"


def commit_generator_rpc(storage, name, params):
    assert name in {"commit_admin_play_generator_session", "commit_public_play_generator_session"}
    snapshot = deepcopy(storage)
    p = params
    public = name == "commit_public_play_generator_session"
    ledger = "public_live_operations" if public else "live_ladder_admin_operations"
    op = next((r for r in snapshot.get(ledger, []) if r["operation_key"] == p["p_operation_key"]), None)
    entity = (op or {}).get("session_key" if public else "entity_id")
    if (not op or op["club_id"] != p["p_club_id"] or op["request_fingerprint"] != p["p_request_fingerprint"]
            or (entity != p["p_session_key"] and not (not public and entity == "new" and p["p_expected_version"] is None))):
        raise ReceiptRejected("Generator operation identity mismatch")
    receipts = snapshot.setdefault("play_generator_mutation_receipts", [])
    receipt = next((r for r in receipts if r["operation_key"] == p["p_operation_key"]), None)
    rows = snapshot.setdefault("live_sessions", [])
    if receipt:
        if (receipt["club_id"] != p["p_club_id"] or receipt["request_fingerprint"] != p["p_request_fingerprint"]
                or (entity != "new" and receipt["session_key"] != p["p_session_key"])):
            raise ReceiptConflict("Generator receipt mismatch")
        row = next(r for r in rows if r["club_id"] == p["p_club_id"] and r["session_key"] == receipt["session_key"])
        return SimpleNamespace(data=deepcopy(row))
    row = next((r for r in rows if r["club_id"] == p["p_club_id"] and r["session_key"] == p["p_session_key"]), None)
    now = datetime.now(timezone.utc).isoformat()
    patch = deepcopy(p["p_patch"])
    if p["p_expected_version"] is None and not public:
        if row or entity != "new" or patch.get("state", {}).get("mode") != "admin_play_generator":
            raise ReceiptRejected("Invalid new generator session")
        row = {**patch, "id": "local-" + p["p_session_key"], "version": 1, "created_at": now}
        rows.append(row)
    else:
        if (not row or row["updated_at"] != p["p_expected_version"]
                or int(row.get("version") or 1) != p["p_expected_numeric_version"]):
            raise ReceiptConflict("This generator session changed. Reload it before saving.")
        state = row.get("state") or {}
        official = state.get("official_publish") or {}
        if (state.get("round_robin_hub_id") or row.get("pending_operation_key") or state.get("generator_submission")
                or official.get("published_at") or official.get("published_match_ids")
                or state.get("mode") not in ({"admin_play_generator", "public_play_generator"} if public else {"admin_play_generator"})):
            raise ReceiptRejected("Generator requires its authorized hub or results-review workflow")
        numeric = int(row.get("version") or 1)
        row.update(patch)
        row["version"] = numeric + 1
    row.update(updated_at=now, last_seen_at=now, last_operation_key=p["p_operation_key"],
               last_request_fingerprint=p["p_request_fingerprint"])
    receipts.append({"club_id": p["p_club_id"], "session_key": row["session_key"],
        "operation_key": p["p_operation_key"], "request_fingerprint": p["p_request_fingerprint"], "committed_version": now})
    storage.clear()
    storage.update(snapshot)
    return SimpleNamespace(data=deepcopy(row))
