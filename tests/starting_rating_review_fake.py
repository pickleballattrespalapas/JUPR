"""RPC double for generator workflow tests; SQL contract is tested separately."""
import copy
import hashlib
import json
from types import SimpleNamespace


def seeded_reviews(players):
    return {(p["id"], fmt): {"rating_elo": p.get("singles_rating" if fmt == "singles" else "starting_rating"),
                            "confirmed_at": "2026-09-01T00:00:00Z", "requested_rating": None}
            for p in players for fmt in ("doubles", "singles")}


def review_rpc(database, name, params, query_type):
    if name not in {"preview_starting_rating_review_v1", "confirm_starting_rating_reviews_v1"}:
        return None
    tables = database.db
    fmt = params["p_match_format"]

    def snapshot(pid):
        original = next(p for p in tables["players"] if p["id"] == pid and p["club_id"] == params["p_club_id"])
        if fmt == "singles":
            rating = original.get("singles_rating")
            start = (original.get("singles_replay_baseline") or {}).get("rating")
            rating = start if rating is None else rating
            prefix = "singles_"
        else:
            rating, start, prefix = original.get("rating"), original.get("starting_rating"), ""
        counted = sum(1 for m in tables.get("matches", []) if pid in [m.get(k) for k in ("t1_p1","t1_p2","t2_p1","t2_p2")]
                      and m.get("match_format", "doubles") == fmt and m.get("rating_scope") != "unrated" and not m.get("deleted_at"))
        history = max(counted, *(int(original.get(prefix + field) or 0) for field in ("matches_played", "wins", "losses")))
        player = {"id": pid, "name": original["name"], "rating": rating, "starting_rating": start}
        fingerprint = hashlib.sha256(json.dumps([original, history], sort_keys=True).encode()).hexdigest()
        return {"player": player, "rated_matches": history, "recorded_rated_matches": counted,
                "state_fingerprint": fingerprint, "confirmation": database.starting_reviews.get((pid, fmt))}

    def execute():
        if name == "preview_starting_rating_review_v1":
            return SimpleNamespace(data={"players": [snapshot(pid) for pid in params["p_player_ids"]]})
        requests = params["p_reviews"]
        snapshots = {r["player_id"]: snapshot(r["player_id"]) for r in requests}
        for r in requests:
            old = snapshots[r["player_id"]]
            if old["state_fingerprint"] != r["expected_state_fingerprint"] or old["rated_matches"]:
                raise RuntimeError("STARTING_REVIEW_CONFLICT")
        # Model an audit failure before the atomic rating mutation.
        try:
            query_type(tables, "admin_activity_log").insert({"action_type": "confirm_starting_rating", "source_page": params["p_source"]}).execute()
        except Exception as exc:
            raise RuntimeError("Required audit intent could not be saved") from exc
        for r in requests:
            pid = r["player_id"]
            current = next(p for p in tables["players"] if p["id"] == pid)
            expected = copy.deepcopy(current)
            patch = {"rating": r["rating_elo"], "starting_rating": r["rating_elo"]} if fmt == "doubles" else {}
            query = query_type(tables, "players").update(patch).eq("id", pid)
            for field in ("rating", "starting_rating", "wins", "losses", "matches_played"):
                query = query.eq(field, expected.get(field))
            if not query.execute().data:
                raise RuntimeError("STARTING_REVIEW_CONFLICT")
            database.starting_reviews[(pid, fmt)] = {"rating_elo": r["rating_elo"], "confirmed_at": "2026-10-30T00:00:00Z", "requested_rating": r.get("requested_rating")}
        return SimpleNamespace(data={"ok": True, "confirmed_count": len(requests), "players": [snapshot(r["player_id"]) for r in requests]})
    return SimpleNamespace(execute=execute)
