"""Chronological, independently calculated interclub and club rating projections.

The database RPC snapshots a connected group of represented clubs and commits
both streams, local match snapshots and the result receipt in one transaction.
No cross-club rows are inserted into the ordinary club matches table.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any

from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.ratings import calculate_hybrid_elo


class InterclubRatingError(ValueError):
    pass


def _time(value: Any) -> str:
    if not value:
        raise InterclubRatingError("An official game is missing its played time.")
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise InterclubRatingError("An official game has an invalid played time.") from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc).isoformat()


def _rating(value: Any) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise InterclubRatingError("A player is missing a rating seed.") from exc
    if not isfinite(result) or result <= 0:
        raise InterclubRatingError("A player has an invalid rating seed.")
    return result


def _rpc(db, name: str, params: dict[str, Any]) -> dict[str, Any]:
    value = db.rpc(name, params).execute().data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise InterclubRatingError("Rating service returned no durable receipt.")
    return value


def _flat_games(document: dict[str, Any]) -> list[dict[str, Any]]:
    # The competition engine validates the official document first; this narrow
    # decoder additionally protects the rating boundary if called independently.
    games: list[dict[str, Any]] = []
    for encounter_index, encounter in enumerate(document.get("encounters", [])):
        for pairing_index, pairing in enumerate(encounter.get("pairings", [])):
            if pairing.get("kind") in {"singles", "rotating_singles", "tiebreak"}:
                continue
            for game_index, game in enumerate(pairing.get("games", [])):
                if game.get("status") != "completed":
                    continue
                left = list(game.get("players_a") or pairing.get("players_a") or [])
                right = list(game.get("players_b") or pairing.get("players_b") or [])
                if len(left) != 2 or len(right) != 2 or len(set(left + right)) != 4:
                    raise InterclubRatingError("A completed doubles game needs four distinct actual players.")
                a, b = game.get("a"), game.get("b")
                if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int):
                    raise InterclubRatingError("An official completed game needs integer scores.")
                if min(a, b) < 0 or not ((max(a, b) == 11 and min(a, b) <= 9) or (max(a, b) > 11 and abs(a - b) == 2)):
                    raise InterclubRatingError("A completed doubles score must reach 11 and win by two.")
                played_at = _time(game.get("played_at"))
                if datetime.fromisoformat(played_at) > datetime.now(timezone.utc):
                    raise InterclubRatingError("A completed game cannot have a future played time.")
                games.append({"id": str(game["id"]), "club_a": str(encounter["club_a"]),
                              "club_b": str(encounter["club_b"]), "players_a": left,
                              "players_b": right, "a": a, "b": b,
                              "played_at": played_at,
                              "source_order": [int(encounter.get("rotation") or encounter_index), game_index, pairing_index, str(encounter.get("id") or encounter_index)]})
    return games


def build_rating_projection(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Pure replay including local games between approved interclub games.

    Player keys include their represented club, even if a database happens to
    allocate globally unique player IDs. Season entry IDs are the league keys.
    """
    players = {(str(p["club_id"]), int(p["id"])): p for p in snapshot.get("players", [])}
    overall: dict[tuple[str, int], dict[str, Any]] = {}
    for key, player in players.items():
        overall[key] = {"rating": _rating(player.get("starting_rating")),
                        "wins": 0, "losses": 0, "matches_played": 0, "last_game_at": None}
    entries = {str(e["id"]): e for e in snapshot.get("entries", [])}
    league = {eid: _rating(entry["starting_rating"]) * 400.0 for eid, entry in entries.items()}
    events: list[dict[str, Any]] = []
    sources = {str(s["batch_id"]): s for s in snapshot.get("sources", [])}
    # A draft correction keeps the last approved source active. Approval replaces
    # exactly that source; it never adds a second copy of the games.
    for batch in snapshot.get("approved", []):
        sources[str(batch["id"])] = {"batch_id": str(batch["id"]), "revision": int(batch["revision"]),
                                    "season_id": str(batch["season_id"]), "meet_id": str(batch["meet_id"]),
                                    "document": batch["document"], "approved_at": batch["approved_at"]}
    seen: set[tuple[str, str]] = set()
    for source in sources.values():
        for game in _flat_games(source["document"]):
            key = (str(source["season_id"]), game["id"])
            if key in seen:
                raise InterclubRatingError("A game occurs in more than one official result batch.")
            seen.add(key)
            for side in ("a", "b"):
                for eid in game[f"players_{side}"]:
                    entry = entries.get(str(eid))
                    if not entry or str(entry["season_id"]) != str(source["season_id"]) or str(entry["club_id"]) != game[f"club_{side}"]:
                        raise InterclubRatingError("A game player does not represent the recorded club and season.")
                    if (str(entry["club_id"]), int(entry["player_id"])) not in overall:
                        raise InterclubRatingError("A represented club player no longer exists.")
            events.append({**game, "source": "interclub", "batch_id": source["batch_id"],
                           "season_id": str(source["season_id"]), "revision": source["revision"]})
    for match in snapshot.get("matches", []):
        if match.get("deleted_at") is not None:
            continue
        club = str(match["club_id"])
        ids = [match.get(field) for field in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        a, b = match.get("score_t1"), match.get("score_t2")
        if a is None or b is None or int(a) + int(b) <= 0:
            continue
        played_at = _time(match.get("date"))
        for pid in ids:
            key = (club, int(pid)) if pid is not None else None
            if key in overall:
                overall[key]["last_game_at"] = max(overall[key]["last_game_at"] or played_at, played_at)
        if str(match.get("rating_scope") or "").strip().casefold() == "unrated" or None in ids:
            continue  # Singles are a separate existing rating island.
        if len(set(ids)) != 4 or any((club, int(pid)) not in overall for pid in ids):
            raise InterclubRatingError("Local match player ownership is invalid; repair it before rating the meet.")
        events.append({"id": str(match["id"]), "source": "local", "club_a": club, "club_b": club,
                       "players_a": [(club, int(pid)) for pid in ids[:2]], "players_b": [(club, int(pid)) for pid in ids[2:]],
                       "a": int(a), "b": int(b), "played_at": played_at,
                       "bonus": max(0.0, float(match.get("rating_bonus_elo") or 0.0))})
    # Fixed total order; local match IDs retain their database chronological tie
    # order. Paper games should carry their real individual played timestamps.
    events.sort(key=lambda e: (e["played_at"], 0 if e["source"] == "local" else 1,
                               (int(e["id"]),) if e["source"] == "local" else tuple(e["source_order"]) + (e["batch_id"],)))
    effects, local_snapshots = [], []
    for ordinal, event in enumerate(events):
        interclub = event["source"] == "interclub"
        eids = event["players_a"] + event["players_b"]
        keys = [(str(entries[eid]["club_id"]), int(entries[eid]["player_id"])) for eid in eids] if interclub else eids
        before = [overall[key]["rating"] for key in keys]
        deltas = calculate_hybrid_elo(sum(before[:2]) / 2, sum(before[2:]) / 2, event["a"], event["b"], k_factor=DEFAULT_K_FACTOR)
        winner_a = event["a"] > event["b"]
        for i, key in enumerate(keys):
            won = winner_a if i < 2 else not winner_a
            state = overall[key]
            delta = deltas[0 if i < 2 else 1] + (event.get("bonus", 0.0) if won else 0.0)
            state["rating"] += delta
            state["wins"] += int(won)
            state["losses"] += int(not won)
            state["matches_played"] += 1
            state["last_game_at"] = max(state["last_game_at"] or event["played_at"], event["played_at"])
            if interclub:
                effects.append({"season_id": event["season_id"], "entry_id": eids[i], "game_id": event["id"],
                                "batch_id": event["batch_id"], "stream": "overall", "before_elo": before[i],
                                "after_elo": state["rating"], "played_at": event["played_at"], "ordinal": ordinal})
        if interclub:
            league_before = [league[eid] for eid in eids]
            league_delta = calculate_hybrid_elo(sum(league_before[:2]) / 2, sum(league_before[2:]) / 2,
                                              event["a"], event["b"], k_factor=DEFAULT_K_FACTOR)
            for i, eid in enumerate(eids):
                league[eid] += league_delta[0 if i < 2 else 1]
                effects.append({"season_id": event["season_id"], "entry_id": eid, "game_id": event["id"],
                                "batch_id": event["batch_id"], "stream": "league", "before_elo": league_before[i],
                                "after_elo": league[eid], "played_at": event["played_at"], "ordinal": ordinal})
        else:
            after = [overall[key]["rating"] for key in keys]
            row = {"id": int(event["id"]), "club_id": event["club_a"],
                   "elo_delta": abs(deltas[0 if winner_a else 1]) + event.get("bonus", 0.0),
                   }
            for i, field in enumerate(("t1_p1_r", "t1_p2_r", "t2_p1_r", "t2_p2_r")):
                row[field], row[f"{field}_end"] = before[i], after[i]
            local_snapshots.append(row)
    return {"players": [{"club_id": key[0], "id": key[1], **state} for key, state in overall.items()],
            "matches": local_snapshots, "effects": effects, "sources": list(sources.values()),
            "league_ratings": [{"entry_id": eid, "rating": rating / 400.0} for eid, rating in league.items()],
            "rated_games": len(seen)}


def _reconcile(db, clubs: list[str], *, batch: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    for attempt in range(3):
        source = _rpc(db, "pcs_interclub_rating_snapshot", {"p_clubs": clubs})
        snapshot = source.get("snapshot", {})
        if not snapshot.get("sources") and not snapshot.get("approved"):
            return {"status": "not_requested", "rated_games": 0}
        if batch is not None:
            canonical = next((item for item in snapshot.get("approved", []) if str(item["id"]) == str(batch["id"])), None)
            if canonical is None or int(canonical["revision"]) != int(batch["revision"]):
                raise InterclubRatingError("The approved result revision changed. Reload the meet.")
        previous = {str(item["batch_id"]): item for item in snapshot.get("sources", [])}
        if (not force and not any(item.get("pending") for item in snapshot.get("repairs", []))
                and all(item.get("ratings_status") == "completed"
                        and int(previous.get(str(item["id"]), {}).get("revision", -1)) == int(item["revision"])
                        for item in snapshot.get("approved", []))):
            return {"status": "completed", "idempotent": True,
                    "rated_games": sum(len(_flat_games(item["document"])) for item in previous.values())}
        projection = build_rating_projection(snapshot)
        result = _rpc(db, "pcs_apply_interclub_rating_projection", {"p_clubs": clubs, "p_fingerprint": source["fingerprint"],
                      "p_projection": projection, "p_expected_batch": str(batch["id"]) if batch else None,
                      "p_expected_revision": int(batch["revision"]) if batch else None})
        if result.get("status") != "conflict":
            return result
    raise InterclubRatingError("Club scores changed during rating calculation. Retry after other score submissions finish.")


def process_interclub_ratings(db, batch: dict[str, Any]) -> dict[str, Any]:
    if batch.get("state") != "approved":
        raise InterclubRatingError("Only the organizer-approved result revision can update ratings.")
    clubs = sorted({str(e[f"club_{side}"]) for e in batch.get("document", {}).get("encounters", []) for side in ("a", "b")})
    if not clubs:
        raise InterclubRatingError("There are no club matchups to rate.")
    try:
        result = _reconcile(db, clubs, batch=batch)
        return {**result, "batch_id": str(batch["id"]), "revision": int(batch["revision"])}
    except Exception as exc:
        error = str(exc) if isinstance(exc, InterclubRatingError) else "Rating calculation failed. Retry this approved result revision."
        _rpc(db, "pcs_fail_interclub_ratings", {"p_batch_id": str(batch["id"]), "p_revision": int(batch["revision"]), "p_error": error})
        return {"status": "failed", "batch_id": str(batch["id"]), "revision": int(batch["revision"]), "error": error}


def reconcile_interclub_for_club(db, club_id: str) -> dict[str, Any]:
    """Post-commit repair hook. Source changes are durably queued in SQL first.

    Legacy clients/fakes without the new RPC are left unchanged. In a migrated
    database real failures return a warning; an already committed local score
    must never be presented as an uncommitted request that may be submitted twice.
    """
    try:
        pending = db.table("pcs_interclub_rating_repairs").select("club_id").eq("club_id", str(club_id)).eq("pending", True).limit(1).execute().data
        if not pending:
            return {"status": "not_requested", "rated_games": 0}
        return _reconcile(db, [str(club_id)], force=True)
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        if code in {"PGRST202", "42883", "PGRST205", "42P01"} or isinstance(exc, (AttributeError, NotImplementedError)):
            return {"status": "not_available", "rated_games": 0}
        try:
            _rpc(db, "pcs_fail_interclub_club_ratings", {"p_club_id": str(club_id), "p_error": "Connected club ratings need a chronological replay retry."})
        except Exception:
            pass  # The original source-change trigger already persisted pending.
        return {"status": "failed", "error": "The score was saved; connected interclub ratings need a retry from meet operations."}
