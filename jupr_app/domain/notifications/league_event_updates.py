"""One durable player update for all published rounds of a League Live event."""
from __future__ import annotations

from datetime import date
from typing import Any

from jupr_app.data.paged_reads import read_all_rows
from jupr_app.domain.gamification.badge_policy import badges_enabled
from jupr_app.domain.matches.side_effects import ensure_notification_badge_job
from jupr_app.domain.notifications.player_profile_update_repo import (
    normalize_session_match_ids,
    queue_player_updates_for_affected_subscribers,
)


def _operations(supabase: Any, club_id: str, session_id: str) -> list[dict]:
    rows = read_all_rows(lambda: supabase.table("league_live_publish_operations")
        .select("*").eq("club_id", club_id).eq("session_id", session_id))
    return sorted(rows, key=lambda row: int(row["round_number"]))


def _scope(operations: list[dict]) -> dict:
    match_ids, matches, updates, league_updates, badge_refs = [], [], [], [], []
    for operation in operations:
        ids = normalize_session_match_ids(operation.get("published_match_ids"))
        request = operation.get("request_json") or {}
        round_matches = request.get("matches") or []
        contexts = operation.get("match_context_ids") or []
        if len(ids) != len(round_matches) or len(ids) != len(contexts):
            raise ValueError("The league round's published game receipt is incomplete.")
        match_ids.extend(ids)
        matches.extend(round_matches)
        evidence = (operation.get("result_json") or {}).get("notification_evidence") or {}
        if evidence.get("player_updates"):
            updates.extend(evidence["player_updates"])
        else:
            before = {int(row["id"]): row for row in operation.get("rating_before_json") or []}
            for after in operation.get("rating_after_json") or []:
                player_id = int(after["id"])
                if player_id in before:
                    updates.append({"player_id": player_id, "rating_mode": "doubles",
                        "expected": before[player_id], "after": after})
        league_updates.extend(evidence.get("league_rating_updates") or [])
        # The atomic round publisher uses its first committed ID as the badge
        # job reference. Preserve receipt order, not lexicographic ID order.
        badge_refs.append(str(operation["published_match_ids"][0]))
    if len(set(match_ids)) != len(match_ids):
        raise ValueError("The league event contains overlapping round receipts.")
    days = [date.fromisoformat(str(match["date"])[:10]) for match in matches]
    return {"update_scope": "session", "match_ids": normalize_session_match_ids(match_ids),
        "session_evidence": {"player_updates": updates, "league_rating_updates": league_updates}, "badge_refs": badge_refs,
        "start_date": min(days).isoformat(), "end_date": max(days).isoformat()}


def queue_league_event_player_updates(supabase: Any, *, club_id: str, session_id: str) -> dict:
    """Queue all participants seen so far; later rounds reuse the event key.

    Nothing is sendable until resolve_league_event_update verifies completion.
    Re-reading every completed round also recovers an earlier failed handoff and
    includes players who left before the final round or joined partway through.
    """
    operations = [row for row in _operations(supabase, club_id, session_id)
        if row.get("status") == "completed" and row.get("completion_audited_at")]
    if not operations:
        return {"mode": "waiting_for_event", "queued": 0}
    scope = _scope(operations)
    matches = [match for row in operations for match in row["request_json"]["matches"]]
    players = {int(match[key]) for match in matches
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if match.get(key) is not None}
    # Keep the outbox's unique date window fixed to the first published round.
    # The sender resolves the full event date range when it is ready to deliver.
    first_date = operations[0]["request_json"]["matches"][0]["date"]
    result = queue_player_updates_for_affected_subscribers(supabase,
        club_id=club_id, affected_player_ids=players, match_dates=[first_date],
        match_ids=scope["match_ids"], league_live_session_id=session_id)
    return {"mode": "queued_for_event", **result}


def resolve_league_event_update(supabase: Any, *, club_id: str, session_id: str) -> dict | None:
    """Resolve exact event games only once every round is durably published.

    An explicit finish permits an early ending. Otherwise every scheduled round
    must be complete, including its audit. Read failures propagate without
    claiming or sending the outbox item, so the worker can retry next poll.
    """
    sessions = (supabase.table("league_live_sessions")
        .select("id,status,total_rounds,completed_at").eq("club_id", club_id)
        .eq("id", session_id).limit(1).execute().data or [])
    if not sessions:
        raise ValueError("The league event for this player update is unavailable.")
    session = sessions[0]
    operations = [row for row in _operations(supabase, club_id, session_id)
        if row.get("status") != "compensated"]
    if not operations or any(row.get("status") != "completed" or not row.get("completion_audited_at")
        for row in operations):
        return None
    finished = session.get("status") in {"complete", "archived"} and bool(session.get("completed_at"))
    rounds = {int(row["round_number"]) for row in operations}
    total_rounds = int(session.get("total_rounds") or 1)
    if not finished and not set(range(1, total_rounds + 1)).issubset(rounds):
        return None
    scope = _scope(operations)
    # Verify every committed game still belongs to this club and event. Never
    # broaden to a same-day query or silently report a partial/deleted round.
    games = []
    for start in range(0, len(scope["match_ids"]), 200):
        batch_ids = scope["match_ids"][start:start + 200]
        games.extend(read_all_rows(lambda: supabase.table("matches")
            .select("id,context_id,deleted_at,match_format,rating_scope,t1_p1,t1_p2,t2_p1,t2_p2")
            .eq("club_id", club_id).eq("context_type", "league_live_session")
            .in_("id", batch_ids)))
    expected = {str(mid): str(context) for row in operations
        for mid, context in zip(row["published_match_ids"], row["match_context_ids"])}
    actual = {str(row["id"]): str(row.get("context_id")) for row in games if not row.get("deleted_at")}
    if actual != expected or len(games) != len(expected):
        raise ValueError("The league event's complete published games could not be verified.")
    if not badges_enabled():
        scope["badge_refs"] = []
        return scope
    by_id = {str(row["id"]): row for row in games}
    refs = []
    for operation in operations:
        ids = operation["published_match_ids"]
        round_games = [by_id[str(mid)] for mid in ids]
        if any(row.get("match_format", "doubles") != "singles" and row.get("rating_scope") != "unrated"
            for row in round_games):
            players = sorted({int(row[key]) for row in round_games
                for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if row.get(key) is not None})
            refs.append(ensure_notification_badge_job(supabase=supabase, club_id=club_id,
                match_ids=ids, player_ids=players, retry_failed=False))
    scope["badge_refs"] = refs
    return scope
