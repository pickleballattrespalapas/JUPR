from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import json
from typing import Any

import pandas as pd

from jupr_app.domain.challenge_ladder import (
    TIER_DEFS,
    TIER_ORDER,
    ladder_bucket_challenge,
    ladder_can_initiate_challenge,
    ladder_can_receive_challenge,
    ladder_compute_status_map,
    ladder_pair_eligibility,
    normalize_tier_id,
)

LADDER_SETTINGS_DEFAULTS = {
    "challenge_range": 7,
    "accept_window_hours": 48,
    "play_window_days": 7,
    "cooldown_hours": 72,
    "protected_hours": 72,
    "pass_hold_hours": 72,
}
STATUS_SHORT = {
    "Ready to Defend": "Ready",
    "Reinstate Required": "Reinstate",
    "Vacation": "Vacation",
    "Pass Hold": "Pass Hold",
    "Locked": "Locked",
    "Protected": "Protected",
    "Cooldown": "Cooldown",
}
PUBLIC_BUCKETS = [
    "Pending Acceptance",
    "Accepted / In Window",
    "Acceptance Overdue",
    "Play Overdue",
    "Recently Completed",
]
# Keep the public activity view bounded without inventing a time window that the
# product has not defined. The rows are ordered by completed_at below.
PUBLIC_RECENT_COMPLETED_LIMIT = 12
PUBLIC_RESULT_MATCH_SELECT = (
    "id,club_id,date,context_type,context_id,deleted_at,"
    "t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2,"
    "t1_p1_r,t1_p1_r_end,t1_p2_r,t1_p2_r_end,"
    "t2_p1_r,t2_p1_r_end,t2_p2_r,t2_p2_r_end"
)
PUBLIC_LEGACY_RESULT_SELECT = (
    "club_id,challenge_id,match_no,chal_partner_id,def_partner_id,"
    "g1_chal,g1_def,g2_chal,g2_def,g3_chal,g3_def,verified,verified_at"
)
PUBLIC_STATUS_LEGEND = [
    {
        "status": "Ready to Defend",
        "short": "Ready",
        "can_initiate": True,
        "can_receive": True,
        "meaning": "Can challenge another eligible player and be challenged.",
    },
    {
        "status": "Protected",
        "short": "Protected",
        "can_initiate": True,
        "can_receive": False,
        "meaning": "Can challenge, but cannot be challenged yet after a win.",
    },
    {
        "status": "Cooldown",
        "short": "Cooldown",
        "can_initiate": False,
        "can_receive": True,
        "meaning": "Can be challenged, but cannot start a challenge yet.",
    },
    {
        "status": "Locked",
        "short": "Locked",
        "can_initiate": False,
        "can_receive": False,
        "meaning": "Already in an open challenge and cannot join another one yet.",
    },
    {
        "status": "Pass Hold",
        "short": "Pass Hold",
        "can_initiate": False,
        "can_receive": False,
        "meaning": "A monthly pass is in effect, so challenges are paused for now.",
    },
    {
        "status": "Vacation",
        "short": "Vacation",
        "can_initiate": False,
        "can_receive": False,
        "meaning": "Temporarily unavailable for ladder challenges.",
    },
    {
        "status": "Reinstate Required",
        "short": "Reinstate",
        "can_initiate": False,
        "can_receive": False,
        "meaning": "Club staff must reinstate this player before they can join challenges again.",
    },
    {
        "status": "Inactive",
        "short": "Inactive",
        "can_initiate": False,
        "can_receive": False,
        "meaning": "Not shown on the active public ladder and not eligible for a challenge.",
    },
]


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _public_status_detail(status_name: str, raw_detail: Any) -> str:
    """Project computed status context without exposing operator-entered notes."""

    if status_name == "Reinstate Required":
        return "Ask club staff to reinstate this player."
    if status_name == "Vacation":
        return "Temporarily unavailable."
    if status_name == "Pass Hold":
        return "Monthly pass in effect."
    detail = str(raw_detail or "").replace("<", "").replace(">", "").strip()
    return detail[:240]


def _fetch_table(supabase: Any, table_name: str, select_cols: str, *, club_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        query = supabase.table(table_name).select(select_cols)
        if club_id is not None:
            query = query.eq("club_id", str(club_id))
        if limit is not None:
            query = query.limit(int(limit))
        return _safe_rows(query.execute())
    except Exception:
        return []


def _settings(supabase: Any, *, club_id: str) -> dict[str, int]:
    rows = _fetch_table(
        supabase,
        "ladder_settings",
        "club_id,challenge_range,accept_window_hours,play_window_days,cooldown_hours,protected_hours,pass_hold_hours",
        club_id=club_id,
        limit=1,
    )
    raw = dict(rows[0]) if rows else {}
    return {
        key: int(_safe_int(raw.get(key), default) or default)
        for key, default in LADDER_SETTINGS_DEFAULTS.items()
    }


def _player_maps(supabase: Any, *, club_id: str) -> tuple[dict[int, str], set[int], dict[int, float]]:
    rows = _fetch_table(supabase, "players", "id,club_id,name,rating,active,inactive_at", club_id=club_id, limit=5000)
    id_to_name: dict[int, str] = {}
    active_ids: set[int] = set()
    rating_map: dict[int, float] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is None:
            continue
        pid = int(pid)
        name = str(row.get("name") or f"Player {pid}")
        id_to_name[pid] = name
        rating_map[pid] = _safe_float(row.get("rating"), 1200.0) or 1200.0
        if row.get("active") is not False and not row.get("inactive_at"):
            active_ids.add(pid)
    return id_to_name, active_ids, rating_map


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df


def _roster_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    return _fetch_table(
        supabase,
        "ladder_roster",
        "id,club_id,player_id,tier_id,rank,is_active,joined_at,left_at,updated_at",
        club_id=club_id,
        limit=5000,
    )


def _flag_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    return _fetch_table(
        supabase,
        "ladder_player_flags",
        "club_id,player_id,vacation_until,reinstate_required,reinstate_notes,updated_at",
        club_id=club_id,
        limit=5000,
    )


def _challenge_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    base_select = (
        "id,club_id,challenger_id,defender_id,challenger_rank_at_create,defender_rank_at_create,"
        "tier_id,status,created_at,accept_by,accepted_at,play_by,completed_at,winner_id"
    )
    rows = _fetch_table(
        supabase,
        "ladder_challenges",
        f"{base_select},public_result_json",
        club_id=club_id,
        limit=5000,
    )
    # Do not retry without public_result_json. A missing projection column or a
    # transient read failure must hide result detail rather than accidentally
    # substituting legacy evidence for a forward-bound result.
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return rows


def _pass_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    rows = _fetch_table(supabase, "ladder_pass_usage", "club_id,player_id,used_at,challenge_id", club_id=club_id, limit=2000)
    rows.sort(key=lambda row: str(row.get("used_at") or ""), reverse=True)
    return rows


def _tier_payload(tier_id: str, roster: list[dict[str, Any]], status_map: dict[int, dict[str, Any]], id_to_name: dict[int, str], active_ids: set[int], rating_map: dict[int, float]) -> dict[str, Any]:
    tier = TIER_DEFS.get(tier_id, {"label": tier_id, "range": ""})
    players: list[dict[str, Any]] = []
    for row in roster:
        if row.get("is_active") is False:
            continue
        pid = _safe_int(row.get("player_id"))
        if pid is None:
            continue
        pid = int(pid)
        if pid not in active_ids:
            continue
        if normalize_tier_id(str(row.get("tier_id") or "")) != tier_id:
            continue
        status = status_map.get(pid, {"status": "Ready to Defend", "until": None, "detail": ""})
        status_name = str(status.get("status") or "Ready to Defend")
        rating = rating_map.get(pid)
        players.append(
            {
                "player_id": pid,
                "player_name": id_to_name.get(pid, f"Player {pid}"),
                "rank": _safe_int(row.get("rank"), 999999),
                "rating_jupr": (rating / 400.0) if rating is not None else None,
                "status": status_name,
                "status_short": STATUS_SHORT.get(status_name, status_name),
                "detail": _public_status_detail(status_name, status.get("detail")),
                "until": _json_safe(status.get("until")),
                "challenge_id": _safe_int(status.get("challenge_id")),
            }
        )
    players.sort(key=lambda item: (_safe_int(item.get("rank"), 999999) or 999999, str(item.get("player_name") or "").lower()))
    return {
        "tier_id": tier_id,
        "label": str(tier.get("label") or tier_id),
        "range": str(tier.get("range") or ""),
        "players": players,
    }


def _challenge_side_payload(
    player_id: int | None,
    *,
    rank_at_create: int | None,
    id_to_name: dict[int, str],
    public_player_state: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    current = public_player_state.get(int(player_id), {}) if player_id is not None else {}
    return {
        "player_id": player_id,
        "player_name": (
            id_to_name.get(int(player_id), f"Player {player_id}")
            if player_id is not None
            else "—"
        ),
        "rank_at_create": rank_at_create,
        "current_rank": _safe_int(current.get("rank")),
        "current_rating_jupr": _safe_float(current.get("rating_jupr")),
    }


def _challenge_payload(
    row: dict[str, Any],
    id_to_name: dict[int, str],
    public_player_state: dict[int, dict[str, Any]],
    result_details: dict[str, Any] | None,
) -> dict[str, Any]:
    challenger_id = _safe_int(row.get("challenger_id"))
    defender_id = _safe_int(row.get("defender_id"))
    winner_id = _safe_int(row.get("winner_id"))
    challenger = _challenge_side_payload(
        challenger_id,
        rank_at_create=_safe_int(row.get("challenger_rank_at_create")),
        id_to_name=id_to_name,
        public_player_state=public_player_state,
    )
    defender = _challenge_side_payload(
        defender_id,
        rank_at_create=_safe_int(row.get("defender_rank_at_create")),
        id_to_name=id_to_name,
        public_player_state=public_player_state,
    )
    if winner_id == challenger_id:
        winner = dict(challenger)
    elif winner_id == defender_id:
        winner = dict(defender)
    else:
        winner = _challenge_side_payload(
            winner_id,
            rank_at_create=None,
            id_to_name=id_to_name,
            public_player_state=public_player_state,
        )
    payload = {
        "id": _safe_int(row.get("id")),
        "tier_id": normalize_tier_id(str(row.get("tier_id") or "")),
        "status": str(row.get("status") or ""),
        "bucket": ladder_bucket_challenge(row),
        "challenger": challenger,
        "defender": defender,
        "winner": winner if winner_id is not None else None,
        "created_at": _json_safe(row.get("created_at")),
        "accept_by": _json_safe(row.get("accept_by")),
        "play_by": _json_safe(row.get("play_by")),
        "completed_at": _json_safe(row.get("completed_at")),
    }
    if result_details is not None:
        payload["result_details"] = result_details
    return payload


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _jupr_snapshot(value: Any) -> float | None:
    rating = _safe_float(value)
    return round(rating / 400.0, 6) if rating is not None else None


def _public_rating_changes(
    match: dict[str, Any],
    *,
    id_to_name: dict[int, str],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for slot in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
        player_id = _safe_int(match.get(slot))
        if player_id is None:
            continue
        before = _jupr_snapshot(match.get(f"{slot}_r"))
        after = _jupr_snapshot(match.get(f"{slot}_r_end"))
        changes.append(
            {
                "player_id": player_id,
                "player_name": id_to_name.get(player_id, f"Player {player_id}"),
                "before_jupr": before,
                "after_jupr": after,
                "delta_jupr": (
                    round(after - before, 6)
                    if before is not None and after is not None
                    else None
                ),
            }
        )
    return changes


def _forward_result_match_ids(challenge: dict[str, Any]) -> tuple[int, int] | None:
    relation = _json_object(challenge.get("public_result_json"))
    if relation.get("version") != 1:
        return None
    match_ids = relation.get("match_ids")
    if not isinstance(match_ids, dict):
        return None
    match_a_id = _safe_int(match_ids.get("a"))
    match_b_id = _safe_int(match_ids.get("b"))
    if (
        match_a_id is None
        or match_b_id is None
        or match_a_id <= 0
        or match_b_id <= 0
        or match_a_id == match_b_id
    ):
        return None
    return int(match_a_id), int(match_b_id)


def _load_forward_result_matches(
    supabase: Any,
    *,
    club_id: str,
    challenges: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    match_ids = sorted(
        {
            match_id
            for challenge in challenges
            if challenge.get("public_result_json") is not None
            for match_id in (_forward_result_match_ids(challenge) or ())
        }
    )
    if not match_ids:
        return {}
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select(PUBLIC_RESULT_MATCH_SELECT)
            .eq("club_id", str(club_id))
            .in_("id", match_ids)
            .limit(len(match_ids))
            .execute()
        )
    except Exception:
        return {}
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        match_id = _safe_int(row.get("id"))
        if (
            match_id is None
            or match_id not in match_ids
            or str(row.get("club_id") or "") != str(club_id)
            or row.get("deleted_at")
            or match_id in result
        ):
            continue
        result[int(match_id)] = dict(row)
    return result


def _load_legacy_result_rows(
    supabase: Any,
    *,
    club_id: str,
    challenges: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    challenge_ids = sorted(
        {
            challenge_id
            for challenge in challenges
            if "public_result_json" in challenge
            and challenge.get("public_result_json") is None
            and str(challenge.get("status") or "") == "COMPLETED"
            and (challenge_id := _safe_int(challenge.get("id"))) is not None
        }
    )
    if not challenge_ids:
        return {}
    try:
        rows = _safe_rows(
            supabase.table("ladder_challenge_matches")
            .select(PUBLIC_LEGACY_RESULT_SELECT)
            .eq("club_id", str(club_id))
            .in_("challenge_id", challenge_ids)
            .limit(len(challenge_ids) * 4)
            .execute()
        )
    except Exception:
        return {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    challenge_id_set = set(challenge_ids)
    for row in rows:
        challenge_id = _safe_int(row.get("challenge_id"))
        if (
            challenge_id not in challenge_id_set
            or str(row.get("club_id") or "") != str(club_id)
            or row.get("verified") is not True
        ):
            continue
        grouped[int(challenge_id)].append(dict(row))
    return dict(grouped)


def _public_result_details(
    *,
    challenge: dict[str, Any],
    id_to_name: dict[int, str],
    forward_matches: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve only the exact match IDs persisted by the guarded publish RPC."""

    if str(challenge.get("status") or "") != "COMPLETED":
        return None
    relation = _json_object(challenge.get("public_result_json"))
    rank_change = relation.get("rank_change")
    resolved_match_ids = _forward_result_match_ids(challenge)
    if resolved_match_ids is None or not isinstance(rank_change, dict):
        return None
    match_a_id, match_b_id = resolved_match_ids

    challenger_id = _safe_int(challenge.get("challenger_id"))
    defender_id = _safe_int(challenge.get("defender_id"))
    if (
        challenger_id is None
        or defender_id is None
        or challenger_id not in id_to_name
        or defender_id not in id_to_name
    ):
        return None
    ranked_changes: dict[str, dict[str, Any]] = {}
    for side, expected_player_id in (
        ("challenger", challenger_id),
        ("defender", defender_id),
    ):
        raw_side = rank_change.get(side)
        if not isinstance(raw_side, dict):
            return None
        player_id = _safe_int(raw_side.get("player_id"))
        before = _safe_int(raw_side.get("before"))
        after = _safe_int(raw_side.get("after"))
        if player_id != expected_player_id or before is None or after is None:
            return None
        ranked_changes[side] = {
            "player_id": player_id,
            "player_name": id_to_name.get(player_id, f"Player {player_id}"),
            "before": before,
            "after": after,
            "delta": after - before,
        }
    swapped = bool(rank_change.get("swapped"))
    challenger_change = ranked_changes["challenger"]
    defender_change = ranked_changes["defender"]
    if swapped:
        if (
            challenger_change["after"] != defender_change["before"]
            or defender_change["after"] != challenger_change["before"]
        ):
            return None
    elif (
        challenger_change["before"] != challenger_change["after"]
        or defender_change["before"] != defender_change["after"]
    ):
        return None

    public_matches: list[dict[str, Any]] = []
    for slot, match_id in (("a", match_a_id), ("b", match_b_id)):
        match = forward_matches.get(int(match_id))
        if not isinstance(match, dict):
            return None
        if (
            _safe_int(match.get("id")) != match_id
            or str(match.get("context_type") or "") != "challenge_ladder"
        ):
            return None
        challenger_partner_id = _safe_int(match.get("t1_p2"))
        defender_partner_id = _safe_int(match.get("t2_p2"))
        if (
            _safe_int(match.get("t1_p1")) != challenger_id
            or _safe_int(match.get("t2_p1")) != defender_id
            or challenger_partner_id is None
            or defender_partner_id is None
            or challenger_partner_id in {challenger_id, defender_id}
            or defender_partner_id in {challenger_id, defender_id}
            or challenger_partner_id == defender_partner_id
            or challenger_partner_id not in id_to_name
            or defender_partner_id not in id_to_name
        ):
            return None
        score_challenger = _safe_int(match.get("score_t1"))
        score_defender = _safe_int(match.get("score_t2"))
        if score_challenger is None or score_defender is None:
            return None
        public_matches.append(
            {
                "slot": slot,
                "match_id": match_id,
                "date": _json_safe(match.get("date")),
                "score_challenger_team": score_challenger,
                "score_defender_team": score_defender,
                "challenger_partner": {
                    "player_id": challenger_partner_id,
                    "player_name": id_to_name[challenger_partner_id],
                },
                "defender_partner": {
                    "player_id": defender_partner_id,
                    "player_name": id_to_name[defender_partner_id],
                },
                "rating_changes": _public_rating_changes(
                    match,
                    id_to_name=id_to_name,
                ),
            }
        )
    if (
        public_matches[0]["challenger_partner"]["player_id"]
        != public_matches[1]["defender_partner"]["player_id"]
        or public_matches[0]["defender_partner"]["player_id"]
        != public_matches[1]["challenger_partner"]["player_id"]
    ):
        return None
    return {
        "version": 1,
        "completeness": "full",
        "rank_change": {
            "swapped": swapped,
            "challenger": challenger_change,
            "defender": defender_change,
        },
        "matches": public_matches,
    }


def _legacy_public_result_details(
    *,
    challenge: dict[str, Any],
    id_to_name: dict[int, str],
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Project only explicitly verified legacy score rows, never inferred links."""

    if str(challenge.get("status") or "") != "COMPLETED":
        return None
    challenge_id = _safe_int(challenge.get("id"))
    challenger_id = _safe_int(challenge.get("challenger_id"))
    defender_id = _safe_int(challenge.get("defender_id"))
    if (
        challenge_id is None
        or challenger_id is None
        or defender_id is None
        or challenger_id not in id_to_name
        or defender_id not in id_to_name
    ):
        return None
    rows = [
        row
        for row in rows
        if _safe_int(row.get("challenge_id")) == challenge_id
        and row.get("verified") is True
    ]
    rows.sort(key=lambda row: _safe_int(row.get("match_no"), 999) or 999)
    seen_match_numbers: set[int] = set()
    matches: list[dict[str, Any]] = []
    warnings: list[str] = []
    ranked_ids = {challenger_id, defender_id}
    for row in rows:
        match_no = _safe_int(row.get("match_no"))
        if match_no not in {1, 2} or match_no in seen_match_numbers:
            continue
        match_label = "A" if match_no == 1 else "B"
        games: list[dict[str, int]] = []
        for game_no in (1, 2, 3):
            challenger_score = _safe_int(row.get(f"g{game_no}_chal"))
            defender_score = _safe_int(row.get(f"g{game_no}_def"))
            if challenger_score is None or defender_score is None:
                continue
            games.append(
                {
                    "game": game_no,
                    "challenger": challenger_score,
                    "defender": defender_score,
                }
            )
        if not games:
            continue
        challenger_partner_id = _safe_int(row.get("chal_partner_id"))
        defender_partner_id = _safe_int(row.get("def_partner_id"))

        def partner_payload(player_id: int | None, *, side: str) -> dict[str, Any] | None:
            if player_id is None:
                warnings.append(f"Partner information is missing for Match {match_label} ({side} team).")
                return None
            if player_id in ranked_ids:
                warnings.append(
                    f"Partner information for Match {match_label} ({side} team) may be incorrect "
                    "and is hidden while club staff reviews it."
                )
                return None
            if player_id not in id_to_name:
                warnings.append(
                    f"The partner listed for Match {match_label} ({side} team) is no longer "
                    "on the club roster, so the name is hidden while club staff reviews it."
                )
                return None
            return {
                "player_id": player_id,
                "player_name": id_to_name[player_id],
            }

        seen_match_numbers.add(match_no)
        matches.append(
            {
                "slot": "a" if match_no == 1 else "b",
                "match_id": None,
                "date": None,
                "score_challenger_team": sum(game["challenger"] for game in games),
                "score_defender_team": sum(game["defender"] for game in games),
                "games": games,
                "challenger_partner": partner_payload(
                    challenger_partner_id,
                    side="challenger",
                ),
                "defender_partner": partner_payload(
                    defender_partner_id,
                    side="defender",
                ),
                "rating_changes": [],
            }
        )
    if not matches:
        return None
    return {
        "version": 1,
        "completeness": "partial",
        "rank_change": None,
        "matches": matches,
        "notice": (
            f"Details are available for {len(matches)} of 2 matches. "
            "Ratings and rank changes are not available for this older result."
        ),
        "warnings": warnings,
    }


def _result_details_for_challenge(
    *,
    challenge: dict[str, Any],
    id_to_name: dict[int, str],
    forward_matches: dict[int, dict[str, Any]],
    legacy_rows_by_challenge: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Never substitute legacy evidence for an explicit forward relation."""

    if "public_result_json" not in challenge:
        return None
    if challenge.get("public_result_json") is not None:
        return _public_result_details(
            challenge=challenge,
            id_to_name=id_to_name,
            forward_matches=forward_matches,
        )
    return _legacy_public_result_details(
        challenge=challenge,
        id_to_name=id_to_name,
        rows=legacy_rows_by_challenge.get(
            _safe_int(challenge.get("id"), -1) or -1,
            [],
        ),
    )


def _challenge_sections(
    supabase: Any,
    *,
    club_id: str,
    challenges: list[dict[str, Any]],
    id_to_name: dict[int, str],
    public_player_state: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in challenges:
        bucket = ladder_bucket_challenge(row)
        if bucket not in PUBLIC_BUCKETS:
            continue
        grouped[bucket].append(row)

    recently_completed = grouped.get("Recently Completed", [])
    recently_completed.sort(
        key=lambda row: (
            str(_json_safe(row.get("completed_at")) or ""),
            str(_json_safe(row.get("created_at")) or ""),
            _safe_int(row.get("id"), -1) or -1,
        ),
        reverse=True,
    )
    grouped["Recently Completed"] = recently_completed[:PUBLIC_RECENT_COMPLETED_LIMIT]
    visible_challenges = [
        row for bucket in PUBLIC_BUCKETS for row in grouped.get(bucket, [])
    ]
    forward_matches = _load_forward_result_matches(
        supabase,
        club_id=club_id,
        challenges=visible_challenges,
    )
    legacy_rows_by_challenge = _load_legacy_result_rows(
        supabase,
        club_id=club_id,
        challenges=visible_challenges,
    )

    return [
        {
            "name": bucket,
            "challenges": [
                _challenge_payload(
                    row,
                    id_to_name,
                    public_player_state,
                    _result_details_for_challenge(
                        challenge=row,
                        id_to_name=id_to_name,
                        forward_matches=forward_matches,
                        legacy_rows_by_challenge=legacy_rows_by_challenge,
                    ),
                )
                for row in grouped.get(bucket, [])
            ],
        }
        for bucket in PUBLIC_BUCKETS
    ]


def _quick_rules(settings: dict[str, int]) -> list[str]:
    return [
        "Challenge someone ranked above you within your tier.",
        f"You may challenge up to {int(settings.get('challenge_range', 7))} ranks higher when both players are eligible.",
        "One active challenge at a time: players in an open challenge are locked until it resolves.",
        f"Defenders have {int(settings.get('accept_window_hours', 48))} hours to accept and {int(settings.get('play_window_days', 7))} days after acceptance to play.",
        f"After a completed challenge, winners are protected and non-winners cool down for {int(settings.get('protected_hours', 72))} hours.",
        "Club staff handles challenges, scores, forfeits, passes, and rank changes.",
    ]


def _rulebook(settings: dict[str, int]) -> list[dict[str, Any]]:
    challenge_range = int(settings.get("challenge_range", 7))
    accept_hours = int(settings.get("accept_window_hours", 48))
    play_days = int(settings.get("play_window_days", 7))
    cooldown_hours = int(settings.get("cooldown_hours", 72))
    protected_hours = int(settings.get("protected_hours", 72))
    return [
        {
            "title": "How to make a challenge",
            "rules": [
                {
                    "title": "Check both statuses",
                    "body": "Your status shows whether you can challenge or be challenged. Ask club staff to confirm.",
                },
                {
                    "title": "Pick an eligible opponent",
                    "body": f"Choose a player ranked above you in the same tier, no more than {challenge_range} ranks away. Both players must be eligible by status.",
                },
                {
                    "title": "Make it official",
                    "body": "Ask club staff to record the challenge before you play.",
                },
                {
                    "title": "Defender response",
                    "body": f"The defender has {accept_hours} hours to accept. If they do not respond and have not used a monthly pass, club staff may record a forfeit.",
                },
                {
                    "title": "Play and report",
                    "body": f"Once accepted, play within {play_days} days and send the scores to club staff.",
                },
            ],
        },
        {
            "title": "Eligibility and timing",
            "rules": [
                {
                    "title": "One active challenge",
                    "body": "A ranked player may be involved in only one open challenge at a time, as challenger or defender.",
                },
                {
                    "title": "Monthly pass",
                    "body": f"A defender may use one pass each month without losing rank. Club staff must record it before the response deadline, and challenges are then paused for {int(settings.get('pass_hold_hours', 72))} hours.",
                },
                {
                    "title": "Missed deadlines",
                    "body": "If the defender does not respond, club staff may record a forfeit. If the match is not played by the deadline, staff will review scheduling attempts and decide what happens.",
                },
                {
                    "title": "Post-result timers",
                    "body": f"A challenge winner is protected for {protected_hours} hours; the non-winner cools down for {cooldown_hours} hours.",
                },
            ],
        },
        {
            "title": "Swing Partner Swap format",
            "rules": [
                {
                    "title": "Two doubles matches",
                    "body": "Each ranked player brings a swing partner. The ranked players remain opponents for two doubles matches, and swing partners swap between matches.",
                },
                {
                    "title": "Challenge winner",
                    "body": "Win both matches to win the challenge. If split, compare total games won, then total point differential; an exact tie favors the defender.",
                },
                {
                    "title": "Rank movement",
                    "body": "If the challenger wins, the ranked challenger and defender swap ranks. A defender win leaves ranks unchanged. Swing partners never move.",
                },
            ],
        },
        {
            "title": "Help from club staff",
            "rules": [
                {
                    "title": "Vacation and reinstatement",
                    "body": "Club staff manages vacation status and reinstatement. Returning players may need to play a reinstatement match before joining challenges again.",
                },
                {
                    "title": "Disputes and enforcement",
                    "body": "Club staff resolves disputes and applies the ladder rules for deadlines, passes, forfeits, and results.",
                },
            ],
        },
    ]


def _attach_public_eligibility(tiers: list[dict[str, Any]], *, challenge_range: int) -> int:
    """Attach privacy-safe opponent hints computed by the Python ladder policy."""

    players: list[tuple[str, dict[str, Any]]] = []
    for tier in tiers:
        tier_id = str(tier.get("tier_id") or "")
        for player in tier.get("players") or []:
            players.append((tier_id, player))

    eligible_pair_count = 0
    for challenger_tier, challenger in players:
        challenger_status = str(challenger.get("status") or "")
        can_initiate = ladder_can_initiate_challenge(challenger_status)
        can_receive = ladder_can_receive_challenge(challenger_status)
        opponents: list[dict[str, Any]] = []
        if can_initiate:
            for defender_tier, defender in players:
                if int(defender.get("player_id") or -1) == int(challenger.get("player_id") or -2):
                    continue
                decision = ladder_pair_eligibility(
                    challenger_tier=challenger_tier,
                    challenger_rank=_safe_int(challenger.get("rank")),
                    challenger_status=challenger_status,
                    defender_tier=defender_tier,
                    defender_rank=_safe_int(defender.get("rank")),
                    defender_status=str(defender.get("status") or ""),
                    challenge_range=int(challenge_range),
                )
                if not decision["eligible"]:
                    continue
                opponents.append(
                    {
                        "player_id": int(defender["player_id"]),
                        "player_name": str(defender.get("player_name") or "Player"),
                        "rank": _safe_int(defender.get("rank")),
                        "status": str(defender.get("status") or ""),
                        "status_short": str(defender.get("status_short") or defender.get("status") or ""),
                        "rank_gap": _safe_int(decision.get("rank_gap")),
                    }
                )
            opponents.sort(key=lambda item: (int(item.get("rank") or 999999), str(item.get("player_name") or "").casefold()))
        eligible_pair_count += len(opponents)
        challenger["eligibility"] = {
            "authority": "python",
            "can_initiate": can_initiate,
            "can_receive": can_receive,
            "eligible_opponents": opponents,
            "hint": (
                f"{len(opponents)} eligible opponent{'s' if len(opponents) != 1 else ''} currently visible in range."
                if can_initiate
                else "Current status does not allow initiating a challenge."
            ),
        }
    return eligible_pair_count


def build_public_challenge_ladder(supabase: Any, *, club_id: str) -> dict[str, Any]:
    """Build a public-safe Challenge Ladder payload for one club."""

    cid = str(club_id).strip()
    settings = _settings(supabase, club_id=cid)
    id_to_name, active_ids, rating_map = _player_maps(supabase, club_id=cid)
    roster = _roster_rows(supabase, club_id=cid)
    flags = _flag_rows(supabase, club_id=cid)
    challenges = _challenge_rows(supabase, club_id=cid)
    passes = _pass_rows(supabase, club_id=cid)

    df_roster = _frame(roster, ["player_id", "tier_id", "rank", "is_active"])
    if not df_roster.empty and "tier_id" in df_roster.columns:
        df_roster["tier_id"] = df_roster["tier_id"].astype(str).apply(normalize_tier_id)
    df_flags = _frame(flags, ["player_id", "vacation_until", "reinstate_required", "reinstate_notes"])
    df_challenges = _frame(challenges, ["id", "challenger_id", "defender_id", "status", "created_at", "accept_by", "accepted_at", "play_by", "completed_at", "winner_id"])
    df_passes = _frame(passes, ["player_id", "used_at"])

    status_map = ladder_compute_status_map(
        df_roster=df_roster,
        df_flags=df_flags,
        df_ch=df_challenges,
        df_pass=df_passes,
        settings=settings,
        id_to_name=id_to_name,
    )
    tiers = [_tier_payload(tid, roster, status_map, id_to_name, active_ids, rating_map) for tid in TIER_ORDER]
    populated = [tier for tier in tiers if tier["players"]]
    eligible_pair_count = _attach_public_eligibility(
        tiers,
        challenge_range=int(settings.get("challenge_range", 7)),
    )
    public_player_state = {
        int(player["player_id"]): player
        for tier in tiers
        for player in tier["players"]
        if _safe_int(player.get("player_id")) is not None
    }
    challenge_sections = _challenge_sections(
        supabase,
        club_id=cid,
        challenges=challenges,
        id_to_name=id_to_name,
        public_player_state=public_player_state,
    )

    active_challenge_count = sum(
        len(section["challenges"])
        for section in challenge_sections
        if section["name"] != "Recently Completed"
    )
    return {
        "settings": settings,
        "summary": {
            "tier_count": len(TIER_ORDER),
            "active_player_count": sum(len(tier["players"]) for tier in tiers),
            "populated_tier_count": len(populated),
            "active_challenge_count": active_challenge_count,
            "eligible_pair_count": eligible_pair_count,
        },
        "tiers": tiers,
        "challenge_sections": challenge_sections,
        "quick_rules": _quick_rules(settings),
        "rulebook": _rulebook(settings),
        "status_legend": PUBLIC_STATUS_LEGEND,
        "eligibility_authority": "python",
    }
