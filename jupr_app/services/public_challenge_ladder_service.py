from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
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
PUBLIC_STATUS_LEGEND = [
    {
        "status": "Ready to Defend",
        "short": "Ready",
        "can_initiate": True,
        "can_receive": True,
        "meaning": "Normal ladder mode: may initiate and receive an otherwise eligible challenge.",
    },
    {
        "status": "Protected",
        "short": "Protected",
        "can_initiate": True,
        "can_receive": False,
        "meaning": "May initiate, but cannot be challenged during the post-win protection window.",
    },
    {
        "status": "Cooldown",
        "short": "Cooldown",
        "can_initiate": False,
        "can_receive": True,
        "meaning": "May receive, but cannot initiate during the post-result cooldown window.",
    },
    {
        "status": "Locked",
        "short": "Locked",
        "can_initiate": False,
        "can_receive": False,
        "meaning": "Already involved in an open challenge; cannot start or receive another.",
    },
    {
        "status": "Pass Hold",
        "short": "Pass Hold",
        "can_initiate": False,
        "can_receive": False,
        "meaning": "A monthly pass was used and ladder activity is paused for the configured hold.",
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
        "meaning": "Staff-managed reinstatement is required before normal ladder activity resumes.",
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
        return "Staff review required before ladder activity."
    if status_name == "Vacation":
        return "Temporarily unavailable."
    if status_name == "Pass Hold":
        return "Monthly pass timing hold."
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
    rows = _fetch_table(
        supabase,
        "ladder_challenges",
        "id,club_id,challenger_id,defender_id,tier_id,status,created_at,accept_by,accepted_at,play_by,completed_at,winner_id",
        club_id=club_id,
        limit=5000,
    )
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


def _challenge_payload(row: dict[str, Any], id_to_name: dict[int, str]) -> dict[str, Any]:
    challenger_id = _safe_int(row.get("challenger_id"))
    defender_id = _safe_int(row.get("defender_id"))
    winner_id = _safe_int(row.get("winner_id"))
    return {
        "id": _safe_int(row.get("id")),
        "tier_id": normalize_tier_id(str(row.get("tier_id") or "")),
        "status": str(row.get("status") or ""),
        "bucket": ladder_bucket_challenge(row),
        "challenger": {
            "player_id": challenger_id,
            "player_name": id_to_name.get(int(challenger_id), f"Player {challenger_id}") if challenger_id is not None else "—",
        },
        "defender": {
            "player_id": defender_id,
            "player_name": id_to_name.get(int(defender_id), f"Player {defender_id}") if defender_id is not None else "—",
        },
        "winner": {
            "player_id": winner_id,
            "player_name": id_to_name.get(int(winner_id), f"Player {winner_id}") if winner_id is not None else None,
        }
        if winner_id is not None
        else None,
        "created_at": _json_safe(row.get("created_at")),
        "accept_by": _json_safe(row.get("accept_by")),
        "play_by": _json_safe(row.get("play_by")),
        "completed_at": _json_safe(row.get("completed_at")),
    }


def _challenge_sections(challenges: list[dict[str, Any]], id_to_name: dict[int, str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in challenges:
        bucket = ladder_bucket_challenge(row)
        if bucket not in PUBLIC_BUCKETS:
            continue
        if bucket == "Recently Completed" and len(grouped[bucket]) >= 12:
            continue
        grouped[bucket].append(_challenge_payload(row, id_to_name))
    return [{"name": bucket, "challenges": grouped.get(bucket, [])} for bucket in PUBLIC_BUCKETS]


def _quick_rules(settings: dict[str, int]) -> list[str]:
    return [
        "Challenge someone ranked above you within your tier.",
        f"You may challenge up to {int(settings.get('challenge_range', 7))} ranks higher when both players are eligible.",
        "One active challenge at a time: players in an open challenge are locked until it resolves.",
        f"Defenders have {int(settings.get('accept_window_hours', 48))} hours to accept and {int(settings.get('play_window_days', 7))} days after acceptance to play.",
        f"After a completed challenge, winners are protected and non-winners cool down for {int(settings.get('protected_hours', 72))} hours.",
        "Official challenge creation, score entry, forfeits, passes, and rank movement remain staff-managed.",
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
                    "body": "Your computed public status controls whether you may initiate or receive a challenge. Staff owns the final official decision.",
                },
                {
                    "title": "Pick an eligible opponent",
                    "body": f"Choose a player ranked above you in the same tier, no more than {challenge_range} ranks away. Both players must be eligible by status.",
                },
                {
                    "title": "Make it official",
                    "body": "A challenge becomes official only after authorized staff records it in the Challenge Ledger.",
                },
                {
                    "title": "Defender response",
                    "body": f"The defender has {accept_hours} hours to accept. No response without a recorded monthly pass may become a forfeit.",
                },
                {
                    "title": "Play and report",
                    "body": f"Once accepted, complete the match within {play_days} days and submit scores to staff for ledger verification.",
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
                    "body": f"A defender may use one pass per calendar month without losing rank when staff records it during the acceptance window; the configured hold is {int(settings.get('pass_hold_hours', 72))} hours.",
                },
                {
                    "title": "Missed deadlines",
                    "body": "No response can become a forfeit. A missed play deadline can require a staff-determined outcome based on good-faith scheduling and the official ledger.",
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
            "title": "Staff-managed exceptions",
            "rules": [
                {
                    "title": "Vacation and reinstatement",
                    "body": "Staff manages vacation status and reinstatement requirements. Returning players may need a reinstatement match before normal activity resumes.",
                },
                {
                    "title": "Disputes and enforcement",
                    "body": "Authorized ladder staff resolves disputes and enforces timing, pass, forfeit, and result rules using the Challenge Ledger as the official record.",
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

    active_challenge_count = sum(len(section["challenges"]) for section in _challenge_sections(challenges, id_to_name) if section["name"] != "Recently Completed")
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
        "challenge_sections": _challenge_sections(challenges, id_to_name),
        "quick_rules": _quick_rules(settings),
        "rulebook": _rulebook(settings),
        "status_legend": PUBLIC_STATUS_LEGEND,
        "eligibility_authority": "python",
    }
