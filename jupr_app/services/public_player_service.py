from __future__ import annotations

from jupr_app.data.paged_reads import read_all_rows
from jupr_app.domain.gamification.presentation import badge_category, badge_requirement, category_sort_key

import json
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any

from jupr_app.services.public_league_visibility import league_is_public

PLAYER_SELECT = "id,club_id,name,rating,starting_rating,wins,losses,matches_played,active,last_game_at,inactive_at,singles_rating,singles_wins,singles_losses,singles_matches_played,singles_last_game_at"
PLAYER_BASE_SELECT = "id,club_id,name,rating,wins,losses,matches_played,active,last_game_at,inactive_at,singles_rating,singles_wins,singles_losses,singles_matches_played,singles_last_game_at"
PLAYER_MINIMAL_SELECT = "id,club_id,name,rating,wins,losses,matches_played"
LEAGUE_RATINGS_SELECT = "id,club_id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active"
LEAGUE_META_VISIBILITY_SELECT = "club_id,league_name,is_active,status"
MATCH_SELECT = "*"
PLAYER_BADGE_SELECT = "club_id,player_id,badge_id,earned_at,context_type,context_id,value_num,value_json,revoked_at"
PLAYER_BADGE_FALLBACK_SELECT = "club_id,player_id,badge_id,earned_at,context_type,context_id,value_num,value_json"
BADGE_SELECT = "badge_id,name,category,prestige,rarity,tier,icon_key,lore,hint,scope,state,is_active"

DIRECTORY_STATUSES = {"active", "inactive", "all"}
DIRECTORY_SORTS = {"rating", "singles", "matches", "name", "win_pct", "recent"}
PUBLIC_PROFILE_HISTORY_LIMIT = 500
PUBLIC_PROFILE_RECENT_LIMIT = 12
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_FORMAL_TROPHY_CONTEXTS = {
    "league",
    "league_award",
    "league_end",
    "league_end_award",
    "season_award",
    "tournament",
    "tournament_award",
    "tournament_podium",
    "podium",
}
_FORMAL_TROPHY_CONTEXT_TOKENS = (":podium:", ":top_performer:", ":league_award:")
_REPEATABLE_BADGE_TOKENS = (
    "level_up",
    "hot_streak",
    "rocket_start",
    "blowout_artist",
    "bounce_back",
    "first_win",
    "participant",
)


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _plain_text(value: Any, *, limit: int = 240) -> str | None:
    text = str(value or "").replace("<", "").replace(">", "").strip()
    if not text:
        return None
    text = _HTML_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()[: max(1, int(limit))] or None


def _to_jupr(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return number / 400.0 if abs(number) > 20 else number


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


def _is_active_player(row: dict[str, Any]) -> bool:
    if row.get("inactive_at"):
        return False
    if "active" in row and row.get("active") is False:
        return False
    return True


def _player_base(row: dict[str, Any]) -> dict[str, Any]:
    wins = _int_or_none(row.get("wins")) or 0
    losses = _int_or_none(row.get("losses")) or 0
    matches_played = _int_or_none(row.get("matches_played"))
    if matches_played is None:
        matches_played = wins + losses
    singles_wins = _int_or_none(row.get("singles_wins")) or 0
    singles_losses = _int_or_none(row.get("singles_losses")) or 0
    singles_matches_played = _int_or_none(row.get("singles_matches_played"))
    if singles_matches_played is None:
        singles_matches_played = singles_wins + singles_losses
    display_name = _plain_text(row.get("name"), limit=160) or "Player"
    return {
        "id": _int_or_none(row.get("id")) or row.get("id"),
        "name": display_name,
        "display_name": display_name,
        "rating": _float_or_none(row.get("rating")),
        "rating_jupr": _to_jupr(row.get("rating")),
        "starting_rating": _float_or_none(row.get("starting_rating")),
        "starting_rating_jupr": _to_jupr(row.get("starting_rating")),
        "wins": wins,
        "losses": losses,
        "matches_played": matches_played,
        "singles_rating": _float_or_none(row.get("singles_rating")),
        "singles_rating_jupr": _to_jupr(row.get("singles_rating")),
        "singles_wins": singles_wins,
        "singles_losses": singles_losses,
        "singles_matches_played": singles_matches_played,
        "singles_last_game_at": _json_safe(row.get("singles_last_game_at")),
        "is_active": _is_active_player(row),
        "last_game_at": _json_safe(row.get("last_game_at")),
    }


def _public_league_rating(row: dict[str, Any]) -> dict[str, Any]:
    wins = _int_or_none(row.get("wins")) or 0
    losses = _int_or_none(row.get("losses")) or 0
    matches_played = _int_or_none(row.get("matches_played"))
    if matches_played is None:
        matches_played = wins + losses
    rating_jupr = _to_jupr(row.get("rating"))
    starting_jupr = _to_jupr(row.get("starting_rating"))
    return {
        "id": row.get("id"),
        "league_name": _plain_text(row.get("league_name"), limit=120),
        "rating": _float_or_none(row.get("rating")),
        "rating_jupr": rating_jupr,
        "starting_rating": _float_or_none(row.get("starting_rating")),
        "starting_rating_jupr": starting_jupr,
        "rating_gain_jupr": (
            round(float(rating_jupr) - float(starting_jupr), 6)
            if rating_jupr is not None and starting_jupr is not None
            else None
        ),
        "wins": wins,
        "losses": losses,
        "matches_played": matches_played,
        "is_active": row.get("is_active", True),
    }


def _fetch_players(supabase: Any, club_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("players")
            .select(PLAYER_SELECT)
            .eq("club_id", club_id)
            .execute()
        )
    except Exception:
        try:
            return _safe_rows(
                supabase.table("players")
                .select(PLAYER_BASE_SELECT)
                .eq("club_id", club_id)
                .execute()
            )
        except Exception:
            return _safe_rows(
                supabase.table("players")
                .select(PLAYER_MINIMAL_SELECT)
                .eq("club_id", club_id)
                .execute()
            )


def _fetch_player(supabase: Any, club_id: str, player_id: int | str) -> dict[str, Any] | None:
    try:
        row = _safe_first(
            supabase.table("players")
            .select(PLAYER_SELECT)
            .eq("club_id", club_id)
            .eq("id", player_id)
            .limit(1)
            .execute()
        )
    except Exception:
        try:
            row = _safe_first(
                supabase.table("players")
                .select(PLAYER_BASE_SELECT)
                .eq("club_id", club_id)
                .eq("id", player_id)
                .limit(1)
                .execute()
            )
        except Exception:
            row = _safe_first(
                supabase.table("players")
                .select(PLAYER_MINIMAL_SELECT)
                .eq("club_id", club_id)
                .eq("id", player_id)
                .limit(1)
                .execute()
            )
    return row


def _fetch_league_ratings(supabase: Any, club_id: str, player_id: int | str | None = None) -> list[dict[str, Any]]:
    query = supabase.table("league_ratings").select(LEAGUE_RATINGS_SELECT).eq("club_id", club_id)
    if player_id is not None:
        query = query.eq("player_id", player_id)
    try:
        return _safe_rows(query.execute())
    except Exception:
        return []


def _public_league_names(supabase: Any, club_id: str) -> set[str]:
    try:
        rows = _safe_rows(
            supabase.table("leagues_metadata")
            .select(LEAGUE_META_VISIBILITY_SELECT)
            .eq("club_id", str(club_id))
            .execute()
        )
    except Exception:
        return set()
    return {
        str(row.get("league_name") or "").strip()
        for row in rows
        if league_is_public(row)
    }


def _fetch_recent_matches(supabase: Any, club_id: str, *, limit: int = 300) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select(MATCH_SELECT)
            .eq("club_id", club_id)
            .order("date", desc=True)
            .limit(int(limit))
            .execute()
        )
        return [row for row in rows if not row.get("deleted_at")]
    except Exception:
        try:
            rows = _safe_rows(
                supabase.table("matches")
                .select(MATCH_SELECT)
                .eq("club_id", club_id)
                .order("id", desc=True)
                .limit(int(limit))
                .execute()
            )
            return [row for row in rows if not row.get("deleted_at")]
        except Exception:
            return []


def _fetch_match(supabase: Any, club_id: str, match_id: int | str) -> dict[str, Any] | None:
    try:
        row = _safe_first(
            supabase.table("matches")
            .select(MATCH_SELECT)
            .eq("club_id", club_id)
            .eq("id", match_id)
            .limit(1)
            .execute()
        )
        return row if row and not row.get("deleted_at") else None
    except Exception:
        return None


def _match_includes_player(row: dict[str, Any], player_id: int | str) -> bool:
    pid = str(player_id)
    return any(str(row.get(col)) == pid for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"))


def _match_sort_key(row: dict[str, Any]) -> tuple[str, tuple[int, int | str]]:
    raw_id = row.get("id")
    numeric_id = _int_or_none(raw_id)
    id_key: tuple[int, int | str] = (0, numeric_id) if numeric_id is not None else (1, str(raw_id or ""))
    return str(row.get("date") or ""), id_key


def _player_position(row: dict[str, Any], player_id: int | str) -> tuple[str, str] | None:
    pid = str(player_id)
    for position, team in (("t1_p1", "team_1"), ("t1_p2", "team_1"), ("t2_p1", "team_2"), ("t2_p2", "team_2")):
        if str(row.get(position)) == pid:
            return position, team
    return None


def _public_match_format(row: dict[str, Any]) -> tuple[str, str]:
    explicit = str(row.get("match_format") or row.get("rating_scope") or "").strip().casefold()
    singles = explicit in {"single", "singles"} or (row.get("t1_p2") in (None, "") and row.get("t2_p2") in (None, ""))
    return ("singles", "Singles") if singles else ("doubles", "Doubles")


def _match_result(row: dict[str, Any], team: str) -> str | None:
    score_t1 = _int_or_none(row.get("score_t1"))
    score_t2 = _int_or_none(row.get("score_t2"))
    if score_t1 is None or score_t2 is None or score_t1 == score_t2:
        return None
    team_one_won = score_t1 > score_t2
    return "win" if (team == "team_1") == team_one_won else "loss"


def _player_ref(pid: Any, name_by_id: dict[str, str]) -> dict[str, Any]:
    return {"id": pid, "name": name_by_id.get(str(pid), "Player")}


def _rating_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "team_1": [
            {
                "player_id": row.get("t1_p1"),
                "start_rating": _float_or_none(row.get("t1_p1_r")),
                "end_rating": _float_or_none(row.get("t1_p1_r_end")),
            },
            {
                "player_id": row.get("t1_p2"),
                "start_rating": _float_or_none(row.get("t1_p2_r")),
                "end_rating": _float_or_none(row.get("t1_p2_r_end")),
            },
        ],
        "team_2": [
            {
                "player_id": row.get("t2_p1"),
                "start_rating": _float_or_none(row.get("t2_p1_r")),
                "end_rating": _float_or_none(row.get("t2_p1_r_end")),
            },
            {
                "player_id": row.get("t2_p2"),
                "start_rating": _float_or_none(row.get("t2_p2_r")),
                "end_rating": _float_or_none(row.get("t2_p2_r_end")),
            },
        ],
    }


def _public_match(row: dict[str, Any], name_by_id: dict[str, str], *, include_rating_snapshot: bool = False) -> dict[str, Any]:
    t1_ids = [row.get("t1_p1"), row.get("t1_p2")]
    t2_ids = [row.get("t2_p1"), row.get("t2_p2")]
    score_t1 = _int_or_none(row.get("score_t1"))
    score_t2 = _int_or_none(row.get("score_t2"))
    winner = None
    if score_t1 is not None and score_t2 is not None and score_t1 != score_t2:
        winner = "team_1" if score_t1 > score_t2 else "team_2"
    match_format, match_format_label = _public_match_format(row)
    payload: dict[str, Any] = {
        "id": row.get("id"),
        "date": _json_safe(row.get("date")),
        "league": _plain_text(row.get("league"), limit=120),
        "week_tag": _plain_text(row.get("week_tag"), limit=80),
        "match_type": _plain_text(row.get("match_type"), limit=80),
        "match_format": match_format,
        "match_format_label": match_format_label,
        "rating_scope": _plain_text(row.get("rating_scope"), limit=80),
        "context_type": _plain_text(row.get("context_type"), limit=80),
        "team_1": [_player_ref(pid, name_by_id) for pid in t1_ids if pid is not None],
        "team_2": [_player_ref(pid, name_by_id) for pid in t2_ids if pid is not None],
        "score_t1": score_t1,
        "score_t2": score_t2,
        "winner": winner,
        "elo_delta": _float_or_none(row.get("elo_delta")),
    }
    if include_rating_snapshot:
        payload["rating_snapshot"] = _rating_snapshot(row)
    return payload


def _fetch_player_badges(supabase: Any, *, club_id: str, player_id: int | str) -> list[dict[str, Any]]:
    rows = read_all_rows(lambda: supabase.table("player_badges").select(PLAYER_BADGE_SELECT)
                         .eq("club_id", str(club_id)).eq("player_id", player_id))
    return [row for row in rows if not row.get("revoked_at")]

def _fetch_badge_definitions(supabase: Any) -> dict[str, dict[str, Any]]:
    try:
        rows = _safe_rows(supabase.table("badges").select(BADGE_SELECT).execute())
    except Exception:
        try:
            rows = _safe_rows(supabase.table("badges").select("badge_id,name,category,prestige,is_active").execute())
        except Exception:
            return {}
    return {str(row.get("badge_id") or "").strip(): row for row in rows if str(row.get("badge_id") or "").strip()}


def _placement_from_award(row: dict[str, Any]) -> int | None:
    value_json = _json_object(row.get("value_json"))
    placement = _int_or_none(value_json.get("placement")) or _int_or_none(row.get("value_num"))
    if placement is not None:
        return placement
    match = re.search(r":podium:(\d+)$", str(row.get("context_id") or ""))
    return _int_or_none(match.group(1)) if match else None


def _award_context_label(row: dict[str, Any]) -> str | None:
    value_json = _json_object(row.get("value_json"))
    for key in ("tournament_name", "league_name", "league", "season_label"):
        label = _plain_text(value_json.get(key), limit=160)
        if label:
            return label
    context_type = _plain_text(row.get("context_type"), limit=80)
    if context_type:
        return context_type.replace("_", " ").title()
    return None


def _is_formal_trophy(row: dict[str, Any], badge_id: str) -> bool:
    """Keep the public Trophy Case for one-time major honors only.

    Repeatable progression badges can be issued in a league or tournament
    context too, so context alone is not enough.  A trophy needs durable
    evidence of an end-of-league award or a tournament podium placement.
    """

    normalized_badge_id = badge_id.casefold()
    if any(token in normalized_badge_id for token in _REPEATABLE_BADGE_TOKENS):
        return False
    context_type = str(row.get("context_type") or "").strip().casefold()
    if context_type not in _FORMAL_TROPHY_CONTEXTS:
        return False
    context_id = str(row.get("context_id") or "").casefold()
    value_json = _json_object(row.get("value_json"))
    if _placement_from_award(row) is not None:
        return True
    if any(token in context_id for token in _FORMAL_TROPHY_CONTEXT_TOKENS):
        return True
    return any(
        value_json.get(key) not in (None, "")
        for key in ("award_key", "category_key", "category_label", "podium_place")
    )


def _public_awards(supabase: Any, *, club_id: str, player_id: int | str) -> dict[str, Any]:
    player_badges = _fetch_player_badges(supabase, club_id=club_id, player_id=player_id)
    definitions = _fetch_badge_definitions(supabase)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in player_badges:
        badge_id = str(row.get("badge_id") or "").strip()
        if badge_id:
            grouped[badge_id].append(row)

    badges: list[dict[str, Any]] = []
    trophies: list[dict[str, Any]] = []
    prestige_total = 0
    for badge_id, rows in grouped.items():
        definition = definitions.get(badge_id, {})
        badge_name = _plain_text(definition.get("name"), limit=160) or badge_id.replace("_", " ").title()
        if definition.get("is_active") is False and not rows:
            continue
        prestige = _int_or_none(definition.get("prestige")) or 0
        trophy_rows = [row for row in rows if _is_formal_trophy(row, badge_id)]
        cabinet_rows = [row for row in rows if row not in trophy_rows]
        if cabinet_rows:
            prestige_total += prestige * len(cabinet_rows)
            earned_values = [str(row.get("earned_at") or "") for row in cabinet_rows]
            badges.append(
                {
                    "badge_id": badge_id,
                    "name": badge_name,
                    "category": badge_category(badge_id),
                    "prestige": prestige,
                    "rarity": _plain_text(definition.get("rarity") or definition.get("tier"), limit=80),
                    "icon_key": _plain_text(definition.get("icon_key"), limit=80),
                    "description": badge_requirement(badge_id),
                    "requirements": badge_requirement(badge_id),
                    "count": len(cabinet_rows),
                    "last_earned_at": max(earned_values) or None,
                }
            )
        for row in trophy_rows:
            trophies.append(
                {
                    "badge_id": badge_id,
                    "title": badge_name,
                    "placement": _placement_from_award(row),
                    "context_type": _plain_text(row.get("context_type"), limit=80),
                    "context_label": _award_context_label(row),
                    "earned_at": _json_safe(row.get("earned_at")),
                }
            )

    badges.sort(key=lambda item: (category_sort_key(item["category"]), str(item.get("name") or "").casefold()))
    trophies.sort(key=lambda item: str(item.get("earned_at") or ""), reverse=True)
    return {
        "badge_count": len(badges),
        "badge_award_count": sum(int(item.get("count") or 0) for item in badges),
        "trophy_count": len(trophies),
        "prestige_total": int(prestige_total),
        "badges": badges,
        "trophies": trophies,
    }


def _verified_update_projection(supabase: Any, *, club_id: str, player_id: int | str) -> dict[str, Any]:
    try:
        rows = _safe_rows(
            supabase.table("player_profile_update_subscriptions")
            .select("request_status,verified_at,created_at")
            .eq("club_id", str(club_id))
            .eq("player_id", player_id)
            .execute()
        )
    except Exception:
        rows = []
    statuses = {str(row.get("request_status") or "").strip().casefold() for row in rows}
    if "active" in statuses:
        status = "enabled"
    elif "pending_admin_review" in statuses:
        status = "pending"
    else:
        status = "available"
    return {"status": status, "can_request": status == "available"}


def _relationship_rows(
    matches: list[dict[str, Any]],
    *,
    player_id: int | str,
    name_by_id: dict[str, str],
) -> dict[str, Any]:
    partner_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"matches": 0, "wins": 0, "losses": 0})
    rival_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"matches": 0, "wins": 0, "losses": 0})
    pid = str(player_id)
    for match in matches:
        position = _player_position(match, player_id)
        if position is None:
            continue
        _, team = position
        result = _match_result(match, team)
        own_columns = ("t1_p1", "t1_p2") if team == "team_1" else ("t2_p1", "t2_p2")
        opponent_columns = ("t2_p1", "t2_p2") if team == "team_1" else ("t1_p1", "t1_p2")
        for value in (match.get(column) for column in own_columns):
            key = str(value or "")
            if not key or key == pid:
                continue
            partner_stats[key]["matches"] += 1
            if result == "win":
                partner_stats[key]["wins"] += 1
            elif result == "loss":
                partner_stats[key]["losses"] += 1
        for value in (match.get(column) for column in opponent_columns):
            key = str(value or "")
            if not key:
                continue
            rival_stats[key]["matches"] += 1
            if result == "win":
                rival_stats[key]["wins"] += 1
            elif result == "loss":
                rival_stats[key]["losses"] += 1

    def public_rows(source: dict[str, dict[str, int]], *, rival: bool) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for raw_id, stats in source.items():
            matches_played = int(stats["matches"])
            wins = int(stats["wins"])
            losses = int(stats["losses"])
            output.append(
                {
                    "player_id": _int_or_none(raw_id) or raw_id,
                    "player_name": name_by_id.get(raw_id, "Player"),
                    "matches": matches_played,
                    "wins": wins,
                    "losses": losses,
                    "win_pct": round((wins / matches_played) * 100.0, 1) if matches_played else None,
                    "balance": round(abs((wins / matches_played) - 0.5), 6) if rival and matches_played else None,
                }
            )
        output.sort(
            key=(
                (lambda item: (-int(item["matches"]), float(item.get("balance") or 0), str(item["player_name"]).casefold()))
                if rival
                else (lambda item: (-int(item["matches"]), -float(item.get("win_pct") or 0), str(item["player_name"]).casefold()))
            )
        )
        return output[:5]

    partners = public_rows(partner_stats, rival=False)
    rivals = public_rows(rival_stats, rival=True)
    best_partner = sorted(partners, key=lambda item: (-float(item.get("win_pct") or 0), -int(item.get("matches") or 0), str(item.get("player_name") or "")))[0] if partners else None
    return {"best_partner": best_partner, "rival": rivals[0] if rivals else None, "partners": partners, "rivals": rivals}


def _rating_history(matches: list[dict[str, Any]], *, player_id: int | str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    ordered = sorted(matches, key=_match_sort_key)
    for index, row in enumerate(ordered, start=1):
        position = _player_position(row, player_id)
        if position is None:
            continue
        slot, team = position
        before = _to_jupr(row.get(f"{slot}_r"))
        after = _to_jupr(row.get(f"{slot}_r_end"))
        result = _match_result(row, team)
        raw_elo_delta = _float_or_none(row.get("elo_delta"))
        raw_delta = (raw_elo_delta / 400.0) if raw_elo_delta is not None else None
        if after is None and before is not None and raw_delta is not None and result:
            after = before + (abs(raw_delta) if result == "win" else -abs(raw_delta))
        delta = (after - before) if after is not None and before is not None else None
        match_format, format_label = _public_match_format(row)
        points.append(
            {
                "match_number": index,
                "match_id": row.get("id"),
                "date": _json_safe(row.get("date")),
                "league": _plain_text(row.get("league"), limit=120),
                "match_type": _plain_text(row.get("match_type"), limit=80),
                "match_format": match_format,
                "match_format_label": format_label,
                "result": result,
                "rating_before_jupr": round(before, 6) if before is not None else None,
                "rating_after_jupr": round(after, 6) if after is not None else None,
                "rating_delta_jupr": round(delta, 6) if delta is not None else None,
            }
        )
    return points


def _rating_projection(player: dict[str, Any], matches: list[dict[str, Any]], *, player_id: int | str) -> dict[str, Any]:
    history = _rating_history(matches, player_id=player_id)
    by_format: dict[str, dict[str, Any]] = {}
    for match_format, label in (("doubles", "Doubles / overall"), ("singles", "Singles")):
        points = [point for point in history if point["match_format"] == match_format]
        wins = sum(1 for point in points if point.get("result") == "win")
        losses = sum(1 for point in points if point.get("result") == "loss")
        deltas = [float(point["rating_delta_jupr"]) for point in points if point.get("rating_delta_jupr") is not None]
        by_format[match_format] = {
            "format": match_format,
            "label": label,
            "matches": len(points),
            "wins": wins,
            "losses": losses,
            "win_pct": round((wins / len(points)) * 100.0, 1) if points else None,
            "rating_delta_jupr": round(sum(deltas), 6) if deltas else None,
        }

    overall_history = [point for point in history if point.get("match_format") == "doubles"] or history
    known = [float(point["rating_after_jupr"]) for point in overall_history if point.get("rating_after_jupr") is not None]
    starts = [float(point["rating_before_jupr"]) for point in overall_history if point.get("rating_before_jupr") is not None]
    current_rating = _float_or_none(player.get("rating_jupr"))
    starting_rating = _float_or_none(player.get("starting_rating_jupr")) or (starts[0] if starts else None)
    overall_values = (
        known
        + starts
        + ([current_rating] if current_rating is not None else [])
        + ([starting_rating] if starting_rating is not None else [])
    )
    recent = history[-10:]
    recent_wins = sum(1 for point in recent if point.get("result") == "win")
    recent_losses = sum(1 for point in recent if point.get("result") == "loss")
    recent_deltas = [float(point["rating_delta_jupr"]) for point in recent if point.get("rating_delta_jupr") is not None]
    streak_result: str | None = None
    streak_count = 0
    for point in reversed(history):
        result = point.get("result")
        if result not in {"win", "loss"}:
            break
        if streak_result is None:
            streak_result = str(result)
        if result != streak_result:
            break
        streak_count += 1
    return {
        "summary": {
            "current_rating_jupr": player.get("rating_jupr"),
            "current_singles_rating_jupr": player.get("singles_rating_jupr"),
            "starting_rating_jupr": starting_rating,
            "highest_rating_jupr": max(overall_values) if overall_values else None,
            "lowest_rating_jupr": min(overall_values) if overall_values else None,
            "last_10_record": f"{recent_wins}-{recent_losses}",
            "last_10_delta_jupr": round(sum(recent_deltas), 6) if recent_deltas else None,
            "current_streak": (f"{'W' if streak_result == 'win' else 'L'}{streak_count}" if streak_result and streak_count else None),
        },
        "formats": [by_format["doubles"], by_format["singles"]],
        "history": history,
    }


def _social_projection(supabase: Any, *, club_id: str, player_id: int | str) -> dict[str, Any]:
    try:
        people = _safe_rows(
            supabase.table("club_people")
            .select("id,linked_player_id")
            .eq("club_id", str(club_id))
            .eq("linked_player_id", player_id)
            .execute()
        )
        participants = _safe_rows(
            supabase.table("live_event_participants")
            .select("id,event_id,club_person_id,linked_player_id")
            .eq("linked_player_id", player_id)
            .execute()
        )
        if not participants and people:
            people_ids = [row.get("id") for row in people if row.get("id")]
            participants = _safe_rows(
                supabase.table("live_event_participants")
                .select("id,event_id,club_person_id,linked_player_id")
                .in_("club_person_id", people_ids)
                .execute()
            )
        event_ids = sorted({str(row.get("event_id")) for row in participants if row.get("event_id")})
        if not event_ids:
            return {
                "available": True,
                "identity": {"linked": bool(people or participants), "label": "Club Social identity linked" if people or participants else "No linked Club Social identity"},
                "summary": {"events": 0, "matches": 0, "wins": 0, "losses": 0, "score_diff": 0, "last_appearance": None},
                "skill_breakdown": [],
                "recent_events": [],
            }
        events = _safe_rows(
            supabase.table("live_events")
            .select("id,name,event_type,event_date,status,result_mode,summary_json")
            .eq("club_id", str(club_id))
            .eq("result_mode", "social_unrated")
            .eq("status", "saved")
            .in_("id", event_ids)
            .execute()
        )
        allowed_event_ids = {str(row.get("id")) for row in events if row.get("id")}
        participant_ids = {str(row.get("id")) for row in participants if str(row.get("event_id")) in allowed_event_ids and row.get("id")}
        matches = _safe_rows(
            supabase.table("live_event_matches")
            .select("event_id,played_on,t1_p1_participant_id,t1_p2_participant_id,t2_p1_participant_id,t2_p2_participant_id,score_t1,score_t2")
            .in_("event_id", sorted(allowed_event_ids))
            .execute()
        ) if allowed_event_ids else []
    except Exception:
        return {"available": False, "identity": {"linked": False, "label": "Club Social history unavailable"}, "summary": None, "skill_breakdown": [], "recent_events": []}

    stats_by_event: dict[str, dict[str, int]] = defaultdict(lambda: {"matches": 0, "wins": 0, "losses": 0, "score_diff": 0})
    for row in matches:
        score_one = _int_or_none(row.get("score_t1"))
        score_two = _int_or_none(row.get("score_t2"))
        if score_one is None or score_two is None or score_one == score_two:
            continue
        team_one = {str(row.get("t1_p1_participant_id") or ""), str(row.get("t1_p2_participant_id") or "")}
        team_two = {str(row.get("t2_p1_participant_id") or ""), str(row.get("t2_p2_participant_id") or "")}
        on_one = bool(team_one & participant_ids)
        on_two = bool(team_two & participant_ids)
        if on_one == on_two:
            continue
        event_id = str(row.get("event_id") or "")
        bucket = stats_by_event[event_id]
        bucket["matches"] += 1
        won = (score_one > score_two) if on_one else (score_two > score_one)
        bucket["wins" if won else "losses"] += 1
        bucket["score_diff"] += (score_one - score_two) if on_one else (score_two - score_one)

    recent_events: list[dict[str, Any]] = []
    skill_buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"events": 0, "matches": 0, "wins": 0, "losses": 0, "score_diff": 0}
    )
    for event in events:
        event_id = str(event.get("id") or "")
        stats = stats_by_event[event_id]
        tags = _json_object(event.get("summary_json")).get("event_tags")
        raw_skills = tags.get("skill_levels") if isinstance(tags, dict) else None
        if isinstance(raw_skills, str):
            raw_skills = [raw_skills]
        skills = [_plain_text(value, limit=40) for value in (raw_skills or [])] if isinstance(raw_skills, (list, tuple, set)) else []
        skills = [value for value in skills if value] or ["All"]
        for skill in skills:
            bucket = skill_buckets[str(skill)]
            bucket["label"] = str(skill)
            bucket["events"] += 1
            bucket["matches"] += stats["matches"]
            bucket["wins"] += stats["wins"]
            bucket["losses"] += stats["losses"]
            bucket["score_diff"] += stats["score_diff"]
        recent_events.append(
            {
                "date": _json_safe(event.get("event_date")),
                "name": _plain_text(event.get("name"), limit=160) or "Club Social event",
                "event_type": _plain_text(event.get("event_type"), limit=80) or "Social",
                "skill_labels": skills,
                **stats,
            }
        )
    recent_events.sort(key=lambda item: str(item.get("date") or ""), reverse=True)
    totals = {
        "events": len(recent_events),
        "matches": sum(int(item["matches"]) for item in recent_events),
        "wins": sum(int(item["wins"]) for item in recent_events),
        "losses": sum(int(item["losses"]) for item in recent_events),
        "score_diff": sum(int(item["score_diff"]) for item in recent_events),
        "last_appearance": recent_events[0].get("date") if recent_events else None,
    }
    return {
        "available": True,
        "identity": {"linked": bool(people or participants), "label": "Club Social identity linked" if people or participants else "No linked Club Social identity"},
        "summary": totals,
        "skill_breakdown": sorted(skill_buckets.values(), key=lambda item: str(item.get("label") or "").casefold()),
        "recent_events": recent_events[:12],
    }


def get_public_players(
    supabase: Any,
    *,
    club_id: str,
    search: str | None = None,
    status: str = "active",
    sort: str = "rating",
    limit: int = 500,
    offset: int = 0,
) -> list[dict[str, Any]]:
    return build_public_player_directory(
        supabase,
        club_id=club_id,
        search=search,
        status=status,
        sort=sort,
        limit=limit,
        offset=offset,
    )["players"]


def _directory_win_pct(player: dict[str, Any]) -> float | None:
    wins = int(player.get("wins") or 0)
    losses = int(player.get("losses") or 0)
    total = wins + losses
    return (wins / total) * 100.0 if total else None


def _sort_directory_players(rows: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    clean_sort = sort if sort in DIRECTORY_SORTS else "rating"
    if clean_sort == "name":
        key = lambda row: (str(row.get("name") or "").casefold(), str(row.get("id") or ""))
    elif clean_sort == "matches":
        key = lambda row: (-int(row.get("matches_played") or 0), str(row.get("name") or "").casefold())
    elif clean_sort == "singles":
        key = lambda row: (-float(row.get("singles_rating_jupr") if row.get("singles_rating_jupr") is not None else -1), str(row.get("name") or "").casefold())
    elif clean_sort == "win_pct":
        key = lambda row: (-float(_directory_win_pct(row) if _directory_win_pct(row) is not None else -1), -int(row.get("matches_played") or 0), str(row.get("name") or "").casefold())
    elif clean_sort == "recent":
        key = lambda row: (str(row.get("last_game_at") or ""), str(row.get("name") or "").casefold())
        return sorted(rows, key=key, reverse=True)
    else:
        key = lambda row: (-float(row.get("rating_jupr") if row.get("rating_jupr") is not None else -1), str(row.get("name") or "").casefold())
    return sorted(rows, key=key)


def build_public_player_directory(
    supabase: Any,
    *,
    club_id: str,
    search: str | None = None,
    status: str = "active",
    sort: str = "rating",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    cid = str(club_id).strip()
    all_rows = [_player_base(row) for row in _fetch_players(supabase, cid)]
    clean_status = str(status or "active").strip().casefold()
    if clean_status not in DIRECTORY_STATUSES:
        clean_status = "active"
    clean_sort = str(sort or "rating").strip().casefold()
    if clean_sort not in DIRECTORY_SORTS:
        clean_sort = "rating"
    query = _plain_text(search, limit=80) or ""
    query = query.casefold()
    rows = list(all_rows)
    if query:
        rows = [row for row in rows if query in str(row.get("name") or "").casefold()]
    if clean_status == "active":
        rows = [row for row in rows if row.get("is_active") is not False]
    elif clean_status == "inactive":
        rows = [row for row in rows if row.get("is_active") is False]
    rows = _sort_directory_players(rows, clean_sort)
    safe_limit = max(1, min(int(limit or 100), 1000))
    safe_offset = max(0, int(offset or 0))
    page = rows[safe_offset : safe_offset + safe_limit]
    return {
        "players": page,
        "filters": {"search": query, "status": clean_status, "sort": clean_sort},
        "summary": {
            "public_players": len(all_rows),
            "active_players": sum(1 for row in all_rows if row.get("is_active") is not False),
            "inactive_players": sum(1 for row in all_rows if row.get("is_active") is False),
            "filtered_players": len(rows),
        },
        "pagination": {
            "total": len(rows),
            "limit": safe_limit,
            "offset": safe_offset,
            "has_more": safe_offset + len(page) < len(rows),
        },
    }


def get_public_player_profile(
    supabase: Any,
    *,
    club_id: str,
    player_id: int | str,
    recent_match_limit: int = PUBLIC_PROFILE_RECENT_LIMIT,
    history_limit: int = PUBLIC_PROFILE_HISTORY_LIMIT,
) -> dict[str, Any] | None:
    cid = str(club_id).strip()
    row = _fetch_player(supabase, cid, player_id)
    if not row:
        return None
    player = _player_base(row)
    public_league_names = _public_league_names(supabase, cid)
    league_ratings = [
        _public_league_rating(row)
        for row in _fetch_league_ratings(supabase, cid, player_id)
        if str(row.get("league_name") or "").strip() in public_league_names
    ]
    league_ratings.sort(key=lambda r: str(r.get("league_name") or "").casefold())

    players = _fetch_players(supabase, cid)
    public_players = [_player_base(item) for item in players]
    name_by_id = {str(item.get("id")): str(item.get("name") or "Player") for item in public_players}
    matches = [m for m in _fetch_recent_matches(supabase, cid, limit=600) if _match_includes_player(m, player_id)]
    matches.sort(key=_match_sort_key, reverse=True)
    rating = _rating_projection(player, matches, player_id=player_id)
    points_by_match = {str(point.get("match_id")): point for point in rating["history"] if point.get("match_id") is not None}
    public_matches: list[dict[str, Any]] = []
    for match in matches:
        projected = _public_match(match, name_by_id)
        position = _player_position(match, player_id)
        point = points_by_match.get(str(match.get("id")), {})
        projected.update(
            {
                "player_result": _match_result(match, position[1]) if position else None,
                "player_rating_before_jupr": point.get("rating_before_jupr"),
                "player_rating_after_jupr": point.get("rating_after_jupr"),
                "player_rating_delta_jupr": point.get("rating_delta_jupr"),
            }
        )
        public_matches.append(projected)

    safe_recent_limit = max(1, min(int(recent_match_limit or PUBLIC_PROFILE_RECENT_LIMIT), 25))
    safe_history_limit = max(safe_recent_limit, min(int(history_limit or PUBLIC_PROFILE_HISTORY_LIMIT), 500))
    verified_updates = _verified_update_projection(supabase, club_id=cid, player_id=player_id)
    return {
        "player": player,
        "identity": {
            "display_name": player["display_name"],
            "public_name_policy": "public_display_name",
            "verification_status": verified_updates["status"],
        },
        "verified_updates": verified_updates,
        "rating_summary": rating["summary"],
        "rating_breakdowns": rating["formats"],
        "rating_history": rating["history"],
        "league_ratings": league_ratings,
        "awards": _public_awards(supabase, club_id=cid, player_id=player_id),
        "relationships": _relationship_rows(matches, player_id=player_id, name_by_id=name_by_id),
        "social": _social_projection(supabase, club_id=cid, player_id=player_id),
        "recent_matches": public_matches[:safe_recent_limit],
        "match_history": public_matches[:safe_history_limit],
        "history": {
            "total_matches": len(public_matches),
            "recent_limit": safe_recent_limit,
            "history_limit": safe_history_limit,
            "has_more": len(public_matches) > safe_history_limit,
        },
    }


def get_public_matches(
    supabase: Any,
    *,
    club_id: str,
    player_id: int | str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    cid = str(club_id).strip()
    players = _fetch_players(supabase, cid)
    name_by_id = {str(p.get("id")): (_plain_text(p.get("name"), limit=160) or "Player") for p in players}
    rows = _fetch_recent_matches(supabase, cid, limit=max(int(limit or 100), 300 if player_id is not None else int(limit or 100)))
    if player_id is not None:
        rows = [row for row in rows if _match_includes_player(row, player_id)]
    public_rows = [_public_match(row, name_by_id) for row in rows]
    return public_rows[: max(1, min(int(limit or 100), 500))]


def get_public_match_detail(
    supabase: Any,
    *,
    club_id: str,
    match_id: int | str,
) -> dict[str, Any] | None:
    cid = str(club_id).strip()
    row = _fetch_match(supabase, cid, match_id)
    if not row:
        return None
    players = _fetch_players(supabase, cid)
    name_by_id = {str(p.get("id")): (_plain_text(p.get("name"), limit=160) or "Player") for p in players}
    return _public_match(row, name_by_id, include_rating_snapshot=True)
