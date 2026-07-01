from __future__ import annotations

import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

_PODIUM_CONTEXT_RE = re.compile(r"^(?P<tournament_id>.+):podium:(?P<placement>\d+)$")
_FINAL_TOURNAMENT_STATUSES = {"complete", "completed", "archived"}


def _parse_value_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        cleaned = str(value).strip()
        if cleaned == "":
            return None
        return int(float(cleaned))
    except (TypeError, ValueError):
        return None


def _normalize_teammate_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if "," in raw:
            return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        return [raw]
    return []


def parse_tournament_podium_context(
    context_id: str | None,
    value_num: Any | None = None,
) -> tuple[str | None, int | None]:
    tournament_id = None
    placement = None
    if context_id:
        match = _PODIUM_CONTEXT_RE.match(str(context_id).strip())
        if match:
            tournament_id = match.group("tournament_id")
            placement = _coerce_int(match.group("placement"))
    if placement is None:
        placement = _coerce_int(value_num)
    return tournament_id, placement


def _join_names(names: list[str]) -> str | None:
    cleaned = [name.strip() for name in names if name.strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return " & ".join(cleaned)
    return ", ".join(cleaned)


def _status_allows_podium_fallback(status: Any) -> bool:
    # Legacy tournament rows may not carry a status, but a recorded podium is still
    # the canonical result. Active/draft statuses are intentionally excluded.
    cleaned = str(status or "").strip().lower()
    if not cleaned:
        return True
    return cleaned in _FINAL_TOURNAMENT_STATUSES


def _fetch_player_tournament_team_rows(supabase: Any, player_id: int) -> dict[str, dict[str, Any]]:
    teams_by_id: dict[str, dict[str, Any]] = {}
    for player_column in ("player1_id", "player2_id"):
        try:
            resp = (
                supabase.table("tournament_teams")
                .select("id,tournament_id,team_number,player1_id,player2_id")
                .eq(player_column, int(player_id))
                .execute()
            )
        except Exception:
            logger.exception(
                "Failed to fetch player tournament teams",
                extra={"player_id": player_id, "player_column": player_column},
            )
            continue
        for row in resp.data or []:
            team_id = str(row.get("id") or "").strip()
            if team_id and team_id not in teams_by_id:
                teams_by_id[team_id] = row
    return teams_by_id


def _fetch_podium_fallback_trophies(
    supabase: Any,
    club_id: str,
    player_id: int,
    existing_context_ids: set[str],
) -> list[dict[str, Any]]:
    teams_by_id = _fetch_player_tournament_team_rows(supabase, player_id)
    if not teams_by_id:
        return []

    tournament_ids = sorted(
        {
            str(row.get("tournament_id") or "").strip()
            for row in teams_by_id.values()
            if str(row.get("tournament_id") or "").strip()
        }
    )
    if not tournament_ids:
        return []

    try:
        tournaments_resp = (
            supabase.table("tournaments")
            .select("id,name,status")
            .eq("club_id", str(club_id))
            .in_("id", tournament_ids)
            .execute()
        )
    except Exception:
        logger.exception("Failed to fetch tournament metadata for podium fallback", extra={"player_id": player_id})
        return []

    tournaments_by_id = {
        str(row.get("id")): row
        for row in (tournaments_resp.data or [])
        if row.get("id") and _status_allows_podium_fallback(row.get("status"))
    }
    if not tournaments_by_id:
        return []

    eligible_team_ids = sorted(
        team_id
        for team_id, row in teams_by_id.items()
        if str(row.get("tournament_id") or "").strip() in tournaments_by_id
    )
    if not eligible_team_ids:
        return []

    try:
        podium_resp = (
            supabase.table("tournament_podium")
            .select("tournament_id,placement,team_id,source")
            .in_("team_id", eligible_team_ids)
            .execute()
        )
    except Exception:
        logger.exception("Failed to fetch tournament podium fallback rows", extra={"player_id": player_id})
        return []

    fallback: list[dict[str, Any]] = []
    seen_context_ids = set(existing_context_ids)
    for row in podium_resp.data or []:
        team_id = str(row.get("team_id") or "").strip()
        team = teams_by_id.get(team_id)
        if not team:
            continue
        tournament_id = str(row.get("tournament_id") or team.get("tournament_id") or "").strip()
        tournament = tournaments_by_id.get(tournament_id)
        if not tournament:
            continue
        placement = _coerce_int(row.get("placement"))
        if placement is None:
            continue
        context_id = f"{tournament_id}:podium:{placement}"
        if context_id in seen_context_ids:
            continue
        seen_context_ids.add(context_id)
        fallback.append(
            {
                "placement": placement,
                "tournament_id": tournament_id,
                "tournament_name": str(tournament.get("name") or "").strip() or None,
                "teammate_names": [],
                "earned_at": None,
                "team_id": team_id,
                "context_id": context_id,
            }
        )

    fallback.sort(
        key=lambda item: (
            str(item.get("tournament_name") or ""),
            int(item.get("placement") or 999),
        )
    )
    return fallback


def get_player_tournament_trophies(
    supabase: Any,
    club_id: str,
    player_id: int,
) -> list[dict[str, Any]]:
    if supabase is None or not club_id:
        return []

    try:
        resp = (
            supabase.table("player_badges")
            .select("player_id,earned_at,context_id,value_num,value_json")
            .eq("club_id", str(club_id))
            .eq("player_id", int(player_id))
            .eq("context_type", "tournament")
            .like("context_id", "%:podium:%")
            .order("earned_at", desc=True)
            .execute()
        )
        rows = resp.data or []
    except Exception:
        logger.exception("Failed to fetch tournament podium trophies", extra={"player_id": player_id})
        rows = []

    existing_context_ids = {
        str(row.get("context_id") or "").strip()
        for row in rows
        if str(row.get("context_id") or "").strip()
    }
    podium_fallbacks = _fetch_podium_fallback_trophies(
        supabase,
        str(club_id),
        int(player_id),
        existing_context_ids,
    )
    if not rows and not podium_fallbacks:
        return []

    normalized: list[dict[str, Any]] = []
    tournament_ids: set[str] = set()
    team_ids: set[str] = set()

    for row in rows:
        value_json = _parse_value_json(row.get("value_json"))
        placement = _coerce_int(value_json.get("placement"))
        context_id = str(row.get("context_id") or "")
        tournament_id, parsed_placement = parse_tournament_podium_context(
            context_id,
            row.get("value_num"),
        )
        if placement is None:
            placement = parsed_placement
        tournament_id = str(value_json.get("tournament_id") or tournament_id or "").strip() or None
        tournament_name = str(value_json.get("tournament_name") or "").strip() or None
        teammate_names = _normalize_teammate_names(value_json.get("teammate_names"))
        team_id = value_json.get("team_id")
        if tournament_id:
            tournament_ids.add(tournament_id)
        if team_id and not teammate_names:
            team_ids.add(str(team_id))

        normalized.append(
            {
                "placement": placement,
                "tournament_id": tournament_id,
                "tournament_name": tournament_name,
                "teammate_names": teammate_names,
                "earned_at": row.get("earned_at"),
                "team_id": str(team_id) if team_id else None,
            }
        )

    for item in podium_fallbacks:
        if item.get("tournament_id"):
            tournament_ids.add(str(item["tournament_id"]))
        if item.get("team_id"):
            team_ids.add(str(item["team_id"]))
        normalized.append(item)

    tournament_name_map: dict[str, str] = {}
    if tournament_ids:
        try:
            t_resp = (
                supabase.table("tournaments")
                .select("id,name")
                .in_("id", list(tournament_ids))
                .execute()
            )
            tournament_name_map = {str(row.get("id")): str(row.get("name")) for row in (t_resp.data or [])}
        except Exception:
            logger.exception("Failed to fetch tournament names", extra={"player_id": player_id})

    team_player_map: dict[str, list[int]] = {}
    if team_ids:
        try:
            teams_resp = (
                supabase.table("tournament_teams")
                .select("id,player1_id,player2_id")
                .in_("id", list(team_ids))
                .execute()
            )
            for team in teams_resp.data or []:
                team_id = str(team.get("id"))
                players = [team.get("player1_id"), team.get("player2_id")]
                team_player_map[team_id] = [int(pid) for pid in players if pid]
        except Exception:
            logger.exception("Failed to fetch tournament teams", extra={"player_id": player_id})

    player_ids: set[int] = set()
    for ids in team_player_map.values():
        player_ids.update(ids)

    player_name_map: dict[int, str] = {}
    if player_ids:
        try:
            players_resp = (
                supabase.table("players")
                .select("id,name")
                .eq("club_id", str(club_id))
                .in_("id", list(player_ids))
                .execute()
            )
            player_name_map = {int(row.get("id")): str(row.get("name")) for row in (players_resp.data or [])}
        except Exception:
            logger.exception("Failed to fetch tournament player names", extra={"player_id": player_id})

    for item in normalized:
        if not item.get("tournament_name") and item.get("tournament_id"):
            item["tournament_name"] = tournament_name_map.get(str(item["tournament_id"]))
        if not item.get("teammate_names"):
            team_id = item.get("team_id")
            team_players = team_player_map.get(str(team_id), []) if team_id else []
            teammate_list = [
                player_name_map.get(pid, "")
                for pid in team_players
                if int(pid) != int(player_id)
            ]
            item["teammate_names"] = _normalize_teammate_names(teammate_list)
        item["teammate_names"] = _join_names(item.get("teammate_names") or [])

    return normalized
