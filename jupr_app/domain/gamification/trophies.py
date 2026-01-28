from __future__ import annotations

import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

_PODIUM_CONTEXT_RE = re.compile(r"^(?P<tournament_id>.+):podium:(?P<placement>\d+)$")


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
    except Exception:
        logger.exception("Failed to fetch tournament podium trophies", extra={"player_id": player_id})
        return []

    rows = resp.data or []
    if not rows:
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
