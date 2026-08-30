from __future__ import annotations

from datetime import date, datetime
from typing import Any

from jupr_app.domain.tournament_registration_repo import (
    get_public_tournament_bundle,
    list_public_tournaments,
)
from jupr_app.domain.tournament_public_references import (
    build_public_tournament_reference,
)
from jupr_app.domain.tournaments import compute_round_robin_standings


MEDAL_LABELS = {1: "Gold", 2: "Silver", 3: "Bronze"}
NON_PLAYED_OUTCOME_LABELS = {
    "FORFEIT": "Forfeit",
    "WALKOVER": "Walkover",
    "NO_SHOW": "No-show",
    "NOSHOW": "No-show",
    "RETIREMENT": "Retirement",
    "RETIRED": "Retirement",
    "WITHDRAWAL": "Withdrawal",
    "WITHDRAWN": "Withdrawal",
    "BYE": "Bye",
}


def _rows(
    supabase: Any,
    table_name: str,
    *,
    filters: tuple[tuple[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    query = supabase.table(table_name).select("*")
    for key, value in filters:
        query = query.eq(key, value)
    try:
        return [dict(row) for row in (query.execute().data or [])]
    except Exception as exc:
        raise RuntimeError(f"Public tournament results could not load {table_name}.") from exc


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _published_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "registration_slug": _text(settings.get("registration_slug"), limit=120),
        "registration_status": _text(
            settings.get("registration_status") or "closed", limit=40
        ).lower(),
    }


def _public_tournament(
    tournament: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": _text(tournament.get("id"), limit=120),
        "name": _text(tournament.get("name") or "Tournament"),
        "status": _text(tournament.get("status"), limit=40).upper(),
        "start_date": _json_safe(tournament.get("start_date")),
        "end_date": _json_safe(tournament.get("end_date")),
        "settings": _published_settings(settings),
    }


def build_public_tournament_index(
    supabase: Any,
    *,
    club_id: str,
    view: str = "current",
) -> dict[str, Any]:
    tournaments = [
        _public_tournament(
            dict(row.get("tournament") or {}),
            dict(row.get("settings") or {}),
        )
        for row in list_public_tournaments(
            supabase,
            str(club_id),
            view=view,
        )
    ]
    return {"view": view, "tournaments": tournaments}


def _player_name(row: dict[str, Any] | None) -> str:
    row = row or {}
    return _text(
        row.get("display_name")
        or row.get("name")
        or " ".join(
            part
            for part in (
                _text(row.get("first_name"), limit=80),
                _text(row.get("last_name"), limit=80),
            )
            if part
        )
        or "Player",
        limit=160,
    )


def _team_name(team: dict[str, Any], players: dict[str, dict[str, Any]]) -> str:
    names = [
        _player_name(players.get(str(player_id)))
        for player_id in (team.get("player1_id"), team.get("player2_id"))
        if player_id not in (None, "")
    ]
    return " / ".join(names) or f"Team {_integer(team.get('team_number')) or '?'}"


def _game_public_state(game: dict[str, Any]) -> str:
    if game.get("finalized_at") and (
        game.get("winner_team_id") or _non_played_outcome_label(game)
    ):
        return "FINAL"
    if game.get("team_a_id") and game.get("team_b_id"):
        return "READY"
    return "PENDING"


def _non_played_outcome_label(game: dict[str, Any]) -> str | None:
    raw = _text(
        game.get("outcome_type")
        or game.get("completion_reason")
        or game.get("result_type"),
        limit=40,
    ).upper().replace("-", "_").replace(" ", "_")
    return NON_PLAYED_OUTCOME_LABELS.get(raw)


def _draw_public_state(games: list[dict[str, Any]], podium: list[dict[str, Any]]) -> str:
    if not games:
        return "SCHEDULED"
    finalized = [row for row in games if _game_public_state(row) == "FINAL"]
    if len(finalized) == len(games) and podium:
        return "COMPLETE"
    if finalized:
        return "LIVE"
    return "READY"


def build_public_tournament_results(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> dict[str, Any]:
    tournament, settings, days, events = get_public_tournament_bundle(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    if not tournament or not settings:
        raise ValueError("tournament results not found")

    event_by_id = {str(row.get("id") or ""): row for row in events}
    day_by_id = {str(row.get("id") or ""): row for row in days}
    draws = [
        row
        for row in _rows(
            supabase,
            "tournament_event_draws",
            filters=(("tournament_id", str(tournament_id)),),
        )
        if str(row.get("event_option_id") or "") in event_by_id
        and str(row.get("draw_kind") or "").upper() != "TEAM_PARENT"
        and str(
            event_by_id.get(str(row.get("event_option_id") or ""), {}).get(
                "competition_format"
            )
            or "STANDARD"
        ).upper()
        != "FOUR_PLAYER_TEAM"
        and str(row.get("status") or "").strip().upper()
        not in {"CANCELLED", "CANCELED", "ARCHIVED", "DISABLED"}
    ]
    teams = _rows(
        supabase,
        "tournament_teams",
        filters=(("tournament_id", str(tournament_id)),),
    )
    games = _rows(
        supabase,
        "tournament_games",
        filters=(("tournament_id", str(tournament_id)),),
    )
    podium_rows = _rows(
        supabase,
        "tournament_podium",
        filters=(("tournament_id", str(tournament_id)),),
    )

    player_ids = {
        str(player_id)
        for team in teams
        for player_id in (team.get("player1_id"), team.get("player2_id"))
        if player_id not in (None, "")
    }
    try:
        player_rows = [
            dict(row)
            for row in (
                supabase.table("players")
                .select("*")
                .eq("club_id", str(club_id))
                .in_("id", [int(value) if value.isdigit() else value for value in player_ids])
                .execute()
                .data
                or []
            )
        ] if player_ids else []
    except Exception as exc:
        raise RuntimeError("Public tournament results could not resolve player names.") from exc
    players = {str(row.get("id") or ""): row for row in player_rows}

    output_draws: list[dict[str, Any]] = []
    for draw in draws:
        draw_id = str(draw.get("id") or "")
        draw_teams = sorted(
            [row for row in teams if str(row.get("draw_id") or "") == draw_id],
            key=lambda row: (_integer(row.get("team_number")) or 0, str(row.get("id") or "")),
        )
        draw_games = [row for row in games if str(row.get("draw_id") or "") == draw_id]
        draw_podium = [
            row for row in podium_rows if str(row.get("draw_id") or "") == draw_id
        ]
        team_names = {
            str(row.get("id") or ""): _team_name(row, players) for row in draw_teams
        }
        rr_games = [
            row
            for row in draw_games
            if str(row.get("stage") or "").upper() == "ROUND_ROBIN"
        ]
        standings = [
            {
                "public_team_key": build_public_tournament_reference(
                    tournament_id=str(tournament_id),
                    namespace="standard-team",
                    source_id=str(row.get("team_id") or ""),
                ),
                "rank": row.get("seed"),
                "team_name": team_names.get(str(row.get("team_id") or ""), "Team"),
                "wins": row.get("wins"),
                "losses": row.get("losses"),
                "points_for": row.get("points_for"),
                "points_against": row.get("points_against"),
                "differential": row.get("differential"),
                "competition_status": row.get("competition_status"),
                "retired": bool(row.get("retired")),
            }
            for row in compute_round_robin_standings(draw_teams, rr_games)
        ]
        public_games = [
            {
                "public_game_key": build_public_tournament_reference(
                    tournament_id=str(tournament_id),
                    namespace="standard-game",
                    source_id=str(row.get("id") or ""),
                ),
                "stage": _text(row.get("stage"), limit=40).upper(),
                "round_number": _integer(row.get("rr_round_number")),
                "slot_number": _integer(row.get("rr_slot_number")),
                "playoff_game_code": _text(row.get("playoff_game_code"), limit=40)
                or None,
                "playoff_round": _text(row.get("playoff_round"), limit=40)
                or None,
                "team_a_name": team_names.get(str(row.get("team_a_id") or ""), "TBD"),
                "team_b_name": team_names.get(str(row.get("team_b_id") or ""), "TBD"),
                "score_a": _integer(row.get("score_a")),
                "score_b": _integer(row.get("score_b")),
                "winner_name": team_names.get(
                    str(row.get("winner_team_id") or ""), ""
                )
                or None,
                "outcome_label": _non_played_outcome_label(row),
                "state": _game_public_state(row),
                "finalized_at": _json_safe(row.get("finalized_at")),
            }
            for row in sorted(
                draw_games,
                key=lambda row: (
                    0 if str(row.get("stage") or "").upper() == "ROUND_ROBIN" else 1,
                    _integer(row.get("rr_round_number")) or 0,
                    _integer(row.get("rr_slot_number")) or 0,
                    str(row.get("playoff_game_code") or ""),
                ),
            )
        ]
        event = event_by_id.get(str(draw.get("event_option_id") or "")) or {}
        registration_day_id = str(draw.get("registration_day_id") or "")
        scheduled_day_ids = [
            str(value)
            for value in (event.get("scheduled_day_ids") or [registration_day_id])
            if str(value)
        ]
        output_draws.append(
            {
                "public_draw_key": build_public_tournament_reference(
                    tournament_id=str(tournament_id),
                    namespace="standard-draw",
                    source_id=draw_id,
                ),
                "name": _text(draw.get("name") or "Tournament draw"),
                "state": _draw_public_state(draw_games, draw_podium),
                "event_family_label": _text(event.get("event_family_label") or "Event"),
                "division_name": _text(event.get("division_name") or event.get("label") or "Division"),
                "event_type": _text(event.get("event_type"), limit=40),
                "scheduled_days": [
                    {
                        "label": _text(day_by_id.get(day_id, {}).get("label") or "Day"),
                        "event_date": _json_safe(day_by_id.get(day_id, {}).get("event_date")),
                    }
                    for day_id in scheduled_day_ids
                    if day_id in day_by_id
                ],
                "teams": [
                    {
                        "public_team_key": build_public_tournament_reference(
                            tournament_id=str(tournament_id),
                            namespace="standard-team",
                            source_id=str(row.get("id") or ""),
                        ),
                        "team_number": _integer(row.get("team_number")),
                        "seed": _integer(row.get("seed")),
                        "name": team_names.get(str(row.get("id") or ""), "Team"),
                        "competition_status": _text(
                            row.get("competition_status") or "ACTIVE",
                            limit=40,
                        ).upper(),
                    }
                    for row in draw_teams
                ],
                "standings": standings,
                # Keep the completed-score feed and bracket mutually exclusive:
                # playoff rows belong in the bracket (including upcoming games),
                # while this feed is a history of finalized non-playoff results.
                "scores": [
                    row
                    for row in public_games
                    if row.get("stage") != "PLAYOFF" and row.get("state") == "FINAL"
                ],
                "bracket": [
                    row for row in public_games if row.get("stage") == "PLAYOFF"
                ],
                "podium": [
                    {
                        "placement": _integer(row.get("placement")),
                        "medal": MEDAL_LABELS.get(_integer(row.get("placement")) or 0),
                        "team_name": team_names.get(str(row.get("team_id") or ""), "Team"),
                    }
                    for row in sorted(
                        draw_podium,
                        key=lambda row: _integer(row.get("placement")) or 99,
                    )
                ],
            }
        )

    return {
        "tournament": _public_tournament(tournament, settings),
        "draws": sorted(
            output_draws,
            key=lambda row: (
                str((row.get("scheduled_days") or [{}])[0].get("event_date") or ""),
                str(row.get("event_family_label") or ""),
                str(row.get("division_name") or ""),
            ),
        ),
    }
