from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os

import pandas as pd

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.match_processing import process_matches
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    is_admin_tournament_admin_enabled,
)

CONFIRM_PUBLISH_MATCHES = "PUBLISH MATCHES"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _fetch_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> dict[str, Any] | None:
    try:
        rows = _safe_rows(
            supabase.table("tournament_event_draws")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("id", str(draw_id))
            .limit(1)
            .execute()
        )
    except Exception:
        rows = []
    return rows[0] if rows else None


def _fetch_event_option(supabase: Any, *, tournament_id: str, event_option_id: str | None) -> dict[str, Any] | None:
    clean_event_option_id = _clean_text(event_option_id, limit=120)
    if not clean_event_option_id:
        return None
    try:
        rows = _safe_rows(
            supabase.table("tournament_event_options")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("id", clean_event_option_id)
            .limit(1)
            .execute()
        )
    except Exception:
        rows = []
    return rows[0] if rows else None


def _teams_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_teams")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception:
        rows = []
    return sorted(rows, key=lambda row: int(_safe_int(row.get("team_number")) or 0))


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_games")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception:
        rows = []
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("stage") or ""),
            int(_safe_int(row.get("rr_round_number")) or 0),
            int(_safe_int(row.get("rr_slot_number")) or 0),
            str(row.get("playoff_game_code") or ""),
            str(row.get("id") or ""),
        ),
    )


def _existing_published_game_ids(supabase: Any, *, club_id: str, tournament_id: str, game_ids: list[str]) -> set[str]:
    if not game_ids:
        return set()
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select("id,tournament_game_id")
            .eq("club_id", str(club_id))
            .eq("tournament_id", str(tournament_id))
            .in_("tournament_game_id", game_ids)
            .execute()
        )
    except Exception:
        rows = []
    return {str(row.get("tournament_game_id")) for row in rows if row.get("tournament_game_id")}


def _table_frame(supabase: Any, table_name: str, *, club_id: str | None = None) -> pd.DataFrame:
    try:
        query = supabase.table(table_name).select("*")
        if club_id and table_name in {"players", "league_ratings", "leagues_metadata"}:
            query = query.eq("club_id", str(club_id))
        rows = _safe_rows(query.execute())
    except Exception:
        rows = []
    return pd.DataFrame(rows)


def _division_label(event_option: dict[str, Any] | None, draw: dict[str, Any]) -> str:
    event_option = event_option or {}
    family = _clean_text(event_option.get("event_family_label") or event_option.get("label"), limit=120)
    division = _clean_text(event_option.get("division_name") or event_option.get("label"), limit=120)
    if family and division and family != division:
        return f"{family} / {division}"
    return division or family or _clean_text(draw.get("name"), limit=160) or "Tournament Draw"


def _published_date(tournament: dict[str, Any], draw: dict[str, Any], game: dict[str, Any]) -> str:
    for key_source in (
        game.get("finalized_at"),
        tournament.get("start_date"),
        tournament.get("end_date"),
        draw.get("created_at"),
        game.get("created_at"),
    ):
        value = _clean_text(key_source, limit=80)
        if value:
            return value
    return _now_iso()


def _validate_scored_game(game: dict[str, Any], *, game_index: int) -> tuple[int, int]:
    score_a = _safe_int(game.get("score_a"))
    score_b = _safe_int(game.get("score_b"))
    if score_a is None or score_b is None:
        raise ValueError(f"Game {game_index} is missing a score.")
    if score_a == score_b:
        raise ValueError(f"Game {game_index} has a tied score; official matches do not support ties.")
    if not _clean_text(game.get("winner_team_id"), limit=120):
        raise ValueError(f"Game {game_index} is not finalized with a winner.")
    if not _clean_text(game.get("team_a_id"), limit=120) or not _clean_text(game.get("team_b_id"), limit=120):
        raise ValueError(f"Game {game_index} is missing team assignments.")
    return int(score_a), int(score_b)


def _build_official_match_payloads(
    *,
    tournament: dict[str, Any],
    draw: dict[str, Any],
    event_option: dict[str, Any] | None,
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    teams_by_id = {str(row.get("id")): row for row in teams if row.get("id")}
    tournament_name = _clean_text(tournament.get("name"), limit=160) or "Tournament"
    division_label = _division_label(event_option, draw)
    league_name = f"Tournament · {tournament_name} · {division_label}"
    week_tag = _clean_text(draw.get("name"), limit=120) or division_label

    payloads: list[dict[str, Any]] = []
    for index, game in enumerate(games, start=1):
        score_a, score_b = _validate_scored_game(game, game_index=index)
        team_a = teams_by_id.get(str(game.get("team_a_id") or ""))
        team_b = teams_by_id.get(str(game.get("team_b_id") or ""))
        if not team_a or not team_b:
            raise ValueError(f"Game {index} references a team that is not in this draw.")
        a1, a2 = _safe_int(team_a.get("player1_id")), _safe_int(team_a.get("player2_id"))
        b1, b2 = _safe_int(team_b.get("player1_id")), _safe_int(team_b.get("player2_id"))
        if a1 is None or a2 is None or b1 is None or b2 is None:
            raise ValueError("Official match publishing currently requires doubles teams with two linked JUPR players per team.")
        payloads.append(
            {
                "date": _published_date(tournament, draw, game),
                "league": league_name,
                "week_tag": week_tag,
                "match_type": "Tournament",
                "t1_p1": a1,
                "t1_p2": a2,
                "t2_p1": b1,
                "t2_p2": b2,
                "score_t1": score_a,
                "score_t2": score_b,
                "context_type": "tournament_game",
                "context_id": _clean_text(game.get("id"), limit=120),
                "tournament_id": _clean_text(tournament.get("id"), limit=120),
                "tournament_game_id": _clean_text(game.get("id"), limit=120),
                "rating_scope": "",
            }
        )
    return payloads


def publish_admin_tournament_draw_matches(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_publish_matches",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_PUBLISH_MATCHES:
        raise ValueError(f"Type {CONFIRM_PUBLISH_MATCHES} to publish official tournament matches.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")

    teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    games = _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not games:
        raise ValueError("This draw has no tournament games to publish.")

    game_ids = [_clean_text(row.get("id"), limit=120) for row in games if _clean_text(row.get("id"), limit=120)]
    already_published = _existing_published_game_ids(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
        game_ids=game_ids,
    )
    if already_published:
        raise ValueError("Some tournament games are already published as official matches: " + ", ".join(sorted(already_published)))

    event_option = _fetch_event_option(
        supabase,
        tournament_id=clean_tournament_id,
        event_option_id=_clean_text(draw.get("event_option_id"), limit=120),
    )
    match_payloads = _build_official_match_payloads(
        tournament=tournament,
        draw=draw,
        event_option=event_option,
        teams=teams,
        games=games,
    )

    df_players_all = _table_frame(supabase, "players", club_id=str(club_id))
    df_leagues = _table_frame(supabase, "league_ratings", club_id=str(club_id))
    df_meta = _table_frame(supabase, "leagues_metadata", club_id=str(club_id))
    process_result = process_matches(
        match_payloads,
        supabase=supabase,
        club_id=str(club_id),
        name_to_id={},
        df_players_all=df_players_all,
        df_leagues=df_leagues,
        df_meta=df_meta,
    )
    inserted_count = int(process_result.get("inserted") or 0)
    if inserted_count != len(match_payloads):
        raise RuntimeError(f"Official match publish inserted {inserted_count} of {len(match_payloads)} tournament games.")

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="publish_tournament_games_to_matches_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "game_count": len(games)},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "match_count": inserted_count,
            "tournament_game_ids": game_ids,
            "process_result": process_result,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")

    return {
        "ok": True,
        "mode": "tournament_official_matches_publish",
        "draw_id": clean_draw_id,
        "match_count": inserted_count,
        "game_count": len(games),
        "tournament_game_ids": game_ids,
        "process_result": process_result,
        "warnings": warnings,
    }
