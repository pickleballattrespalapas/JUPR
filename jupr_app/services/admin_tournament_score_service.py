from __future__ import annotations

from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournaments import finalize_game, resolve_playoff_dependencies
from jupr_app.services.admin_tournament_game_service import _game_payload
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_SAVE_SCORE = "SAVE SCORE"


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _fetch_game(supabase: Any, *, tournament_id: str, game_id: str) -> dict[str, Any] | None:
    try:
        rows = _safe_rows(
            supabase.table("tournament_games")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("id", str(game_id))
            .limit(1)
            .execute()
        )
    except Exception:
        rows = []
    return rows[0] if rows else None


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str | None) -> list[dict[str, Any]]:
    try:
        query = supabase.table("tournament_games").select("*").eq("tournament_id", str(tournament_id))
        if draw_id:
            query = query.eq("draw_id", str(draw_id))
        return _safe_rows(query.execute())
    except Exception:
        return []


def _apply_playoff_dependency_updates(supabase: Any, *, tournament_id: str, draw_id: str | None, after_game: dict[str, Any]) -> list[dict[str, Any]]:
    games = _games_for_draw(supabase, tournament_id=str(tournament_id), draw_id=draw_id)
    games = [{**row, **after_game} if str(row.get("id")) == str(after_game.get("id")) else row for row in games]
    updates = resolve_playoff_dependencies(games)
    applied: list[dict[str, Any]] = []
    for update in updates:
        update_id = _clean_text(update.get("id"), limit=120)
        if not update_id:
            continue
        update_payload = {key: value for key, value in update.items() if key != "id"}
        if not update_payload:
            continue
        rows = _safe_rows(
            supabase.table("tournament_games")
            .update(update_payload)
            .eq("tournament_id", str(tournament_id))
            .eq("id", update_id)
            .execute()
        )
        applied.extend(rows or [{"id": update_id, **update_payload}])
    return [_game_payload(row) for row in applied]


def update_admin_tournament_game_score(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    game_id: str,
    score_a: Any,
    score_b: Any,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_score_game",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_SAVE_SCORE:
        raise ValueError(f"Type {CONFIRM_SAVE_SCORE} to save this tournament score.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_game_id = _clean_text(game_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    before = _fetch_game(supabase, tournament_id=clean_tournament_id, game_id=clean_game_id)
    if not before:
        raise ValueError("game not found for this tournament")

    stage = _clean_text(before.get("stage"), limit=80).upper()
    if stage not in {"ROUND_ROBIN", "PLAYOFF"}:
        raise ValueError("Next scoring currently supports round-robin and playoff tournament games only.")
    if stage == "ROUND_ROBIN":
        existing_games = _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=_clean_text(before.get("draw_id"), limit=120) or None)
        if any(_clean_text(row.get("stage"), limit=80).upper() == "PLAYOFF" for row in existing_games):
            raise ValueError("This draw already has playoff games. Remove/recreate playoffs before changing round-robin scores.")

    next_score_a = _safe_int(score_a)
    next_score_b = _safe_int(score_b)
    if next_score_a is None or next_score_b is None:
        raise ValueError("Both scores are required.")
    updated_fields = finalize_game({**before, "score_a": next_score_a, "score_b": next_score_b})
    updated_rows = _safe_rows(
        supabase.table("tournament_games")
        .update(updated_fields)
        .eq("tournament_id", clean_tournament_id)
        .eq("id", clean_game_id)
        .execute()
    )
    after = updated_rows[0] if updated_rows else {**before, **updated_fields}
    game = _game_payload(after)
    dependency_updates = _apply_playoff_dependency_updates(
        supabase,
        tournament_id=clean_tournament_id,
        draw_id=_clean_text(before.get("draw_id"), limit=120) or None,
        after_game=after,
    ) if stage == "PLAYOFF" else []

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="score_tournament_game_admin",
        entity_type="tournament_game",
        entity_id=clean_game_id,
        before_json={"game": _game_payload(before)},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "game": game,
            "dependency_updates": dependency_updates,
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
    return {"ok": True, "mode": "tournament_game_score", "game": game, "dependency_updates": dependency_updates, "warnings": warnings}
