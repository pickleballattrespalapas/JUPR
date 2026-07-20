from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournaments import finalize_game, resolve_playoff_dependencies
from jupr_app.services.admin_tournament_game_service import (
    _fetch_draw,
    _game_payload,
    _require_reviewed_draw_version,
    _require_reviewed_row_versions,
)
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_SAVE_SCORE = "SAVE SCORE"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    except Exception as exc:
        raise RuntimeError("Could not verify the tournament game; score save was refused.") from exc
    return rows[0] if rows else None


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str | None) -> list[dict[str, Any]]:
    try:
        query = supabase.table("tournament_games").select("*").eq("tournament_id", str(tournament_id))
        if draw_id:
            query = query.eq("draw_id", str(draw_id))
        return _safe_rows(query.execute())
    except Exception as exc:
        raise RuntimeError("Could not load draw games; score save was refused.") from exc


def _podium_for_draw(supabase: Any, *, tournament_id: str, draw_id: str | None) -> list[dict[str, Any]]:
    try:
        query = supabase.table("tournament_podium").select("id").eq("tournament_id", str(tournament_id))
        if draw_id:
            query = query.eq("draw_id", str(draw_id))
        return _safe_rows(query.execute())
    except Exception as exc:
        raise RuntimeError("Could not verify draw podium state; score save was refused.") from exc


def _published_game_ids(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    game_ids: list[str],
) -> set[str]:
    if not game_ids:
        return set()
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select("tournament_game_id")
            .eq("club_id", str(club_id))
            .eq("tournament_id", str(tournament_id))
            .in_("tournament_game_id", game_ids)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify official-match publication state; score save was refused.") from exc
    return {str(row.get("tournament_game_id")) for row in rows if row.get("tournament_game_id")}


def _preview_playoff_dependency_updates(
    games: list[dict[str, Any]],
    *,
    after_game: dict[str, Any],
) -> list[dict[str, Any]]:
    prospective = [
        {**row, **after_game} if str(row.get("id")) == str(after_game.get("id")) else dict(row)
        for row in games
    ]
    return [dict(update) for update in resolve_playoff_dependencies(prospective)]


def _rpc_payload(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    raise RuntimeError("Atomic tournament score RPC returned no result.")


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
    expected_updated_at: str | None = None,
    expected_draw_updated_at: str | None = None,
    expected_source_game_versions: list[dict[str, Any]] | None = None,
    source: str = "next_tournament_admin_score_game",
    dry_run: bool = False,
    atomic: bool = False,
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
    reviewed_version = str(expected_updated_at or "").strip()
    if atomic and not reviewed_version:
        raise StaleTournamentAdminStateError(
            "A reviewed game version is required for staging score saves. Reload the draw before submitting."
        )
    if reviewed_version and str(before.get("updated_at") or "") != reviewed_version:
        raise StaleTournamentAdminStateError(
            "This tournament game changed after it was loaded. Reload the draw before saving a score."
        )

    stage = _clean_text(before.get("stage"), limit=80).upper()
    if stage not in {"ROUND_ROBIN", "PLAYOFF"}:
        raise ValueError("Next scoring currently supports round-robin and playoff tournament games only.")
    draw_id = _clean_text(before.get("draw_id"), limit=120) or None
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=draw_id) if draw_id else None
    if atomic and not draw:
        raise StaleTournamentAdminStateError(
            "A draw-scoped reviewed game is required for staging score saves. Reload the Ops snapshot."
        )
    reviewed_draw_version = _require_reviewed_draw_version(
        draw or {},
        expected_draw_updated_at=expected_draw_updated_at,
        atomic=atomic,
    )
    existing_games = _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=draw_id)
    reviewed_source_game_versions = (
        _require_reviewed_row_versions(
            existing_games,
            expected_source_game_versions,
            label="source game set",
            atomic=atomic,
        )
        if expected_source_game_versions is not None
        else []
    )
    reviewed_source_versions_by_id = {
        str(row["id"]): str(row["updated_at"])
        for row in reviewed_source_game_versions
    }
    if stage == "ROUND_ROBIN":
        if any(_clean_text(row.get("stage"), limit=80).upper() == "PLAYOFF" for row in existing_games):
            raise ValueError("This draw already has playoff games. Remove/recreate playoffs before changing round-robin scores.")
    if _podium_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=draw_id):
        raise ValueError("This draw already has a podium. Use the documented recovery workflow before changing any score.")
    game_ids = [_clean_text(row.get("id"), limit=120) for row in existing_games if _clean_text(row.get("id"), limit=120)]
    if _published_game_ids(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
        game_ids=game_ids,
    ):
        raise ValueError(
            "This draw already has official Match Log rows. Correct published results through Match Log and Replay History."
        )

    next_score_a = _safe_int(score_a)
    next_score_b = _safe_int(score_b)
    if next_score_a is None or next_score_b is None:
        raise ValueError("Both scores are required.")
    if next_score_a < 0 or next_score_b < 0:
        raise ValueError("Tournament scores cannot be negative.")
    updated_fields = {
        **finalize_game({**before, "score_a": next_score_a, "score_b": next_score_b}),
        "updated_at": _now_iso(),
    }
    after_preview = {**before, **updated_fields}
    dependency_preview = (
        _preview_playoff_dependency_updates(existing_games, after_game=after_preview)
        if stage == "PLAYOFF"
        else []
    )
    existing_by_id = {str(row.get("id")): row for row in existing_games}
    dependency_cas_updates: list[dict[str, Any]] = []
    for update in dependency_preview:
        target = existing_by_id.get(str(update.get("id"))) or {}
        if any(target.get(field) not in (None, "") for field in ("score_a", "score_b", "winner_team_id", "finalized_at")):
            raise ValueError(
                "This score would invalidate an already-scored downstream playoff game. Recover downstream results before changing it."
            )
        expected_dependency_version = str(
            reviewed_source_versions_by_id.get(str(update.get("id")))
            or target.get("updated_at")
            or ""
        ).strip()
        if atomic and not expected_dependency_version:
            raise StaleTournamentAdminStateError(
                "A reviewed version is required for every downstream playoff game. Reload the Ops snapshot."
            )
        dependency_cas_updates.append({**update, "expected_updated_at": expected_dependency_version})
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_game_score_preview",
            "dry_run": True,
            "write_count": 0,
            "game": _game_payload(after_preview),
            "dependency_updates": [_game_payload({**existing_by_id.get(str(row.get("id")), {}), **row}) for row in dependency_preview],
            "warnings": [],
        }
    if atomic:
        try:
            response = supabase.rpc(
                "admin_score_tournament_game_cas",
                {
                    "p_club_id": str(club_id),
                    "p_tournament_id": clean_tournament_id,
                    "p_game_id": clean_game_id,
                    "p_expected_updated_at": reviewed_version,
                    "p_expected_draw_updated_at": reviewed_draw_version,
                    "p_game_patch": updated_fields,
                    "p_dependency_updates": dependency_cas_updates,
                },
            ).execute()
        except Exception as exc:
            if any(
                marker in str(exc)
                for marker in (
                    "JUPR_TOURNAMENT_GAME_STALE",
                    "JUPR_TOURNAMENT_DRAW_STALE",
                    "JUPR_TOURNAMENT_DEPENDENCY_STALE",
                    "JUPR_TOURNAMENT_DOWNSTREAM_SCORE_LOCK",
                    "JUPR_TOURNAMENT_SCORE_PODIUM_LOCK",
                    "JUPR_TOURNAMENT_SCORE_PUBLISHED_LOCK",
                )
            ):
                raise StaleTournamentAdminStateError(
                    "The draw or one of its dependent games changed while the score was being saved. Reload the Ops snapshot."
                ) from exc
            raise
        rpc_result = _rpc_payload(response)
        after = dict(rpc_result.get("game") or after_preview)
        game = _game_payload(after)
        dependency_updates = [
            _game_payload(row) for row in (rpc_result.get("dependency_updates") or []) if isinstance(row, dict)
        ]
    else:
        update_query = (
            supabase.table("tournament_games")
            .update(updated_fields)
            .eq("tournament_id", clean_tournament_id)
            .eq("id", clean_game_id)
        )
        if reviewed_version:
            update_query = update_query.eq("updated_at", reviewed_version)
        updated_rows = _safe_rows(update_query.execute())
        if reviewed_version and not updated_rows:
            raise StaleTournamentAdminStateError(
                "This tournament game changed while the score was being saved. Reload before retrying."
            )
        after = updated_rows[0] if updated_rows else after_preview
        game = _game_payload(after)
        dependency_updates = _apply_playoff_dependency_updates(
            supabase,
            tournament_id=clean_tournament_id,
            draw_id=draw_id,
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
