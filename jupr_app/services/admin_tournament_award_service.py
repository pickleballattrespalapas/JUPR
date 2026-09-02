from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.gamification.badges_repo import build_player_badge_rows
from jupr_app.domain.tournament_podium import award_tournament_trophies_from_podium, build_tournament_podium_candidates
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_game_service import _require_reviewed_row_versions
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_podium_service import _podium_payload
from jupr_app.services.admin_tournament_podium_review_service import (
    build_admin_tournament_podium_review_fingerprint,
    find_current_admin_tournament_podium_review,
)
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_AWARD_PODIUM = "AWARD PODIUM"


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


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
    except Exception as exc:
        raise RuntimeError("Could not verify the tournament draw; podium awards were refused.") from exc
    return rows[0] if rows else None


def _podium_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_podium")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load the draw podium; podium awards were refused.") from exc
    return rows


def _teams_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_teams")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load draw teams; podium awards were refused.") from exc


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_games")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load draw games; podium awards were refused.") from exc


def _podium_structure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "placement": int(row.get("placement") or 0),
                "team_id": str(row.get("team_id") or ""),
                "source": str(row.get("source") or "").upper(),
            }
            for row in rows
        ],
        key=lambda row: row["placement"],
    )


def _candidate_keys(candidates: list[Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "player_id": int(candidate.player_id),
                "badge_id": str(candidate.badge_id),
                "context_id": str(candidate.context_id or ""),
            }
            for candidate in candidates
        ],
        key=lambda row: (row["context_id"], row["badge_id"], row["player_id"]),
    )


def _award_podium_atomic(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    expected_team_versions: list[dict[str, str]],
    expected_podium: list[dict[str, Any]],
    expected_awards: list[dict[str, Any]],
    badge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        response = supabase.rpc(
            "admin_award_tournament_draw_podium_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_expected_draw_updated_at": str(expected_draw_updated_at),
                "p_expected_teams": list(expected_team_versions),
                "p_expected_podium": list(expected_podium),
                "p_expected_awards": list(expected_awards),
                "p_badges": list(badge_rows),
            },
        ).execute()
    except Exception as exc:
        if any(
            marker in str(exc)
            for marker in (
                "JUPR_TOURNAMENT_DRAW_STALE",
                "JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE",
                "JUPR_TOURNAMENT_PODIUM_SNAPSHOT_STALE",
                "JUPR_TOURNAMENT_AWARD_PLAN_STALE",
                "JUPR_TOURNAMENT_AWARD_ALREADY_EXISTS",
            )
        ):
            raise StaleTournamentAdminStateError(
                "The draw, teams, podium, or award set changed while trophies were being awarded. Reload Tournament Live."
            ) from exc
        raise RuntimeError("Atomic tournament podium awards failed; no badge set was committed.") from exc
    data = getattr(response, "data", None)
    payload = data if isinstance(data, dict) else data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    rows = payload.get("badges") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Atomic tournament podium awards returned no saved badge set.")
    return [dict(row) for row in rows if isinstance(row, dict)]


def award_admin_tournament_draw_podium(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    expected_team_versions: list[dict[str, Any]] | None = None,
    expected_source_game_versions: list[dict[str, Any]] | None = None,
    expected_podium_versions: list[dict[str, Any]] | None = None,
    expected_podium: list[dict[str, Any]] | None = None,
    expected_awards: list[dict[str, Any]] | None = None,
    source: str = "next_tournament_admin_award_podium",
    dry_run: bool = False,
    atomic: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_AWARD_PODIUM:
        raise ValueError(f"Type {CONFIRM_AWARD_PODIUM} to award draw podium trophies.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")
    reviewed_draw_version = str(expected_draw_updated_at or "").strip()
    if atomic and not reviewed_draw_version:
        raise StaleTournamentAdminStateError("A reviewed draw version is required for staging podium awards.")
    if reviewed_draw_version and str(draw.get("updated_at") or "") != reviewed_draw_version:
        raise StaleTournamentAdminStateError("This tournament draw changed after podium awards were reviewed.")
    teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    reviewed_team_versions = _require_reviewed_row_versions(
        teams,
        expected_team_versions,
        label="team set",
        atomic=atomic,
    )
    podium_rows = _podium_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not podium_rows:
        raise ValueError("Generate a draw-scoped podium before awarding trophies.")
    _require_reviewed_row_versions(
        podium_rows,
        expected_podium_versions,
        label="podium set",
        atomic=atomic,
    )
    games = _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    _require_reviewed_row_versions(
        games,
        expected_source_game_versions,
        label="source game set",
        atomic=atomic,
    )
    review_fingerprint = build_admin_tournament_podium_review_fingerprint(
        draw=draw,
        teams=teams,
        games=games,
        podium=podium_rows,
    )
    review = find_current_admin_tournament_podium_review(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
        review_fingerprint=review_fingerprint,
    )
    if not bool(review.get("current")):
        reason = str(
            (review.get("blockers") or ["The current podium has not been explicitly reviewed."])[0]
        )
        raise ValueError(f"Podium awards are blocked: {reason}")

    ctx = SimpleNamespace(supabase=supabase, club_id=str(club_id))
    candidates = build_tournament_podium_candidates(ctx, clean_tournament_id, str(tournament.get("name") or ""), draw_id=clean_draw_id)
    if not candidates:
        raise ValueError("No podium badge candidates could be built for this draw.")
    current_podium = _podium_structure(podium_rows)
    current_awards = _candidate_keys(candidates)
    if atomic and (current_podium != list(expected_podium or []) or current_awards != list(expected_awards or [])):
        raise StaleTournamentAdminStateError(
            "The podium or exact award recipient set changed after review. Reload Tournament Live."
        )
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_draw_podium_award_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "candidate_count": len(candidates),
            "awarded_count": 0,
            "badge_ids": sorted({str(candidate.badge_id) for candidate in candidates}),
            "warnings": [],
        }
    if atomic:
        badge_rows = build_player_badge_rows(str(club_id), candidates, awarded_by="engine")
        saved_badges = _award_podium_atomic(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            expected_draw_updated_at=reviewed_draw_version,
            expected_team_versions=reviewed_team_versions,
            expected_podium=current_podium,
            expected_awards=current_awards,
            badge_rows=badge_rows,
        )
        awarded_count = len(saved_badges)
    else:
        awarded = award_tournament_trophies_from_podium(
            ctx,
            clean_tournament_id,
            str(tournament.get("name") or ""),
            draw_id=clean_draw_id,
        )
        awarded_count = len(awarded)

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="award_tournament_draw_podium_badges_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "candidate_count": len(candidates)},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "podium": [_podium_payload(row) for row in podium_rows],
            "candidate_count": len(candidates),
            "awarded_count": awarded_count,
            "badge_ids": sorted({str(candidate.badge_id) for candidate in candidates}),
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
        "mode": "tournament_draw_podium_award",
        "draw_id": clean_draw_id,
        "candidate_count": len(candidates),
        "awarded_count": awarded_count,
        "warnings": warnings,
    }
