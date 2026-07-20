from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_game_service import _require_reviewed_draw_version
from jupr_app.services.admin_tournament_team_service import _team_payload, write_admin_tournament_draw_teams_atomic
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_IMPORT_REGISTRATIONS = "IMPORT REGISTRATIONS"


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


def _fetch_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> dict[str, Any] | None:
    try:
        rows = _safe_rows(supabase.table("tournament_event_draws").select("*").eq("tournament_id", str(tournament_id)).eq("id", str(draw_id)).limit(1).execute())
    except Exception as exc:
        raise RuntimeError("Could not verify the tournament draw; registration import was refused.") from exc
    return rows[0] if rows else None


def _registrations(supabase: Any, *, tournament_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table("tournament_registrations").select("*").eq("tournament_id", str(tournament_id)).execute())
    except Exception as exc:
        raise RuntimeError("Could not load tournament registrations; registration import was refused.") from exc


def _selections_for_draw(supabase: Any, *, tournament_id: str, draw: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(supabase.table("tournament_registration_selections").select("*").eq("tournament_id", str(tournament_id)).execute())
    except Exception as exc:
        raise RuntimeError("Could not load tournament registration selections; registration import was refused.") from exc
    event_option_id = _clean_text(draw.get("event_option_id"), limit=120)
    day_id = _clean_text(draw.get("registration_day_id"), limit=120)
    if event_option_id:
        rows = [row for row in rows if _clean_text(row.get("event_option_id"), limit=120) == event_option_id]
    if day_id:
        rows = [row for row in rows if _clean_text(row.get("registration_day_id"), limit=120) == day_id]
    return rows


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table("tournament_games").select("id").eq("tournament_id", str(tournament_id)).eq("draw_id", str(draw_id)).limit(1).execute())
    except Exception as exc:
        raise RuntimeError("Could not verify whether this draw already has games; registration import was refused.") from exc


def _teams_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table("tournament_teams").select("*").eq("tournament_id", str(tournament_id)).eq("draw_id", str(draw_id)).execute())
    except Exception as exc:
        raise RuntimeError("Could not load current draw teams; registration import was refused.") from exc


def _email(value: Any) -> str:
    return str(value or "").strip().lower()


def import_admin_tournament_registrations_to_draw(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    import_mode: str = "REPLACE",
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    source: str = "next_tournament_admin_import_registrations",
    dry_run: bool = False,
    atomic: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_IMPORT_REGISTRATIONS:
        raise ValueError(f"Type {CONFIRM_IMPORT_REGISTRATIONS} to import confirmed registrations into this draw.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")
    reviewed_draw_version = _require_reviewed_draw_version(
        draw,
        expected_draw_updated_at=expected_draw_updated_at,
        atomic=atomic,
    )
    if _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id):
        raise ValueError("This draw already has games. Registration import is blocked after scheduling begins.")

    mode = _clean_text(import_mode or "REPLACE", limit=20).upper()
    if mode not in {"REPLACE", "APPEND"}:
        raise ValueError("import_mode must be REPLACE or APPEND")

    registrations = _registrations(supabase, tournament_id=clean_tournament_id)
    registrations_by_id = {_clean_text(row.get("id"), limit=120): row for row in registrations}
    registrations_by_email = {_email(row.get("email")): row for row in registrations if _email(row.get("email"))}
    selections = _selections_for_draw(supabase, tournament_id=clean_tournament_id, draw=draw)

    current_teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    start_slot = max([_safe_int(row.get("team_number")) or 0 for row in current_teams], default=0) + 1 if mode == "APPEND" else 1
    selected_player_ids: list[int] = []
    unresolved: list[str] = []
    rows: list[dict[str, Any]] = []
    now = _now_iso()

    for selection in selections:
        registration = registrations_by_id.get(_clean_text(selection.get("registration_id"), limit=120))
        if not registration:
            continue
        if _clean_text(registration.get("status") or registration.get("registration_status") or "confirmed", limit=40).lower() != "confirmed":
            continue
        player1_id = _safe_int(registration.get("player_id"))
        if player1_id is None:
            unresolved.append(_clean_text(registration.get("display_name") or registration.get("email") or registration.get("id"), limit=180))
            continue
        player2_id = None
        partner_email = _email(selection.get("partner_email"))
        if partner_email:
            partner = registrations_by_email.get(partner_email)
            player2_id = _safe_int((partner or {}).get("player_id"))
            if player2_id is None:
                unresolved.append(_clean_text(selection.get("partner_name") or partner_email, limit=180))
                continue
        selected_player_ids.append(player1_id)
        if player2_id is not None:
            selected_player_ids.append(player2_id)
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "tournament_id": clean_tournament_id,
                "draw_id": clean_draw_id,
                "registration_day_id": _clean_text(draw.get("registration_day_id"), limit=120) or None,
                "event_option_id": _clean_text(draw.get("event_option_id"), limit=120) or None,
                "team_number": start_slot + len(rows),
                "player1_id": player1_id,
                "player2_id": player2_id,
                "source": "REGISTRATION",
                "notes": f"Imported from registration {_clean_text(registration.get('id'), limit=120)}",
                "created_at": now,
            }
        )

    duplicates = sorted({pid for pid in selected_player_ids if selected_player_ids.count(pid) > 1})
    if duplicates:
        raise ValueError("Duplicate player IDs in confirmed registration import: " + ", ".join(str(pid) for pid in duplicates))
    if unresolved:
        raise ValueError("Some confirmed registrations could not be resolved to JUPR players: " + ", ".join(sorted(set(filter(None, unresolved)))))
    if not rows:
        raise ValueError("No confirmed registrations with linked player IDs were available for this draw.")

    before = [_team_payload(row) for row in current_teams]
    if dry_run:
        teams = [_team_payload(row) for row in rows]
        return {
            "ok": True,
            "mode": "tournament_registration_team_import_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "import_mode": mode,
            "updated_count": len(teams),
            "teams": teams,
            "warnings": [],
        }
    if atomic:
        inserted = write_admin_tournament_draw_teams_atomic(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            expected_draw_updated_at=reviewed_draw_version,
            rows=rows,
            replace=mode == "REPLACE",
        )
    else:
        if mode == "REPLACE":
            supabase.table("tournament_teams").delete().eq("tournament_id", clean_tournament_id).eq("draw_id", clean_draw_id).execute()
        inserted = _safe_rows(supabase.table("tournament_teams").insert(rows).execute())
    teams = [_team_payload(row) for row in (inserted or rows)]

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="import_tournament_registration_teams_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "teams": before, "mode": mode},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "draw": _draw_payload(draw), "mode": mode, "teams": teams},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_registration_team_import", "draw_id": clean_draw_id, "import_mode": mode, "updated_count": len(teams), "teams": teams, "warnings": warnings}
