from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tournament_service import (
    EVENT_OPTION_SELECT,
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    _table_rows_for_tournament,
    is_admin_tournament_admin_enabled,
)
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
)

CONFIRM_CREATE_DRAW = "CREATE DRAW"
CONFIRM_CANCEL_EMPTY_DRAW = "CANCEL EMPTY DRAW"
CONFIRM_CANCEL_EMPTY_EVENT = "CANCEL EMPTY EVENT"


def _require_atomic_recovery(
    *, atomic: bool, allow_non_atomic_test_adapter: bool
) -> None:
    if atomic:
        return
    if (
        os.getenv("JUPR_ENV", "").strip().lower() == "test"
        and allow_non_atomic_test_adapter
    ):
        return
    raise PermissionError(
        "Tournament draw/event recovery requires its atomic database RPC; the non-atomic adapter is unit-test-only."
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception as exc:
        raise RuntimeError("Could not verify existing tournament draws; draw creation was refused.") from exc


def _event_label(row: dict[str, Any] | None) -> str:
    row = row or {}
    family = _clean_text(row.get("event_family_label"), limit=120)
    division = _clean_text(row.get("division_name") or row.get("label"), limit=120)
    if family and division and family != division:
        return f"{family} / {division}"
    return division or family or "Tournament Draw"


def _fetch_event_option(supabase: Any, *, tournament_id: str, event_option_id: str) -> dict[str, Any] | None:
    if not event_option_id:
        return None
    rows = _table_rows_for_tournament(
        supabase,
        "tournament_event_options",
        EVENT_OPTION_SELECT,
        tournament_id=str(tournament_id),
    )
    for row in rows:
        if _clean_text(row.get("id"), limit=120) == str(event_option_id):
            return row
    return None


def _existing_draws(supabase: Any, *, tournament_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_event_draws")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .execute()
        )
    except Exception:
        return []


def _draw_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "tournament_id": _clean_text(row.get("tournament_id"), limit=120),
        "registration_day_id": _clean_text(row.get("registration_day_id"), limit=120) or None,
        "event_option_id": _clean_text(row.get("event_option_id"), limit=120) or None,
        "name": _clean_text(row.get("name"), limit=180),
        "status": _clean_text(row.get("status") or "draft", limit=40).lower(),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def create_admin_tournament_draw(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_day_id: str | None = None,
    event_option_id: str | None = None,
    name: str | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_create_draw",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_CREATE_DRAW:
        raise ValueError(f"Type {CONFIRM_CREATE_DRAW} to create a tournament draw.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")

    clean_event_option_id = _clean_text(event_option_id, limit=120) or None
    event_option = _fetch_event_option(supabase, tournament_id=clean_tournament_id, event_option_id=clean_event_option_id or "")
    if clean_event_option_id and not event_option:
        raise ValueError("event option not found for this tournament")

    clean_day_id = _clean_text(registration_day_id, limit=120) or None
    if event_option:
        event_day_id = _clean_text(event_option.get("registration_day_id"), limit=120) or None
        if clean_day_id and event_day_id and clean_day_id != event_day_id:
            raise ValueError("registration_day_id does not match the selected event option")
        clean_day_id = event_day_id or clean_day_id

    clean_name = _clean_text(name, limit=180)
    if not clean_name:
        clean_name = f"{_event_label(event_option)} Ops Draw" if event_option else "Tournament Ops Draw"

    for existing in _existing_draws(supabase, tournament_id=clean_tournament_id):
        same_event = _clean_text(existing.get("event_option_id"), limit=120) == (clean_event_option_id or "")
        same_day = _clean_text(existing.get("registration_day_id"), limit=120) == (clean_day_id or "")
        same_name = _clean_text(existing.get("name"), limit=180).lower() == clean_name.lower()
        if same_event and same_day and same_name:
            raise ValueError("A draw with this day, event, and name already exists.")

    now = _now_iso()
    insert_payload = {
        "id": str(uuid.uuid4()),
        "tournament_id": clean_tournament_id,
        "registration_day_id": clean_day_id,
        "event_option_id": clean_event_option_id,
        "name": clean_name,
        "status": "draft",
        "created_at": now,
        "updated_at": now,
    }
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_draw_create_preview",
            "dry_run": True,
            "write_count": 0,
            "draw": _draw_payload(insert_payload),
            "warnings": [],
        }
    rows = _safe_rows(supabase.table("tournament_event_draws").insert(insert_payload).execute())
    draw = _draw_payload(rows[0] if rows else insert_payload)

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="create_tournament_event_draw_admin",
        entity_type="tournament_event_draw",
        entity_id=str(draw.get("id") or clean_name),
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": draw,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and str(__import__("os").getenv("JUPR_REQUIRE_API_AUDIT_LOG", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_draw_create", "draw": draw, "warnings": warnings}


def cancel_admin_tournament_empty_draw(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_cancel_empty_draw",
    dry_run: bool = False,
    atomic: bool = False,
    allow_non_atomic_test_adapter: bool = False,
) -> dict[str, Any]:
    """Exclude a truly empty draw from live operations and closeout.

    The draw row is retained for audit/recovery.  Any team, game, podium,
    official result, award, or day-live dependency blocks cancellation.
    """

    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    _require_atomic_recovery(
        atomic=atomic,
        allow_non_atomic_test_adapter=allow_non_atomic_test_adapter,
    )
    if str(confirmation_text or "").strip().upper() != CONFIRM_CANCEL_EMPTY_DRAW:
        raise ValueError(
            f"Type {CONFIRM_CANCEL_EMPTY_DRAW} to remove this empty draw from tournament operations."
        )
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=clean_tournament_id,
    )
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = next(
        (
            row
            for row in _existing_draws(supabase, tournament_id=clean_tournament_id)
            if str(row.get("id") or "") == clean_draw_id
        ),
        None,
    )
    if draw is None:
        raise ValueError("draw not found for this tournament")
    if str(draw.get("status") or "").strip().lower() in {
        "cancelled",
        "canceled",
        "inactive",
        "disabled",
        "archived",
    }:
        raise ValueError("This draw is already inactive.")
    if str(draw.get("updated_at") or "") != str(expected_draw_updated_at or ""):
        raise StaleTournamentAdminStateError(
            "This draw changed after it was reviewed. Reload Tournament Ops."
        )

    dependent_tables = (
        "tournament_teams",
        "tournament_games",
        "tournament_podium",
        "tournament_day_live_draws",
    )
    dependencies: dict[str, int] = {}
    for table_name in dependent_tables:
        try:
            count = len(
                _safe_rows(
                    supabase.table(table_name)
                    .select("*")
                    .eq("tournament_id", clean_tournament_id)
                    .eq("draw_id", clean_draw_id)
                    .execute()
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not verify {table_name}; empty draw cancellation was refused."
            ) from exc
        dependencies[table_name] = count
    try:
        award_rows = _safe_rows(
            supabase.table("player_badges")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("context_type", "tournament")
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not verify player_badges; empty draw cancellation was refused."
        ) from exc
    dependencies["player_badges"] = len(
        [
            row
            for row in award_rows
            if str(row.get("context_id") or "").startswith(
                f"{clean_tournament_id}:draw:{clean_draw_id}:"
            )
        ]
    )
    populated = {name: count for name, count in dependencies.items() if count}
    if populated:
        labels = ", ".join(f"{name}={count}" for name, count in sorted(populated.items()))
        raise ValueError(
            "Only a draw with no teams, games, podium, or day-live evidence can be cancelled as empty. "
            + labels
        )
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_empty_draw_cancel_preview",
            "dry_run": True,
            "write_count": 0,
            "draw": _draw_payload({**draw, "status": "cancelled"}),
            "dependencies": dependencies,
            "warnings": [],
        }

    if atomic:
        try:
            response = supabase.rpc(
                "admin_cancel_empty_tournament_draw_cas",
                {
                    "p_club_id": str(club_id),
                    "p_tournament_id": clean_tournament_id,
                    "p_draw_id": clean_draw_id,
                    "p_expected_draw_updated_at": str(expected_draw_updated_at),
                },
            ).execute()
        except Exception as exc:
            detail = str(exc)
            if "JUPR_TOURNAMENT_DRAW_STALE" in detail:
                raise StaleTournamentAdminStateError(
                    "This draw changed while empty cancellation was being committed. Reload Tournament Ops."
                ) from exc
            if "JUPR_TOURNAMENT_DRAW_NOT_EMPTY" in detail:
                raise ValueError(
                    "The draw gained teams, games, podium, official, award, or day-live evidence and was not cancelled."
                ) from exc
            raise RuntimeError(
                "Atomic empty draw cancellation failed; the draw remains active."
            ) from exc
        data = getattr(response, "data", None)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict) or not isinstance(data.get("draw"), dict):
            raise RuntimeError(
                "Atomic empty draw cancellation returned no authoritative draw."
            )
        saved = dict(data["draw"])
    else:
        rows = _safe_rows(
            supabase.table("tournament_event_draws")
            .update({"status": "cancelled", "updated_at": _now_iso()})
            .eq("tournament_id", clean_tournament_id)
            .eq("id", clean_draw_id)
            .eq("updated_at", str(expected_draw_updated_at))
            .execute()
        )
        if not rows:
            raise StaleTournamentAdminStateError(
                "This draw changed while empty cancellation was being committed. Reload Tournament Ops."
            )
        saved = rows[0]

    saved_draw = _draw_payload(saved)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="cancel_empty_tournament_draw_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "dependencies": dependencies},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": saved_draw,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and str(__import__("os").getenv("JUPR_REQUIRE_API_AUDIT_LOG", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "tournament_empty_draw_cancel",
        "draw": saved_draw,
        "dependencies": dependencies,
        "warnings": warnings,
    }


def cancel_admin_tournament_empty_event(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event_option_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_cancel_empty_event",
    dry_run: bool = False,
    atomic: bool = False,
    allow_non_atomic_test_adapter: bool = False,
) -> dict[str, Any]:
    """Disable an unused event option without deleting its setup history."""

    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    _require_atomic_recovery(
        atomic=atomic,
        allow_non_atomic_test_adapter=allow_non_atomic_test_adapter,
    )
    if str(confirmation_text or "").strip().upper() != CONFIRM_CANCEL_EMPTY_EVENT:
        raise ValueError(
            f"Type {CONFIRM_CANCEL_EMPTY_EVENT} to remove this unused event from registration and closeout."
        )
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_event_id = _clean_text(event_option_id, limit=120)
    tournament = _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=clean_tournament_id,
    )
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    event = _fetch_event_option(
        supabase,
        tournament_id=clean_tournament_id,
        event_option_id=clean_event_id,
    )
    if event is None:
        raise ValueError("event option not found for this tournament")
    enabled_value = event.get("enabled", True)
    enabled = (
        enabled_value
        if isinstance(enabled_value, bool)
        else str(enabled_value).strip().lower()
        not in {"0", "false", "no", "off", "disabled", "cancelled", "canceled"}
    )
    if not enabled:
        raise ValueError("This event is already disabled.")

    dependency_tables = (
        "tournament_registration_selections",
        "tournament_registration_team_links",
        "tournament_registration_team_members",
        "tournament_event_draws",
        "tournament_teams",
        "tournament_games",
    )
    dependencies: dict[str, int] = {}
    for table_name in dependency_tables:
        try:
            rows = _safe_rows(
                supabase.table(table_name)
                .select("*")
                .eq("tournament_id", clean_tournament_id)
                .eq("event_option_id", clean_event_id)
                .execute()
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not verify {table_name}; empty event cancellation was refused."
            ) from exc
        dependencies[table_name] = len(rows)
    populated = {name: count for name, count in dependencies.items() if count}
    if populated:
        labels = ", ".join(
            f"{name}={count}" for name, count in sorted(populated.items())
        )
        raise ValueError(
            "Only an event with no registrations, draws, teams, or games can be cancelled as empty. "
            + labels
        )
    cancelled = {**event, "enabled": False, "status": "cancelled"}
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_empty_event_cancel_preview",
            "dry_run": True,
            "write_count": 0,
            "event_option": cancelled,
            "dependencies": dependencies,
            "warnings": [],
        }
    if atomic:
        try:
            response = supabase.rpc(
                "admin_cancel_empty_tournament_event_cas",
                {
                    "p_club_id": str(club_id),
                    "p_tournament_id": clean_tournament_id,
                    "p_event_option_id": clean_event_id,
                },
            ).execute()
        except Exception as exc:
            detail = str(exc)
            if "JUPR_TOURNAMENT_EVENT_NOT_EMPTY" in detail:
                raise ValueError(
                    "The event gained registration or draw evidence and was not cancelled."
                ) from exc
            if "JUPR_TOURNAMENT_EVENT_STALE" in detail:
                raise StaleTournamentAdminStateError(
                    "The event changed while cancellation was being committed. Reload Tournament Ops."
                ) from exc
            raise RuntimeError(
                "Atomic empty event cancellation failed; the event remains enabled."
            ) from exc
        data = getattr(response, "data", None)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict) or not isinstance(data.get("event_option"), dict):
            raise RuntimeError(
                "Atomic empty event cancellation returned no authoritative event."
            )
        saved = dict(data["event_option"])
    else:
        rows = _safe_rows(
            supabase.table("tournament_event_options")
            .update({"enabled": False, "status": "cancelled"})
            .eq("tournament_id", clean_tournament_id)
            .eq("id", clean_event_id)
            .eq("enabled", True)
            .execute()
        )
        if not rows:
            raise StaleTournamentAdminStateError(
                "The event changed while cancellation was being committed. Reload Tournament Ops."
            )
        saved = rows[0]
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="cancel_empty_tournament_event_admin",
        entity_type="tournament_event_option",
        entity_id=clean_event_id,
        before_json={"event_option": event, "dependencies": dependencies},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "event_option": saved,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and str(__import__("os").getenv("JUPR_REQUIRE_API_AUDIT_LOG", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "tournament_empty_event_cancel",
        "event_option": saved,
        "dependencies": dependencies,
        "warnings": warnings,
    }
