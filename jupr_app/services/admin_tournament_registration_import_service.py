from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
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


def _event_option_for_draw(
    supabase: Any,
    *,
    tournament_id: str,
    draw: dict[str, Any],
) -> dict[str, Any]:
    event_option_id = _clean_text(draw.get("event_option_id"), limit=120)
    if not event_option_id:
        return {}
    try:
        rows = _safe_rows(
            supabase.table("tournament_event_options")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("id", event_option_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not verify the tournament event eligibility contract; "
            "registration import was refused."
        ) from exc
    if not rows:
        raise ValueError(
            "The draw's tournament event option is missing; registration "
            "import was refused."
        )
    return rows[0]


def _write_combined_rating_teams_atomic(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    rows: list[dict[str, Any]],
    replace: bool,
    actor_email: str,
) -> list[dict[str, Any]]:
    """Use the eligibility-snapshot RPC for every combined-cap import."""

    if not str(expected_draw_updated_at or "").strip():
        raise ValueError(
            "A reviewed draw version is required for a combined-rating import."
        )
    operation = build_tournament_admin_operation_request(
        club_id=str(club_id),
        surface="import_handoff",
        action="combined_rating_draw_team_write",
        entity_type="tournament_event_draw",
        entity_id=str(draw_id),
        lock_scope=str(tournament_id),
        expected_state=str(expected_draw_updated_at),
        # Exclude generated row IDs/timestamps and derived append slots from
        # the operation identity. An exact recovery attempt after a committed
        # response loss must reach the database operation replay even though
        # the current draw rows now exist.
        payload={
            "replace": bool(replace),
            "entries": sorted(
                [
                    {
                        "source_selection_id": str(
                            row.get("source_selection_id") or ""
                        ),
                        "player1_id": _safe_int(row.get("player1_id")),
                        "player2_id": _safe_int(row.get("player2_id")),
                        "seed": _safe_int(row.get("seed")),
                    }
                    for row in rows
                ],
                key=lambda row: (
                    row["source_selection_id"],
                    int(row["player1_id"] or 0),
                    int(row["player2_id"] or 0),
                ),
            ),
        },
    )
    try:
        response = supabase.rpc(
            "admin_write_combined_rating_draw_teams_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_expected_draw_updated_at": str(expected_draw_updated_at),
                "p_replace": bool(replace),
                "p_teams": list(rows),
                "p_operation_key": operation["operation_key"],
                "p_request_fingerprint": operation["request_fingerprint"],
                "p_actor": str(actor_email or ""),
            },
        ).execute()
    except Exception as exc:
        if any(
            marker in str(exc)
            for marker in (
                "JUPR_TOURNAMENT_COMBINED_RATING_DRAW_STALE",
                "JUPR_TOURNAMENT_COMBINED_RATING_DRAW_IN_USE",
                "JUPR_TOURNAMENT_COMBINED_RATING_DRAW_APPEND_CONFLICT",
                "JUPR_TOURNAMENT_COMBINED_RATING_DRAW_BLOCKED",
            )
        ):
            raise ValueError(
                "The combined-rating draw or its finalized eligibility "
                "evidence changed. Reload the reviewed Ops snapshot."
            ) from exc
        raise RuntimeError(
            "Atomic combined-rating registration import failed; no team set "
            "was committed."
        ) from exc
    data = getattr(response, "data", None)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    saved = data.get("teams") if isinstance(data, dict) else None
    if not isinstance(saved, list):
        raise RuntimeError(
            "Combined-rating import returned no durable saved-team evidence."
        )
    return [dict(row) for row in saved if isinstance(row, dict)]


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


def _finalized_combined_rating_reviews(
    supabase: Any,
    *,
    tournament_id: str,
    event_option_id: str,
) -> dict[str, dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_rating_eligibility_reviews")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("event_option_id", str(event_option_id))
            .eq("review_phase", "REGISTRATION_CLOSE")
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load finalized combined-rating evidence; "
            "registration import was refused."
        ) from exc
    reviews: dict[str, dict[str, Any]] = {}
    for review in rows:
        selection_id = _clean_text(review.get("selection_id"), limit=120)
        if not selection_id:
            continue
        if selection_id in reviews:
            raise RuntimeError(
                "Duplicate finalized combined-rating evidence exists for a "
                "registration selection."
            )
        reviews[selection_id] = review
    return reviews


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


def _is_confirmed_registration(row: dict[str, Any]) -> bool:
    return (
        _clean_text(
            row.get("status") or row.get("registration_status") or "",
            limit=40,
        )
        .lower()
        .replace("-", "_")
        in {"confirmed", "admin_confirmed"}
    )


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
    event_option = _event_option_for_draw(
        supabase,
        tournament_id=clean_tournament_id,
        draw=draw,
    )
    combined_rating_import = (
        str(event_option.get("eligibility_mode") or "").upper()
        == "COMBINED_RATING_CAP"
    )
    if combined_rating_import:
        reviewed_draw_version = str(expected_draw_updated_at or "").strip()
        if not reviewed_draw_version:
            raise ValueError(
                "A reviewed draw version is required for a combined-rating "
                "registration import."
            )
        # The combined-rating RPC checks the draw CAS after first checking its
        # durable operation record. Do not reject an exact recovery attempt
        # locally merely because its first commit advanced draw.updated_at.
    else:
        reviewed_draw_version = _require_reviewed_draw_version(
            draw,
            expected_draw_updated_at=expected_draw_updated_at,
            atomic=atomic,
        )
    if (
        not combined_rating_import
        and _games_for_draw(
            supabase,
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
        )
    ):
        raise ValueError("This draw already has games. Registration import is blocked after scheduling begins.")

    mode = _clean_text(import_mode or "REPLACE", limit=20).upper()
    if mode not in {"REPLACE", "APPEND"}:
        raise ValueError("import_mode must be REPLACE or APPEND")

    registrations = _registrations(supabase, tournament_id=clean_tournament_id)
    registrations_by_id = {_clean_text(row.get("id"), limit=120): row for row in registrations}
    registrations_by_email = {_email(row.get("email")): row for row in registrations if _email(row.get("email"))}
    selections = _selections_for_draw(supabase, tournament_id=clean_tournament_id, draw=draw)
    finalized_reviews = (
        _finalized_combined_rating_reviews(
            supabase,
            tournament_id=clean_tournament_id,
            event_option_id=_clean_text(event_option.get("id"), limit=120),
        )
        if combined_rating_import
        else {}
    )

    current_teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    start_slot = max([_safe_int(row.get("team_number")) or 0 for row in current_teams], default=0) + 1 if mode == "APPEND" else 1
    selected_player_ids: list[int] = []
    unresolved: list[str] = []
    incomplete_combined_reviews: list[str] = []
    rows: list[dict[str, Any]] = []
    now = _now_iso()

    for selection in selections:
        registration = registrations_by_id.get(_clean_text(selection.get("registration_id"), limit=120))
        if not registration:
            continue
        if not _is_confirmed_registration(registration):
            continue
        player1_id = None
        player2_id = None
        if combined_rating_import:
            selection_id = _clean_text(selection.get("id"), limit=120)
            review = finalized_reviews.get(selection_id)
            effective_state = str(
                (review or {}).get("override_state")
                or (review or {}).get("state")
                or ""
            ).upper()
            override_valid = not (review or {}).get("override_state") or bool(
                _clean_text((review or {}).get("override_reason"), limit=500)
            )
            if (
                not review
                or not review.get("finalized_at")
                or effective_state not in {"ELIGIBLE", "INELIGIBLE"}
                or not override_valid
            ):
                incomplete_combined_reviews.append(
                    _clean_text(
                        registration.get("display_name")
                        or registration.get("email")
                        or registration.get("id"),
                        limit=180,
                    )
                )
                continue
            if effective_state == "INELIGIBLE":
                continue
            if (
                _clean_text(review.get("registration_id"), limit=120)
                != _clean_text(registration.get("id"), limit=120)
            ):
                incomplete_combined_reviews.append(
                    _clean_text(
                        registration.get("display_name")
                        or registration.get("email")
                        or registration.get("id"),
                        limit=180,
                    )
                )
                continue
            player1_id = _safe_int(review.get("player_id_snapshot"))
            player2_id = _safe_int(review.get("partner_player_id_snapshot"))
            if (
                player1_id is None
                or player2_id is None
                or _safe_int(registration.get("player_id")) != player1_id
            ):
                unresolved.append(
                    _clean_text(
                        registration.get("display_name")
                        or registration.get("email")
                        or registration.get("id"),
                        limit=180,
                    )
                )
                continue
            partner_registration_id = _clean_text(
                review.get("partner_registration_id"), limit=120
            )
            partner = registrations_by_id.get(partner_registration_id)
            if (
                not partner
                or not _is_confirmed_registration(partner)
            ):
                unresolved.append(
                    _clean_text(
                        (partner or {}).get("display_name")
                        or partner_registration_id
                        or "missing combined-rating partner",
                        limit=180,
                    )
                )
                continue
            if _safe_int(partner.get("player_id")) != player2_id:
                unresolved.append(
                    _clean_text(
                        partner.get("display_name")
                        or partner.get("email")
                        or partner_registration_id,
                        limit=180,
                    )
                )
                continue
        else:
            player1_id = _safe_int(registration.get("player_id"))
            if player1_id is None:
                unresolved.append(_clean_text(registration.get("display_name") or registration.get("email") or registration.get("id"), limit=180))
                continue
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
                "source": (
                    "REGISTRATION_COMBINED_RATING"
                    if combined_rating_import
                    else "REGISTRATION"
                ),
                "notes": f"Imported from registration {_clean_text(registration.get('id'), limit=120)}",
                **(
                    {
                        "source_selection_id": _clean_text(
                            selection.get("id"), limit=120
                        )
                    }
                    if combined_rating_import
                    else {}
                ),
                "created_at": now,
            }
        )

    duplicates = sorted({pid for pid in selected_player_ids if selected_player_ids.count(pid) > 1})
    if duplicates:
        raise ValueError("Duplicate player IDs in confirmed registration import: " + ", ".join(str(pid) for pid in duplicates))
    if incomplete_combined_reviews:
        raise ValueError(
            "Combined-rating registration-close reviews are incomplete for: "
            + ", ".join(sorted(set(filter(None, incomplete_combined_reviews))))
        )
    if unresolved:
        raise ValueError("Some confirmed registrations could not be resolved to JUPR players: " + ", ".join(sorted(set(filter(None, unresolved)))))
    if not rows:
        if combined_rating_import:
            raise ValueError(
                "No eligible finalized combined-rating registrations were "
                "available for this draw."
            )
        raise ValueError("No confirmed registrations with linked player IDs were available for this draw.")

    before = [_team_payload(row) for row in current_teams]
    if dry_run:
        teams = [_team_payload(row) for row in rows]
        if combined_rating_import:
            teams = [
                {
                    **team,
                    "source_selection_id": _clean_text(
                        row.get("source_selection_id"), limit=120
                    )
                    or None,
                }
                for team, row in zip(teams, rows, strict=False)
            ]
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
    if combined_rating_import:
        inserted = _write_combined_rating_teams_atomic(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            expected_draw_updated_at=reviewed_draw_version,
            rows=rows,
            replace=mode == "REPLACE",
            actor_email=actor_email,
        )
    elif atomic:
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
    if combined_rating_import:
        teams = [
            {
                **team,
                "source_selection_id": _clean_text(
                    row.get("source_selection_id"), limit=120
                )
                or None,
            }
            for team, row in zip(teams, inserted or rows, strict=False)
        ]

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
