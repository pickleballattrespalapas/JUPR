from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_game_service import _require_reviewed_draw_version
from jupr_app.services.admin_tournament_team_service import _team_payload, write_admin_tournament_draw_teams_atomic
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_IMPORT_TEAMS = "IMPORT TEAMS"


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


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _fetch_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> dict[str, Any] | None:
    try:
        rows = _safe_rows(supabase.table("tournament_event_draws").select("*").eq("tournament_id", str(tournament_id)).eq("id", str(draw_id)).limit(1).execute())
    except Exception as exc:
        raise RuntimeError("Could not verify the tournament draw; bulk team import was refused.") from exc
    return rows[0] if rows else None


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table("tournament_games").select("id").eq("tournament_id", str(tournament_id)).eq("draw_id", str(draw_id)).limit(1).execute())
    except Exception as exc:
        raise RuntimeError("Could not verify whether this draw already has games; bulk team import was refused.") from exc


def _teams_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table("tournament_teams").select("*").eq("tournament_id", str(tournament_id)).eq("draw_id", str(draw_id)).execute())
    except Exception as exc:
        raise RuntimeError("Could not load current draw teams; bulk team import was refused.") from exc


def _players_by_name_and_id(supabase: Any, *, club_id: str) -> tuple[dict[str, int], set[int]]:
    try:
        rows = _safe_rows(supabase.table("players").select("id,name,club_id,active").eq("club_id", str(club_id)).execute())
    except Exception as exc:
        raise RuntimeError("Could not load the club player roster; bulk team import was refused.") from exc
    by_name: dict[str, int] = {}
    ids: set[int] = set()
    for row in rows:
        player_id = _safe_int(row.get("id"))
        if player_id is None:
            continue
        ids.add(player_id)
        name = _normalize_name(row.get("name"))
        if name:
            by_name[name] = player_id
    return by_name, ids


def _parse_rows(raw_text: str) -> list[dict[str, str]]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Paste CSV or TSV team rows before importing.")
    delimiter = "\t" if "\t" in text and text.count("\t") >= text.count(",") else ","
    reader = csv.DictReader(StringIO(text), delimiter=delimiter)
    rows = [{str(key or "").strip(): str(value or "").strip() for key, value in row.items()} for row in reader]
    if not reader.fieldnames or not rows:
        raise ValueError("Could not parse team rows. Include headers such as Player 1, Player 2, Seed, Notes.")
    return rows


def _first(row: dict[str, str], *keys: str) -> str:
    normalized = {" ".join(str(key).strip().lower().replace("_", " ").split()): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(" ".join(key.strip().lower().replace("_", " ").split()))
        if value:
            return value
    return ""


def _resolve_player(value: str, by_name: dict[str, int], ids: set[int]) -> int | None:
    numeric = _safe_int(value)
    if numeric is not None and numeric in ids:
        return numeric
    return by_name.get(_normalize_name(value))


def import_admin_tournament_bulk_teams(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    raw_text: str,
    import_mode: str = "REPLACE",
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    source: str = "next_tournament_admin_import_bulk_teams",
    dry_run: bool = False,
    atomic: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_IMPORT_TEAMS:
        raise ValueError(f"Type {CONFIRM_IMPORT_TEAMS} to import bulk teams into this draw.")

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
        raise ValueError("This draw already has games. Bulk team import is blocked after scheduling begins.")

    mode = _clean_text(import_mode or "REPLACE", limit=20).upper()
    if mode not in {"REPLACE", "APPEND"}:
        raise ValueError("import_mode must be REPLACE or APPEND")

    parsed = _parse_rows(raw_text)
    by_name, ids = _players_by_name_and_id(supabase, club_id=str(club_id))
    current_teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    start_slot = max([_safe_int(row.get("team_number")) or 0 for row in current_teams], default=0) + 1 if mode == "APPEND" else 1
    selected_player_ids: list[int] = []
    unresolved: list[str] = []
    now = _now_iso()
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(parsed):
        p1_text = _first(item, "Player 1", "Player1", "Player One", "P1", "player1_id")
        p2_text = _first(item, "Player 2", "Player2", "Player Two", "P2", "player2_id")
        if not p1_text and not p2_text:
            continue
        player1_id = _resolve_player(p1_text, by_name, ids)
        player2_id = _resolve_player(p2_text, by_name, ids) if p2_text else None
        if player1_id is None:
            unresolved.append(p1_text or f"row {index + 1} player 1")
            continue
        if p2_text and player2_id is None:
            unresolved.append(p2_text)
            continue
        selected_player_ids.append(player1_id)
        if player2_id is not None:
            selected_player_ids.append(player2_id)
        team_number = _safe_int(_first(item, "Team / Slot", "Team Number", "Team", "Slot")) or (start_slot + len(rows))
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "tournament_id": clean_tournament_id,
                "draw_id": clean_draw_id,
                "registration_day_id": _clean_text(draw.get("registration_day_id"), limit=120) or None,
                "event_option_id": _clean_text(draw.get("event_option_id"), limit=120) or None,
                "team_number": team_number,
                "player1_id": player1_id,
                "player2_id": player2_id,
                "seed": _safe_int(_first(item, "Seed")),
                "source": "BULK_UPLOAD",
                "notes": _clean_text(_first(item, "Notes", "Note", "Team Name"), limit=500) or None,
                "created_at": now,
            }
        )

    if unresolved:
        raise ValueError("Unresolved player names or IDs: " + ", ".join(sorted(set(filter(None, unresolved)))))
    duplicates = sorted({pid for pid in selected_player_ids if selected_player_ids.count(pid) > 1})
    if duplicates:
        raise ValueError("Duplicate player IDs in bulk team import: " + ", ".join(str(pid) for pid in duplicates))
    if not rows:
        raise ValueError("No team rows were available to import.")
    team_numbers = [int(row["team_number"]) for row in rows]
    duplicate_slots = sorted({slot for slot in team_numbers if team_numbers.count(slot) > 1})
    if duplicate_slots:
        raise ValueError("Duplicate team numbers in bulk team import: " + ", ".join(str(slot) for slot in duplicate_slots))

    before = [_team_payload(row) for row in current_teams]
    if dry_run:
        teams = [_team_payload(row) for row in rows]
        return {
            "ok": True,
            "mode": "tournament_bulk_team_import_preview",
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
        action_type="import_tournament_bulk_teams_admin",
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
    return {"ok": True, "mode": "tournament_bulk_team_import", "draw_id": clean_draw_id, "import_mode": mode, "updated_count": len(teams), "teams": teams, "warnings": warnings}
