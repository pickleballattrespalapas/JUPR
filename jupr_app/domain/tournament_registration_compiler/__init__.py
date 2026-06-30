from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_LEGACY_MODULE_NAME = "jupr_app.domain._tournament_registration_compiler_legacy"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "tournament_registration_compiler.py"
_CANCELLED_REGISTRATION_STATUSES = {"cancelled", "canceled"}


def _load_legacy_module():
    module = sys.modules.get(_LEGACY_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy tournament registration compiler from {_LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()
for _name in dir(_legacy):
    if _name.startswith("__") and _name.endswith("__"):
        continue
    globals()[_name] = getattr(_legacy, _name)

_LEGACY_COMPILE_SINGLES_ROSTER = _legacy._compile_singles_roster
_LEGACY_COMPILE_DOUBLES_ROSTER = _legacy._compile_doubles_roster


def _first_text(values: list[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _registration_is_cancelled(registration: dict[str, Any] | None) -> bool:
    return str((registration or {}).get("status") or "").strip().lower() in _CANCELLED_REGISTRATION_STATUSES


def _active_public_selections(selections: list[dict[str, Any]], reg_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        selection
        for selection in selections
        if not _registration_is_cancelled(reg_lookup.get(str(selection.get("registration_id"))))
    ]


def _compile_singles_roster(
    tournament_id: str,
    day: dict[str, Any],
    event: dict[str, Any],
    selections: list[dict[str, Any]],
    reg_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return _LEGACY_COMPILE_SINGLES_ROSTER(
        tournament_id,
        day,
        event,
        _active_public_selections(selections, reg_lookup),
        reg_lookup,
    )


def _append_missing_public_partner_board_rows(
    *,
    tournament_id: str,
    day: dict[str, Any],
    event: dict[str, Any],
    rows: list[dict[str, Any]],
    partner_board: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_selection_ids = {str(row.get("selection_id") or "").strip() for row in partner_board}
    for row in rows:
        if str(row.get("status") or "").upper() != "NEEDS_PARTNER":
            continue
        selection_ids = [str(value or "").strip() for value in (row.get("source_selection_ids") or []) if str(value or "").strip()]
        selection_id = selection_ids[0] if selection_ids else ""
        if not selection_id or selection_id in existing_selection_ids:
            continue

        registration_id = _first_text(row.get("source_registration_ids") or [])
        player_id = _first_text(row.get("source_player_ids") or [])
        member = ((row.get("members") or [{}])[0] or {}).copy()
        partner_board.append(
            {
                "id": _legacy._uid("partner"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": str(event.get("id")),
                "event_label": str(event.get("label") or ""),
                "selection_id": selection_id,
                "registration_id": registration_id,
                "player_id": player_id or member.get("player_id"),
                "player": member,
                "note": row.get("notes"),
                "show_contact_email": True,
            }
        )
        existing_selection_ids.add(selection_id)
    return partner_board


def _compile_doubles_roster(
    tournament_id: str,
    day: dict[str, Any],
    event: dict[str, Any],
    selections: list[dict[str, Any]],
    reg_lookup: dict[str, dict[str, Any]],
    *,
    partner_requests: list[dict[str, Any]] | None = None,
    partner_links: list[dict[str, Any]] | None = None,
    team_members: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, partner_board, issues = _LEGACY_COMPILE_DOUBLES_ROSTER(
        tournament_id,
        day,
        event,
        _active_public_selections(selections, reg_lookup),
        reg_lookup,
        partner_requests=partner_requests,
        partner_links=partner_links,
        team_members=team_members,
    )
    partner_board = _append_missing_public_partner_board_rows(
        tournament_id=tournament_id,
        day=day,
        event=event,
        rows=rows,
        partner_board=partner_board,
    )
    return rows, partner_board, issues


_legacy._compile_singles_roster = _compile_singles_roster
_legacy._compile_doubles_roster = _compile_doubles_roster
globals()["_compile_singles_roster"] = _compile_singles_roster
globals()["_compile_doubles_roster"] = _compile_doubles_roster
compile_tournament_registration_state = _legacy.compile_tournament_registration_state
