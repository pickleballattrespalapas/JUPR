from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_admin_operations import stable_tournament_admin_fingerprint
from jupr_app.domain.tournament_results_import import (
    build_draw_import_payload,
    parse_dupr_results_csv,
    suggest_player_matches,
)
from jupr_app.domain.tournaments import finalize_game
from jupr_app.services.admin_tournament_game_service import _require_reviewed_draw_version
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    is_admin_tournament_admin_enabled,
)

MAX_RESULTS_IMPORT_BYTES = 1_000_000
TRUTHY = {"1", "true", "yes", "y", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


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
        raise RuntimeError("Could not verify the tournament draw; results import was refused.") from exc
    return rows[0] if rows else None


def _club_players(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("players")
            .select("*")
            .eq("club_id", str(club_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load the club player roster; results import was refused.") from exc


def _draw_teams(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_teams")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load current draw teams; results import was refused.") from exc


def _default_mapping_decisions(
    imported_players: list[dict[str, Any]],
    suggestions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for imported in imported_players:
        import_key = str(imported.get("import_key") or "")
        suggestion = suggestions.get(import_key) or {}
        player_id = suggestion.get("suggested_player_id")
        decisions[import_key] = (
            {"action": "use_existing", "player_id": player_id}
            if player_id not in (None, "")
            else {"action": "unresolved", "player_id": None}
        )
    return decisions


def _default_match_reviews(matches: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(int(row.get("source_row") or 0)): {
            "include": bool(row.get("include", True)),
            "stage": str(row.get("stage") or "PLAYOFF").upper(),
        }
        for row in matches
    }


def build_admin_tournament_results_import_preview(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    raw_text: str,
    import_mode: str = "REPLACE",
    mapping_decisions: dict[str, dict[str, Any]] | None = None,
    match_reviews: dict[str, dict[str, Any]] | None = None,
    podium_refs: dict[str, str | None] | None = None,
    allow_duplicate_mapping: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")

    encoded = str(raw_text or "").encode("utf-8")
    if not encoded:
        raise ValueError("Paste a DUPR-style results CSV before previewing the import.")
    if len(encoded) > MAX_RESULTS_IMPORT_BYTES:
        raise ValueError("Results imports are capped at 1 MB per reviewed request.")
    mode = _clean_text(import_mode or "REPLACE", limit=20).upper()
    if mode not in {"REPLACE", "APPEND"}:
        raise ValueError("import_mode must be REPLACE or APPEND")

    bundle = parse_dupr_results_csv(encoded)
    if bundle.get("errors"):
        raise ValueError("; ".join(str(error) for error in bundle.get("errors") or []))
    if not bundle.get("matches"):
        raise ValueError("The results CSV did not contain any importable matches.")
    players = _club_players(supabase, club_id=str(club_id))
    suggestions = suggest_player_matches(list(bundle.get("players") or []), players)
    decisions = {
        str(key): dict(value or {})
        for key, value in (
            mapping_decisions
            if mapping_decisions is not None
            else _default_mapping_decisions(list(bundle.get("players") or []), suggestions)
        ).items()
    }
    reviews = {
        str(key): dict(value or {})
        for key, value in (
            match_reviews
            if match_reviews is not None
            else _default_match_reviews(list(bundle.get("matches") or []))
        ).items()
    }
    compiled = build_draw_import_payload(
        bundle=bundle,
        mapping_decisions=decisions,
        match_reviews=reviews,
    )
    errors = [str(error) for error in compiled.get("errors") or []]

    existing_ids = {str(row.get("id")) for row in players if row.get("id") not in (None, "")}
    existing_names = {
        _normalize_name(row.get("name")): str(row.get("id"))
        for row in players
        if row.get("id") not in (None, "") and _normalize_name(row.get("name"))
    }
    imported_by_key = {str(row.get("import_key")): row for row in bundle.get("players") or []}
    mapped_existing = []
    create_names: dict[str, str] = {}
    for import_key, decision in decisions.items():
        action = str(decision.get("action") or "")
        if action == "use_existing":
            player_id = str(decision.get("player_id") or "")
            if player_id not in existing_ids:
                errors.append(f"Player mapping {import_key} does not belong to this club.")
            mapped_existing.append(player_id)
        elif action == "create_new":
            display_name = _clean_text((imported_by_key.get(str(import_key)) or {}).get("display_name"), limit=180)
            normalized = _normalize_name(display_name)
            if not normalized:
                errors.append(f"Player mapping {import_key} has no valid name to create.")
            elif normalized in existing_names:
                errors.append(
                    f"Player mapping {import_key} was reviewed as create_new, but {display_name} already exists in this club. Choose that player explicitly."
                )
            elif normalized in create_names:
                errors.append(
                    f"Player mappings {create_names[normalized]} and {import_key} would create the same normalized name. Resolve the duplicate first."
                )
            else:
                create_names[normalized] = str(import_key)
            continue
        else:
            # build_draw_import_payload already supplies a friendly unresolved error.
            continue
    duplicates = sorted({player_id for player_id in mapped_existing if player_id and mapped_existing.count(player_id) > 1})
    if duplicates and not allow_duplicate_mapping:
        errors.append("Multiple imported players map to the same existing JUPR player; resolve them or explicitly allow duplicate mapping.")

    # One player may belong to at most one team in a draw. Enforce this on the
    # reviewed references before any transaction and repeat it inside the RPC.
    seen_roster_refs: set[str] = set()
    for team in compiled.get("teams") or []:
        for ref in (team.get("p1_ref"), team.get("p2_ref")):
            clean_ref = str(ref or "")
            if not clean_ref:
                continue
            if clean_ref in seen_roster_refs:
                errors.append(f"Reviewed player reference {clean_ref} appears on more than one imported team.")
            seen_roster_refs.add(clean_ref)
    if mode == "APPEND":
        assigned_player_ids = {
            str(value)
            for team in _draw_teams(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
            for value in (team.get("player1_id"), team.get("player2_id"))
            if value not in (None, "")
        }
        appended_existing_ids = {
            ref.split("existing:", 1)[1]
            for ref in seen_roster_refs
            if ref.startswith("existing:")
        }
        conflicts = sorted(assigned_player_ids.intersection(appended_existing_ids))
        if conflicts:
            errors.append("APPEND cannot assign a player who is already on another team in this draw.")

    default_podium = {
        str(index): team_ref
        for index, team_ref in enumerate(list(compiled.get("podium_candidates") or [])[:3], start=1)
    }
    reviewed_podium = {
        str(key): (str(value) if value else None)
        for key, value in (podium_refs if podium_refs is not None else default_podium).items()
        if str(key) in {"1", "2", "3"}
    }
    valid_team_refs = {str(row.get("team_ref")) for row in compiled.get("teams") or []}
    for placement, team_ref in reviewed_podium.items():
        if team_ref and team_ref not in valid_team_refs:
            errors.append(f"Podium placement {placement} references a team outside this reviewed import.")

    review_contract = {
        "contract": "jupr:tournament-results-import:v1",
        "club_id": str(club_id),
        "tournament_id": clean_tournament_id,
        "draw_id": clean_draw_id,
        "raw_fingerprint": stable_tournament_admin_fingerprint(str(raw_text or "")),
        "import_mode": mode,
        "mapping_decisions": decisions,
        "match_reviews": reviews,
        "podium_refs": reviewed_podium,
        "allow_duplicate_mapping": bool(allow_duplicate_mapping),
        "compiled": compiled,
    }
    review_fingerprint = stable_tournament_admin_fingerprint(review_contract)
    return {
        "ok": not errors,
        "mode": "tournament_results_import_preview",
        "dry_run": True,
        "write_count": 0,
        "tournament_id": clean_tournament_id,
        "draw_id": clean_draw_id,
        "import_mode": mode,
        "review_fingerprint": review_fingerprint,
        "players": list(bundle.get("players") or []),
        "player_options": [
            {"id": row.get("id"), "name": row.get("name"), "dupr_id": row.get("dupr_id")}
            for row in players
            if row.get("id") not in (None, "")
        ],
        "suggestions": suggestions,
        "mapping_decisions": decisions,
        "matches": list(bundle.get("matches") or []),
        "match_reviews": reviews,
        "teams": list(compiled.get("teams") or []),
        "podium_candidates": list(compiled.get("podium_candidates") or []),
        "podium_refs": reviewed_podium,
        "summary": {
            "imported_players": len(bundle.get("players") or []),
            "teams": len(compiled.get("teams") or []),
            "matches": len(compiled.get("matches") or []),
            "create_players": len(compiled.get("create_import_keys") or []),
        },
        "errors": errors,
        "warnings": list(compiled.get("warnings") or []),
        "review_contract": review_contract,
    }


def _player_refs_for_commit(*, preview: dict[str, Any]) -> tuple[set[str], list[dict[str, str]]]:
    imported_by_key = {str(row.get("import_key")): row for row in preview.get("players") or []}
    existing_refs: set[str] = set()
    new_players: list[dict[str, str]] = []
    for import_key, ref in (preview.get("review_contract", {}).get("compiled", {}).get("mapped_player_refs") or {}).items():
        ref_text = str(ref or "")
        if ref_text.startswith("existing:"):
            existing_refs.add(ref_text)
            continue
        imported = imported_by_key.get(str(import_key)) or {}
        display_name = _clean_text(imported.get("display_name"), limit=180)
        if not ref_text.startswith("create:") or not display_name:
            raise RuntimeError(f"Reviewed player mapping {import_key} is invalid.")
        new_players.append({"ref": ref_text, "name": display_name})
    return existing_refs, new_players


def apply_admin_tournament_results_import(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    raw_text: str,
    import_mode: str,
    mapping_decisions: dict[str, dict[str, Any]],
    match_reviews: dict[str, dict[str, Any]],
    podium_refs: dict[str, str | None],
    allow_duplicate_mapping: bool,
    expected_review_fingerprint: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    source: str = "next_tournament_ops_results_import",
    dry_run: bool = False,
    atomic: bool = False,
) -> dict[str, Any]:
    preview = build_admin_tournament_results_import_preview(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        draw_id=str(draw_id),
        raw_text=raw_text,
        import_mode=import_mode,
        mapping_decisions=mapping_decisions,
        match_reviews=match_reviews,
        podium_refs=podium_refs,
        allow_duplicate_mapping=allow_duplicate_mapping,
    )
    if preview.get("errors"):
        raise ValueError("; ".join(str(error) for error in preview.get("errors") or []))
    draw = _fetch_draw(supabase, tournament_id=str(tournament_id), draw_id=str(draw_id))
    if not draw:
        raise ValueError("draw not found for this tournament")
    reviewed_draw_version = _require_reviewed_draw_version(
        draw,
        expected_draw_updated_at=expected_draw_updated_at,
        atomic=atomic,
    )
    reviewed = str(expected_review_fingerprint or "").strip()
    if not reviewed or reviewed != str(preview.get("review_fingerprint") or ""):
        raise ValueError("Results import preview changed. Review the parsed mappings and matches again before committing.")
    mode = str(preview.get("import_mode") or "").upper()
    expected_confirmation = "REPLACE RESULTS" if mode == "REPLACE" else "IMPORT RESULTS"
    if str(confirmation_text or "").strip().upper() != expected_confirmation:
        raise ValueError(f"Type {expected_confirmation} to commit this reviewed results import.")
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_results_import_commit_preview",
            "dry_run": True,
            "write_count": 0,
            "review_fingerprint": reviewed,
            **dict(preview.get("summary") or {}),
        }

    compiled = dict(preview.get("review_contract", {}).get("compiled") or {})
    existing_refs, new_players = _player_refs_for_commit(preview=preview)
    current_teams = _draw_teams(supabase, tournament_id=str(tournament_id), draw_id=str(draw_id))
    start_slot = 1 if mode == "REPLACE" else max([int(row.get("team_number") or 0) for row in current_teams], default=0) + 1
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"jupr:tournament-results-import:{reviewed}")
    team_id_by_ref: dict[str, str] = {}
    team_number_by_ref: dict[str, int] = {}
    team_rows: list[dict[str, Any]] = []
    for index, row in enumerate(compiled.get("teams") or [], start=0):
        team_ref = str(row.get("team_ref") or "")
        team_id = str(uuid.uuid5(namespace, f"team:{team_ref}"))
        team_number = start_slot + index
        team_id_by_ref[team_ref] = team_id
        team_number_by_ref[team_ref] = team_number
        p1_ref = str(row.get("p1_ref") or "")
        p2_ref = str(row.get("p2_ref") or "") if row.get("p2_ref") else ""
        known_refs = existing_refs.union({str(row.get("ref")) for row in new_players})
        if p1_ref not in known_refs or (p2_ref and p2_ref not in known_refs):
            raise RuntimeError("Reviewed player mappings could not be resolved for an imported team.")
        team_rows.append(
            {
                "id": team_id,
                "team_number": team_number,
                "player1_ref": p1_ref,
                "player2_ref": p2_ref or None,
                "seed": team_number,
                "notes": "Reviewed DUPR results import",
            }
        )
    if not team_rows:
        raise ValueError("No reviewed teams were available to import.")

    game_rows: list[dict[str, Any]] = []
    rr_index = 0
    for index, row in enumerate(compiled.get("matches") or [], start=1):
        team_a_id = team_id_by_ref.get(str(row.get("team_a_ref") or ""))
        team_b_id = team_id_by_ref.get(str(row.get("team_b_ref") or ""))
        if not team_a_id or not team_b_id:
            raise RuntimeError(f"Reviewed match row {row.get('source_row')} no longer resolves to imported teams.")
        imported_stage = str(row.get("stage") or "PLAYOFF").upper()
        stage = "ROUND_ROBIN" if imported_stage == "ROUND_ROBIN" else "PLAYOFF"
        game_id = str(uuid.uuid5(namespace, f"game:{row.get('source_row')}:{index}"))
        game = {
            "id": game_id,
            "stage": stage,
            "rr_round_number": None,
            "rr_slot_number": None,
            "playoff_game_code": None,
            "playoff_round": None,
            "team_a_id": team_a_id,
            "team_b_id": team_b_id,
            "score_a": row.get("score_a"),
            "score_b": row.get("score_b"),
        }
        if stage == "ROUND_ROBIN":
            rr_index += 1
            game["rr_round_number"] = rr_index
            game["rr_slot_number"] = 1
        else:
            game["playoff_game_code"] = f"IMPORT-{row.get('source_row') or index}"
            game["playoff_round"] = (
                "Final" if imported_stage == "FINAL" else "Bronze" if imported_stage == "BRONZE" else imported_stage.title()
            )
        if game.get("score_a") is not None and int(game["score_a"]) < 0:
            raise ValueError(f"Reviewed match row {row.get('source_row')} has a negative score.")
        if game.get("score_b") is not None and int(game["score_b"]) < 0:
            raise ValueError(f"Reviewed match row {row.get('source_row')} has a negative score.")
        if game.get("score_a") is not None and game.get("score_b") is not None:
            game.update(finalize_game(game))
        game_rows.append(game)

    podium_rows: list[dict[str, Any]] = []
    for placement_text, team_ref in (preview.get("podium_refs") or {}).items():
        if not team_ref:
            continue
        team_id = team_id_by_ref.get(str(team_ref))
        if not team_id:
            raise RuntimeError("Reviewed podium no longer resolves to an imported team.")
        placement = int(placement_text)
        podium_rows.append(
            {
                "id": str(uuid.uuid5(namespace, f"podium:{placement}")),
                "placement": placement,
                "team_id": team_id,
            }
        )

    try:
        response = supabase.rpc(
            "admin_import_tournament_draw_results_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_expected_draw_updated_at": reviewed_draw_version,
                "p_import_mode": mode,
                "p_new_players": new_players,
                "p_teams": team_rows,
                "p_games": game_rows,
                "p_podium": podium_rows,
            },
        ).execute()
    except Exception as exc:
        if "JUPR_TOURNAMENT_DRAW_STALE" in str(exc):
            raise StaleTournamentAdminStateError(
                "The draw changed while reviewed results were being imported. Reload and preview the exact results again."
            ) from exc
        raise
    rpc_data = getattr(response, "data", None)
    if isinstance(rpc_data, list) and rpc_data and isinstance(rpc_data[0], dict):
        rpc_data = rpc_data[0]
    if not isinstance(rpc_data, dict) or not rpc_data.get("ok"):
        raise RuntimeError("Atomic tournament results import returned no completion result.")

    audit = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type="import_tournament_draw_results_admin",
            entity_type="tournament_event_draw",
            entity_id=str(draw_id),
            before_json={"import_mode": mode, "review_fingerprint": reviewed},
            after_json={
                "source_client": "fastapi/nextjs",
                "source_page": source,
                "review_fingerprint": reviewed,
                "import_mode": mode,
                "team_count": int(rpc_data.get("team_count") or 0),
                "game_count": int(rpc_data.get("game_count") or 0),
                "podium_count": int(rpc_data.get("podium_count") or 0),
                "created_player_count": int(rpc_data.get("created_player_count") or 0),
            },
            source_page=source,
            flagged_for_review=True,
        ),
    )
    warnings = list(preview.get("warnings") or [])
    if audit.warning:
        warnings.append(audit.warning)
    if not audit.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("Results were imported but the required domain completion audit failed.")
    return {
        "ok": True,
        "mode": "tournament_results_import_commit",
        "tournament_id": str(tournament_id),
        "draw_id": str(draw_id),
        "import_mode": mode,
        "review_fingerprint": reviewed,
        "team_count": int(rpc_data.get("team_count") or 0),
        "game_count": int(rpc_data.get("game_count") or 0),
        "podium_count": int(rpc_data.get("podium_count") or 0),
        "created_player_count": int(rpc_data.get("created_player_count") or 0),
        "committed_at": _now_iso(),
        "warnings": warnings,
    }
