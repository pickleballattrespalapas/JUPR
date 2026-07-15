from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.player_ops import safe_add_player

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
DEFAULT_NEW_PLAYER_JUPR = 3.5


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_player_editor_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR")


def is_api_audit_log_required() -> bool:
    return _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _jupr_to_elo(value: Any, *, field_name: str) -> float:
    rating = _safe_float(value)
    if rating is None:
        raise ValueError(f"{field_name} is required.")
    if rating < 1.0 or rating > 7.0:
        raise ValueError(f"{field_name} must be between 1.0 and 7.0.")
    return float(rating) * 400.0


def _elo_to_jupr(value: Any) -> float | None:
    rating = _safe_float(value)
    return None if rating is None else float(rating) / 400.0


def _player_payload(row: dict[str, Any]) -> dict[str, Any]:
    pid = _safe_int(row.get("id"))
    return {
        "id": int(pid or 0),
        "club_id": str(row.get("club_id") or ""),
        "name": _clean_text(row.get("name"), limit=160),
        "rating": row.get("rating"),
        "rating_jupr": _elo_to_jupr(row.get("rating")),
        "starting_rating": row.get("starting_rating"),
        "starting_jupr": _elo_to_jupr(row.get("starting_rating")),
        "wins": row.get("wins"),
        "losses": row.get("losses"),
        "matches_played": row.get("matches_played"),
        "active": bool(row.get("active", row.get("is_active", True))),
        "inactive_at": row.get("inactive_at"),
        "last_game_at": row.get("last_game_at"),
    }


def _league_rating_payload(row: dict[str, Any]) -> dict[str, Any]:
    rid = _safe_int(row.get("id"))
    return {
        "id": int(rid or 0),
        "league_name": _clean_text(row.get("league_name"), limit=120),
        "rating": row.get("rating"),
        "rating_jupr": _elo_to_jupr(row.get("rating")),
        "starting_rating": row.get("starting_rating"),
        "starting_jupr": _elo_to_jupr(row.get("starting_rating")),
        "wins": row.get("wins"),
        "losses": row.get("losses"),
        "matches_played": row.get("matches_played"),
        "is_active": bool(row.get("is_active", True)),
        "inactive_at": row.get("inactive_at"),
    }


def _fetch_players(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    rows = _safe_rows(
        supabase.table("players")
        .select("id,club_id,name,rating,starting_rating,wins,losses,matches_played,active,inactive_at,last_game_at")
        .eq("club_id", str(club_id))
        .order("name", desc=False)
        .execute()
    )
    return [_player_payload(row) for row in rows if _safe_int(row.get("id")) is not None]


def _fetch_player(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("players")
        .select("id,club_id,name,rating,starting_rating,wins,losses,matches_played,active,inactive_at,last_game_at")
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    return _player_payload(rows[0]) if rows else None


def _fetch_league_ratings(supabase: Any, *, club_id: str, player_id: int) -> list[dict[str, Any]]:
    rows = _safe_rows(
        supabase.table("league_ratings")
        .select("id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at")
        .eq("club_id", str(club_id))
        .eq("player_id", int(player_id))
        .order("league_name", desc=False)
        .execute()
    )
    return [_league_rating_payload(row) for row in rows if _safe_int(row.get("id")) is not None]


def _match_reference_counts(supabase: Any, *, club_id: str, player_id: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
        try:
            rows = _safe_rows(
                supabase.table("matches")
                .select("id")
                .eq("club_id", str(club_id))
                .eq(column, int(player_id))
                .execute()
            )
        except Exception:
            rows = []
        counts[column] = len(rows)
    counts["total"] = sum(counts.values())
    return counts


def build_admin_player_editor_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "players_endpoint": None,
            "player_detail_endpoint": None,
            "social_identities_endpoint": None,
            "warnings": ["Next Player Editor is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR on FastAPI for a closed-club pilot."],
        }
    player_count = None
    if supabase is not None:
        try:
            player_count = len(_fetch_players(supabase, club_id=str(club_id)))
        except Exception:
            player_count = None
    return {
        "enabled": True,
        "status": "ready_for_player_editor_social_identity_pilot",
        "players_endpoint": "/admin/clubs/{club_id}/players/editor/players",
        "player_detail_endpoint": "/admin/clubs/{club_id}/players/editor/players/{player_id}",
        "social_identities_endpoint": "/admin/clubs/{club_id}/players/editor/social-identities",
        "player_count": player_count,
        "warnings": ["Player create/update, league-rating edits, and social identity linking are enabled. Merge remains Streamlit-only until replay/correction safety is proven in Next."],
    }


def list_admin_player_editor_players(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    players = _fetch_players(supabase, club_id=str(club_id))
    return {"ok": True, "mode": "player_editor_list", "players": players, "count": len(players)}


def get_admin_player_editor_detail(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    player = _fetch_player(supabase, club_id=str(club_id), player_id=int(player_id))
    if player is None:
        raise ValueError("player not found")
    league_ratings = _fetch_league_ratings(supabase, club_id=str(club_id), player_id=int(player_id))
    return {
        "ok": True,
        "mode": "player_editor_detail",
        "player": player,
        "league_ratings": league_ratings,
        "match_reference_counts": _match_reference_counts(supabase, club_id=str(club_id), player_id=int(player_id)),
    }


def create_admin_player_editor_player(
    supabase: Any,
    *,
    club_id: str,
    name: str,
    starting_jupr: Any = DEFAULT_NEW_PLAYER_JUPR,
    actor_email: str,
    actor_role: str,
    source: str = "next_player_editor",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    clean_name = _clean_text(name, limit=160)
    if not clean_name:
        raise ValueError("Player name is required.")
    rating = _safe_float(starting_jupr)
    if rating is None:
        rating = DEFAULT_NEW_PLAYER_JUPR
    if rating < 1.0 or rating > 7.0:
        raise ValueError("Starting JUPR must be between 1.0 and 7.0.")
    ok, error = safe_add_player(supabase=supabase, club_id=str(club_id), name=clean_name, rating_jupr=float(rating))
    if not ok:
        raise ValueError(error or "Unable to create player.")
    players = [player for player in _fetch_players(supabase, club_id=str(club_id)) if player.get("name") == clean_name]
    player = players[0] if players else None
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="create_player_editor_player",
        entity_type="player",
        entity_id=str(player.get("id") if player else clean_name),
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "player": player or {"name": clean_name}},
        source_page=source,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "player_editor_create", "player": player, "warnings": warnings}


def update_admin_player_editor_player(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str = "next_player_editor",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    before = _fetch_player(supabase, club_id=str(club_id), player_id=int(player_id))
    if before is None:
        raise ValueError("player not found")
    update_payload: dict[str, Any] = {}
    if "name" in patch:
        name = _clean_text(patch.get("name"), limit=160)
        if not name:
            raise ValueError("Player name is required.")
        update_payload["name"] = name
    if "rating_jupr" in patch:
        update_payload["rating"] = _jupr_to_elo(patch.get("rating_jupr"), field_name="Overall JUPR")
    if "starting_jupr" in patch:
        update_payload["starting_rating"] = _jupr_to_elo(patch.get("starting_jupr"), field_name="Starting JUPR")
    if "active" in patch:
        next_active = bool(patch.get("active"))
        update_payload["active"] = next_active
        update_payload["inactive_at"] = None if next_active else (before.get("inactive_at") or datetime.now(timezone.utc).isoformat())
    if not update_payload:
        raise ValueError("No supported player fields were provided.")
    updated_rows = _safe_rows(
        supabase.table("players")
        .update(update_payload)
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .execute()
    )
    after = _player_payload(updated_rows[0]) if updated_rows else _fetch_player(supabase, club_id=str(club_id), player_id=int(player_id))
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_player_editor_player",
        entity_type="player",
        entity_id=str(int(player_id)),
        before_json={"player": before},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": update_payload, "player": after},
        source_page=source,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "player_editor_update", "player": after, "warnings": warnings}
