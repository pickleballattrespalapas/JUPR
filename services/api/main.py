from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import Client, create_client

from jupr_app.data.load import load_data
from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.context import ServiceContext
from jupr_app.services.leaderboard_service import get_public_leaderboard
from jupr_app.services.match_service import submit_match_batch
from jupr_app.services.public_live_service import (
    is_public_live_session_row,
    public_live_session_detail,
    public_live_sessions_from_rows,
)
from services.api.auth import authenticate_bearer, auth_header
from services.api.middleware import StructuredRequestLoggingMiddleware


DEFAULT_CORS_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://juprleagues.com",
    "https://www.juprleagues.com",
)

# Production data historically uses underscore-style club IDs while public SaaS URLs
# use hyphenated slugs. Keep this explicit until every club has a durable `clubs`
# table row that maps slug -> club_id.
PUBLIC_CLUB_SLUG_TO_ID = {
    "tres-palapas": "tres_palapas",
}

PUBLIC_CLUB_FALLBACKS: dict[str, dict[str, Any]] = {
    "tres-palapas": {
        "id": "tres_palapas",
        "slug": "tres-palapas",
        "name": "Tres Palapas",
        "tagline": None,
        "support_email": None,
        "public_base_url": "https://juprleagues.com/clubs/tres-palapas",
        "logo_url": None,
        "primary_color": None,
        "is_active": True,
    }
}



def get_jupr_env() -> str:
    return os.getenv("JUPR_ENV", "").strip().lower()



def is_staging_env() -> bool:
    return get_jupr_env() == "staging"



def is_next_admin_score_entry_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "").strip().lower() in {"1", "true", "yes"}



def is_api_audit_log_required() -> bool:
    return os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in {"1", "true", "yes"}



def _split_csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return []
    return [value.strip().rstrip("/") for value in raw.split(",") if value.strip()]



def get_cors_allowed_origins() -> list[str]:
    return _split_csv_env("JUPR_ALLOWED_ORIGINS") or list(DEFAULT_CORS_ALLOWED_ORIGINS)



def _log_runtime_guardrails() -> None:
    env = get_jupr_env()
    if not env:
        print(
            "[JUPR API] JUPR_ENV is not set. Local development is allowed, "
            "but deployed API runtimes should set JUPR_ENV=staging or JUPR_ENV=production."
        )
        return

    if env not in {"local", "dev", "development", "staging", "production"}:
        print(
            f"[JUPR API] WARNING: unexpected JUPR_ENV={env!r}. Expected staging or production for deployed runtimes."
        )

    if env == "production" and is_next_admin_score_entry_enabled():
        print(
            "[JUPR API] WARNING: JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY is enabled in production. "
            "Leave it disabled until real production admin auth, club-scoped authorization, and audit review are approved."
        )


app = FastAPI(title="JUPR API", version="0.1.0")
app.add_middleware(StructuredRequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


PUBLIC_LEADERBOARD_ENTRY_FIELDS = {
    "rank",
    "rank_position",
    "club_id",
    "league_name",
    "player_id",
    "player_name",
    "rating",
    "rating_jupr",
    "wins",
    "losses",
    "matches_played",
    "is_active",
    "updated_at",
}

# The list endpoint must stay lightweight. Do not select `state` for every row;
# that JSONB is internal recovery data and can be large. The detail endpoint reads
# `state` for one session and returns only a sanitized public projection.
PUBLIC_LIVE_SESSION_SUMMARY_SELECT = "club_id,session_key,title,status,created_at,updated_at,last_seen_at,expires_at"
PUBLIC_LIVE_SESSION_DETAIL_SELECT = "club_id,session_key,title,status,state,created_at,updated_at,last_seen_at,expires_at"
LIVE_SESSIONS_SETUP_ERROR = (
    "JUPR Live is not fully configured on the API backend. Apply the live_sessions Supabase migrations "
    "and set SUPABASE_SERVICE_ROLE_KEY on the FastAPI deployment so the API can build the sanitized public projection."
)


class MatchBatchRequest(BaseModel):
    matches: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "next_admin_score_entry"



def _get_supabase_credentials() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    key = service_role_key or anon_key

    if not url or not key:
        raise RuntimeError(
            "Supabase config missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "(or SUPABASE_ANON_KEY for read-only local development)."
        )
    return url, key



def _supabase_host_for_diagnostics() -> str:
    raw = os.getenv("SUPABASE_URL", "").strip()
    if not raw:
        return "<missing>"
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return parsed.netloc or parsed.path or "<unparseable>"



def _has_supabase_service_role_key() -> bool:
    return bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip())



def _require_live_sessions_service_role() -> None:
    # The public Live detail API intentionally reads raw durable Streamlit recovery
    # state server-side, then returns a sanitized DTO. Do not grant anonymous
    # Supabase clients direct access to `live_sessions.state`; run this API with
    # service role.
    if not _has_supabase_service_role_key():
        raise HTTPException(status_code=503, detail=LIVE_SESSIONS_SETUP_ERROR)



def get_supabase_client() -> Client:
    url, key = _get_supabase_credentials()
    return create_client(url, key)



def _error_payload_text(exc: Exception) -> str:
    pieces = [str(exc)]
    for attr in ("code", "message", "details", "hint"):
        value = getattr(exc, attr, None)
        if value:
            pieces.append(str(value))
    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if text:
            pieces.append(str(text))
        json_fn = getattr(response, "json", None)
        if callable(json_fn):
            try:
                payload = json_fn()
            except Exception:
                payload = None
            if payload:
                pieces.append(str(payload))
    return " | ".join(pieces).lower()



def _is_missing_table_error(exc: Exception, table_name: str) -> bool:
    detail = _error_payload_text(exc)
    table = table_name.lower()
    return table in detail and (
        "does not exist" in detail
        or "undefined table" in detail
        or "relation" in detail
        or "not found" in detail
        or "could not find" in detail
        or "schema cache" in detail
        or "pgrst205" in detail
    )



def _is_live_sessions_schema_error(exc: Exception) -> bool:
    detail = _error_payload_text(exc)
    if _is_missing_table_error(exc, "live_sessions"):
        return True
    if "live_sessions" not in detail:
        return False
    return any(
        marker in detail
        for marker in (
            "pgrst204",
            "pgrst205",
            "42p01",
            "42501",
            "permission denied",
            "schema cache",
            "could not find",
            "does not exist",
            "undefined table",
            "column",
            "relation",
        )
    )



def _live_sessions_backend_error_detail(exc: Exception) -> str:
    raw_detail = _error_payload_text(exc) or exc.__class__.__name__
    if len(raw_detail) > 500:
        raw_detail = raw_detail[:500] + "..."
    return (
        f"{LIVE_SESSIONS_SETUP_ERROR} Supabase host: {_supabase_host_for_diagnostics()}. "
        f"Backend error: {raw_detail}"
    )



def _raise_live_sessions_setup_error(exc: Exception) -> None:
    raise HTTPException(status_code=503, detail=_live_sessions_backend_error_detail(exc)) from exc



def _raise_live_sessions_backend_error(exc: Exception) -> None:
    raise HTTPException(status_code=503, detail=_live_sessions_backend_error_detail(exc)) from exc



def _club_lookup_candidates(club_slug: str) -> list[str]:
    slug = str(club_slug).strip()
    candidates = [slug]

    explicit_id = PUBLIC_CLUB_SLUG_TO_ID.get(slug)
    if explicit_id:
        candidates.append(explicit_id)

    normalized_id = slug.replace("-", "_")
    if normalized_id != slug:
        candidates.append(normalized_id)

    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique



def _display_name_from_slug(club_slug: str) -> str:
    return str(club_slug).replace("-", " ").replace("_", " ").title()



def _known_public_club_fallback(club_slug: str) -> dict[str, Any] | None:
    slug = str(club_slug or "").strip()
    fallback = PUBLIC_CLUB_FALLBACKS.get(slug)
    return dict(fallback) if fallback else None



def _public_club_payload(club: dict[str, Any], club_slug: str) -> dict[str, str]:
    return {
        "id": str(club.get("id") or club.get("club_id") or club_slug),
        "slug": str(club.get("slug") or club.get("club_slug") or club_slug),
        "name": str(club.get("name") or club.get("club_name") or club.get("display_name") or club_slug),
    }



def _normalize_public_leaderboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        clean = {k: row.get(k) for k in PUBLIC_LEADERBOARD_ENTRY_FIELDS if k in row}
        if clean.get("rank") is None:
            if clean.get("rank_position") is not None:
                clean["rank"] = clean.get("rank_position")
            else:
                clean["rank"] = idx
        normalized.append(clean)
    return normalized



def _build_leaderboard_response(club_slug: str, league_name: str | None) -> dict[str, Any]:
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    rows = get_public_leaderboard(supabase=supabase, club_id=club_id, league_name=league_name)
    return {
        "club": _public_club_payload(club, club_slug),
        "leaderboard": _normalize_public_leaderboard_rows(rows),
    }



def _build_live_sessions_response(club_slug: str, limit: int) -> dict[str, Any]:
    _require_live_sessions_service_role()
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    safe_limit = max(1, min(int(limit or 20), 50))
    supabase = get_supabase_client()
    try:
        rows = (
            supabase.table("live_sessions")
            .select(PUBLIC_LIVE_SESSION_SUMMARY_SELECT)
            .eq("club_id", club_id)
            .order("updated_at", desc=True)
            .limit(safe_limit * 3)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        if _is_live_sessions_schema_error(exc):
            _raise_live_sessions_setup_error(exc)
        _raise_live_sessions_backend_error(exc)
    return {
        "club": _public_club_payload(club, club_slug),
        "sessions": public_live_sessions_from_rows(rows, limit=safe_limit),
    }



def _build_live_session_detail_response(club_slug: str, session_key: str) -> dict[str, Any]:
    _require_live_sessions_service_role()
    clean_session_key = str(session_key or "").strip()
    if not clean_session_key:
        raise HTTPException(status_code=400, detail="session_key is required")

    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    try:
        rows = (
            supabase.table("live_sessions")
            .select(PUBLIC_LIVE_SESSION_DETAIL_SELECT)
            .eq("club_id", club_id)
            .eq("session_key", clean_session_key)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        if _is_live_sessions_schema_error(exc):
            _raise_live_sessions_setup_error(exc)
        _raise_live_sessions_backend_error(exc)

    if not rows or not is_public_live_session_row(rows[0]):
        raise HTTPException(status_code=404, detail="live session not found")

    return {
        "club": _public_club_payload(club, club_slug),
        "session": public_live_session_detail(rows[0]),
    }


@app.on_event("startup")
def startup_checks() -> None:
    _log_runtime_guardrails()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "jupr-api"}


@app.get("/health/live-sessions")
def health_live_sessions() -> dict[str, Any]:
    service_role_configured = _has_supabase_service_role_key()
    host = _supabase_host_for_diagnostics()
    if not service_role_configured:
        return {
            "ok": False,
            "service": "jupr-api",
            "supabase_host": host,
            "service_role_configured": False,
            "detail": LIVE_SESSIONS_SETUP_ERROR,
        }
    try:
        supabase = get_supabase_client()
        rows = (
            supabase.table("live_sessions")
            .select("club_id,session_key,status,updated_at")
            .limit(1)
            .execute()
            .data
            or []
        )
        return {
            "ok": True,
            "service": "jupr-api",
            "supabase_host": host,
            "service_role_configured": True,
            "live_sessions_query_ok": True,
            "sample_count": len(rows),
        }
    except Exception as exc:
        return {
            "ok": False,
            "service": "jupr-api",
            "supabase_host": host,
            "service_role_configured": True,
            "live_sessions_query_ok": False,
            "detail": _error_payload_text(exc) or exc.__class__.__name__,
        }


@app.get("/clubs/{club_slug}")
def get_club(club_slug: str) -> dict[str, Any]:
    slug = str(club_slug).strip()
    if not slug:
        raise HTTPException(status_code=400, detail="club_slug is required")

    known_fallback = _known_public_club_fallback(slug)
    try:
        supabase = get_supabase_client()
    except Exception as exc:
        if known_fallback:
            return known_fallback
        raise HTTPException(status_code=503, detail="Club lookup is unavailable because Supabase is not configured.") from exc

    club_fields = "id,slug,name,tagline,support_email,public_base_url,logo_url,primary_color,is_active"
    club_minimal_fields = "id,slug,name"
    rows: list[dict[str, Any]] = []

    try:
        rows = (
            supabase.table("clubs").select(club_fields).eq("slug", slug).limit(1).execute().data or []
        )
        if not rows:
            for club_id in _club_lookup_candidates(slug):
                rows = supabase.table("clubs").select(club_fields).eq("id", club_id).limit(1).execute().data or []
                if rows:
                    break
    except Exception:
        try:
            rows = (
                supabase.table("clubs").select(club_minimal_fields).eq("slug", slug).limit(1).execute().data or []
            )
            if not rows:
                for club_id in _club_lookup_candidates(slug):
                    rows = (
                        supabase.table("clubs")
                        .select(club_minimal_fields)
                        .eq("id", club_id)
                        .limit(1)
                        .execute()
                        .data
                        or []
                    )
                    if rows:
                        break
        except Exception:
            rows = []

    if rows:
        row = rows[0] or {}
        return {
            "id": row.get("id"),
            "slug": row.get("slug") or slug,
            "name": row.get("name") or _display_name_from_slug(slug),
            "tagline": row.get("tagline"),
            "support_email": row.get("support_email"),
            "public_base_url": row.get("public_base_url"),
            "logo_url": row.get("logo_url"),
            "primary_color": row.get("primary_color"),
            "is_active": row.get("is_active", True),
        }

    for club_id in _club_lookup_candidates(slug):
        try:
            fallback = (
                supabase.table("players")
                .select("club_id")
                .eq("club_id", club_id)
                .limit(1)
                .execute()
                .data
                or []
            )
        except Exception:
            fallback = []
        if fallback:
            resolved_id = str(fallback[0].get("club_id") or club_id)
            return {
                "id": resolved_id,
                "slug": slug,
                "name": _display_name_from_slug(slug),
                "tagline": None,
                "support_email": None,
                "public_base_url": None,
                "logo_url": None,
                "primary_color": None,
                "is_active": True,
            }

    if known_fallback:
        return known_fallback

    raise HTTPException(status_code=404, detail="club not found")


@app.get("/clubs/{club_slug}/leaderboards")
def get_club_leaderboard(club_slug: str, league_name: str | None = Query(default=None)) -> dict[str, Any]:
    return _build_leaderboard_response(club_slug, league_name)


@app.get("/clubs/{club_slug}/leaderboards/public")
def get_club_leaderboard_compat(club_slug: str, league_name: str | None = Query(default=None)) -> dict[str, Any]:
    # Temporary compatibility alias for Next.js clients still calling /leaderboards/public.
    # Remove this route after the web app fully migrates to /leaderboards.
    return _build_leaderboard_response(club_slug, league_name)


@app.get("/clubs/{club_slug}/live-sessions")
def get_club_live_sessions(club_slug: str, limit: int = Query(default=20, ge=1, le=50)) -> dict[str, Any]:
    return _build_live_sessions_response(club_slug, limit)


@app.get("/clubs/{club_slug}/live-sessions/{session_key}")
def get_club_live_session(club_slug: str, session_key: str) -> dict[str, Any]:
    return _build_live_session_detail_response(club_slug, session_key)


@app.post("/admin/clubs/{club_id}/matches/batch")
def submit_admin_match_batch(
    club_id: str,
    payload: MatchBatchRequest,
    authorization: str | None = auth_header(),
) -> dict[str, Any]:
    if not is_next_admin_score_entry_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Next admin score entry is disabled. Use Streamlit admin until Supabase JWT role auth is implemented."
            ),
        )

    user = authenticate_bearer(authorization)

    supabase = get_supabase_client()
    role_resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied_payload = build_activity_payload(
            club_id=str(club_id),
            actor_email=user.email,
            actor_role=role_resolution.role,
            action_type="submit_match_batch_denied",
            entity_type="matches",
            entity_id="batch",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"},
            source_page=payload.source,
            flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    (
        df_players_all,
        _df_players_active,
        df_leagues,
        _df_matches,
        df_meta,
        _df_badges,
        _df_player_badges,
        name_to_id,
        _id_to_name,
        _schema_degraded,
        _schema_degraded_reason,
    ) = load_data(supabase, club_id)

    service_ctx = ServiceContext(
        supabase=supabase,
        club_id=str(club_id),
        source=payload.source,
        actor_email=user.email,
        actor_role=role_resolution.role,
    )
    result = submit_match_batch(
        service_ctx,
        payload.matches,
        name_to_id=name_to_id,
        df_players_all=df_players_all,
        df_leagues=df_leagues,
        df_meta=df_meta,
    )
    if not result.ok:
        raise HTTPException(status_code=400, detail="; ".join(result.errors) or "Unable to submit match batch")

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=user.email,
        actor_role=role_resolution.role,
        action_type="submit_match_batch",
        entity_type="matches",
        entity_id="batch",
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": payload.source,
            "match_count": len(payload.matches),
            "result_summary": result.data if isinstance(result.data, dict) else {"ok": True},
        },
        source_page=payload.source,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    if not audit_write.ok and is_api_audit_log_required():
        raise HTTPException(status_code=500, detail="audit log write required but unavailable")

    return {
        "ok": True,
        "auth_mode": "supabase_jwt",
        "required_permission": PERMISSION_ENTER_SCORES,
        "result": result.data,
    }
