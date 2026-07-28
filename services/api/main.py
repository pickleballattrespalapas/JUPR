from __future__ import annotations

import hashlib
import os
import re
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import Client, create_client

from jupr_app.data.load import load_data
from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.direct_match_entry_service import (
    DirectMatchConflictError,
    DirectMatchRecoveryRequiredError,
    submit_atomic_direct_matches,
)
from jupr_app.services.leaderboard_service import LeaderboardDataUnavailable, build_public_leaderboard
from jupr_app.services.public_live_service import is_public_live_session_row, public_live_session_detail, public_live_sessions_from_rows
from jupr_app.services.public_live_operation_service import (
    PublicLiveConflictError,
    PublicLiveRateLimitError,
    PublicLiveRecoveryRequiredError,
)
from jupr_app.services.public_live_write_service import (
    PublicLiveSessionError,
    advance_public_live_session,
    build_public_live_export,
    complete_public_live_session,
    create_public_live_session,
    substitute_public_live_participant,
    update_public_live_scores,
)
from jupr_app.services.public_player_service import build_public_player_directory, get_public_match_detail, get_public_matches, get_public_player_profile
from scripts.deployment_verifier import (
    PRODUCTION_FEATURE_FLAGS,
    feature_flag_fingerprint,
)
from scripts.staging_write_waves import ALL_STAGING_WRITE_FLAGS
from services.api.auth import (
    authenticate_bearer,
    auth_header,
    jwt_verification_configured,
    jwt_verification_mode,
    jwt_verification_project_ref,
)
from services.api.middleware import StagingWriteWaveMiddleware, StructuredRequestLoggingMiddleware
from services.api.admin_team_league_routes import install_admin_team_league_routes
from services.api.admin_tournament_team_competition_routes import (
    install_admin_tournament_team_competition_routes,
)
from services.api.public_badge_codex_routes import install_public_badge_codex_routes
from services.api.public_challenge_ladder_routes import install_public_challenge_ladder_routes
from services.api.public_league_results_routes import install_public_league_results_routes
from services.api.public_match_explorer_routes import install_public_match_explorer_routes
from services.api.public_team_league_routes import install_public_team_league_routes
from services.api.public_tournament_team_routes import (
    install_public_tournament_team_routes,
)
from services.api.public_weekly_recap_routes import install_public_weekly_recap_routes

DEFAULT_CORS_ALLOWED_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000", "https://juprleagues.com", "https://www.juprleagues.com")
PUBLIC_CLUB_SLUG_TO_ID = {"tres-palapas": "tres_palapas"}
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
PUBLIC_LEADERBOARD_ENTRY_FIELDS = {
    "rank",
    "rank_position",
    "club_id",
    "league_name",
    "player_id",
    "player_name",
    "rating",
    "rating_jupr",
    "starting_rating",
    "starting_rating_jupr",
    "rating_gain_jupr",
    "gap_jupr",
    "wins",
    "losses",
    "matches_played",
    "win_pct",
    "is_active",
    "qualified",
    "min_games",
    "badges",
    "badge_count",
    "updated_at",
}
PUBLIC_LEADERBOARD_BADGE_FIELDS = {"badge_id", "name", "prestige", "category", "icon_key", "rarity", "earned_at"}
PUBLIC_LIVE_SESSION_SUMMARY_SELECT = "club_id,session_key,title,status,state,version,created_at,updated_at,last_seen_at,expires_at,completed_at"
PUBLIC_LIVE_SESSION_DETAIL_SELECT = "club_id,session_key,title,status,state,version,created_at,updated_at,last_seen_at,expires_at,completed_at"
PUBLIC_LIVE_SESSION_LEGACY_SELECT = "club_id,session_key,title,status,state,created_at,updated_at,last_seen_at,expires_at"
LIVE_SESSIONS_SETUP_ERROR = "JUPR Live is not fully configured on the API backend. Apply the live_sessions Supabase migrations and set SUPABASE_SERVICE_ROLE_KEY on the FastAPI deployment so the API can build the sanitized public projection."
PUBLIC_LIVE_WRITES_DISABLED_ERROR = "Public JUPR Live writes are not enabled in this environment. Use the Streamlit fallback or a shared view-only session."
SCORE_ENTRY_SETUP_ERROR = "Next score entry is not write-ready. Enable the backend flag and configure SUPABASE_SERVICE_ROLE_KEY on FastAPI; otherwise use Match Uploader or the Streamlit fallback."


def get_jupr_env() -> str:
    return os.getenv("JUPR_ENV", "").strip().lower()


def _canonical_https_origin(raw: str | None) -> str | None:
    """Return a credential-free HTTPS origin, or None for non-origin input."""
    try:
        parsed = urlparse(str(raw or "").strip())
        port = parsed.port
    except (TypeError, ValueError):
        return None
    host = (parsed.hostname or "").strip().lower()
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return None
    return f"https://{host}"


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


def get_cors_allowed_origin_regex() -> str | None:
    value = os.getenv("JUPR_ALLOWED_ORIGIN_REGEX", "").strip()
    if value:
        re.compile(value)
    return value or None


def _log_runtime_guardrails() -> None:
    env = get_jupr_env()
    if not env:
        print("[JUPR API] JUPR_ENV is not set. Local development is allowed, but deployed APIs should set JUPR_ENV.")
    elif env not in {"local", "dev", "development", "staging", "production"}:
        print(f"[JUPR API] WARNING: unexpected JUPR_ENV={env!r}.")
    if env == "production" and is_next_admin_score_entry_enabled():
        print("[JUPR API] WARNING: JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY is enabled in production.")


app = FastAPI(title="JUPR API", version="0.1.0")
app.add_middleware(StagingWriteWaveMiddleware)
app.add_middleware(StructuredRequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allowed_origins(),
    allow_origin_regex=get_cors_allowed_origin_regex(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)


class MatchBatchRequest(BaseModel):
    matches: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "next_admin_score_entry"
    idempotency_key: str = Field(
        min_length=8,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$",
    )


class PublicLiveSessionCreateRequest(BaseModel):
    event_name: str = Field(default="JUPR Live Round Robin", max_length=160)
    event_type: str = Field(default="round_robin", max_length=32)
    participant_names: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(min_length=4, max_length=20)
    live_mode: str = Field(default="quick", max_length=32)
    total_rounds: int = Field(default=3, ge=1, le=20)
    court_sizes: list[Annotated[int, Field(ge=4, le=5)]] = Field(default_factory=list, max_length=5)
    host_name: str | None = Field(default=None, max_length=160)
    skill_levels: list[Annotated[str, Field(max_length=16)]] = Field(default_factory=list, max_length=7)
    participant_player_ids: dict[Annotated[str, Field(max_length=80)], int] = Field(default_factory=dict, max_length=20)
    idempotency_key: str = Field(min_length=8, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$")


class PublicLiveScorePayload(BaseModel):
    match_id: str = Field(min_length=1, max_length=160)
    score_a: int | None = None
    score_b: int | None = None


class PublicLiveScoreUpdateRequest(BaseModel):
    edit_token: str = Field(min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$")
    scores: list[PublicLiveScorePayload] = Field(min_length=1, max_length=500)


class PublicLiveMutationRequest(BaseModel):
    edit_token: str = Field(min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$")


class PublicLiveSubstitutionRequest(PublicLiveMutationRequest):
    scope: str = Field(default="round", pattern=r"^(round|game)$")
    round_number: int = Field(ge=1)
    original_participant_id: str = Field(min_length=1, max_length=160)
    substitute_name: str = Field(min_length=1, max_length=80)
    substitute_player_id: int | None = None
    match_id: str | None = Field(default=None, max_length=160)
    note: str | None = Field(default=None, max_length=300)


def _get_supabase_credentials() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    key = service_role_key or anon_key
    if not url or not key:
        raise RuntimeError("Supabase config missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY.")
    return url, key


def get_supabase_client() -> Client:
    url, key = _get_supabase_credentials()
    return create_client(url, key)


def _supabase_host_for_diagnostics() -> str:
    raw = os.getenv("SUPABASE_URL", "").strip()
    if not raw:
        return "<missing>"
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return parsed.netloc or parsed.path or "<unparseable>"


def _has_supabase_service_role_key() -> bool:
    return bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip())


def _require_live_sessions_service_role() -> None:
    if not _has_supabase_service_role_key():
        raise HTTPException(status_code=503, detail=LIVE_SESSIONS_SETUP_ERROR)


def is_public_live_write_enabled() -> bool:
    enabled = os.getenv("JUPR_ENABLE_PUBLIC_LIVE_WRITES", "").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return False
    environment = get_jupr_env()
    if environment == "production":
        return os.getenv("JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION", "").strip().lower() in {"1", "true", "yes"}
    return environment in {"staging", "local", "test", "development", "dev"}


def _public_live_secrets_ready() -> bool:
    token_secret = os.getenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "").strip()
    rate_secret = (
        os.getenv("JUPR_PUBLIC_LIVE_RATE_LIMIT_SECRET", "").strip()
        or token_secret
    )
    return len(token_secret) >= 32 and len(rate_secret) >= 32


def is_public_live_write_ready() -> bool:
    return bool(
        is_public_live_write_enabled()
        and _has_supabase_service_role_key()
        and _public_live_secrets_ready()
    )


def _require_public_live_writes() -> None:
    if not is_public_live_write_enabled():
        raise HTTPException(status_code=403, detail=PUBLIC_LIVE_WRITES_DISABLED_ERROR)
    if not _public_live_secrets_ready():
        raise HTTPException(
            status_code=503,
            detail="Public JUPR Live secrets are unavailable; no write was attempted. Use the Streamlit fallback.",
        )


def _public_live_requester_hash(request: Request) -> str:
    """Build a private rate-limit scope that survives the Vercel-to-Fly proxy hop.

    Fly's client address is the Vercel egress for proxied requests, so it cannot
    be used alone without grouping every visitor together. Keeping it in the
    scope while also including the forwarded visitor tail preserves normal
    per-visitor fairness. The database's per-club ceiling remains authoritative
    if an untrusted direct caller spoofs forwarding metadata.
    """

    peer_address = str(request.client.host if request.client else "unknown")
    fly_address = str(
        request.headers.get("fly-client-ip")
        or request.headers.get("x-real-ip")
        or peer_address
    ).strip()[:128]
    forwarded = str(
        request.headers.get("x-vercel-forwarded-for")
        or request.headers.get("x-forwarded-for")
        or ""
    )
    addresses = [value.strip() for value in forwarded.split(",") if value.strip()]
    visitor_address = (addresses[-1] if addresses else fly_address)[:128]
    address_scope = f"{fly_address}\x1f{visitor_address}"
    secret = (
        os.getenv("JUPR_PUBLIC_LIVE_RATE_LIMIT_SECRET", "").strip()
        or os.getenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "").strip()
    )
    if len(secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="Public JUPR Live anti-abuse configuration is unavailable; no write was attempted.",
        )
    return hashlib.sha256(f"{secret}\x1f{address_scope}".encode("utf-8")).hexdigest()


def _raise_public_live_write_error(exc: Exception) -> None:
    if isinstance(exc, PublicLiveRateLimitError):
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    if isinstance(exc, PublicLiveConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, (PublicLiveSessionError, ValueError)):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, PublicLiveRecoveryRequiredError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _raise_live_sessions_backend_error(exc)


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
    return table in detail and any(marker in detail for marker in ("does not exist", "undefined table", "relation", "not found", "could not find", "schema cache", "pgrst205"))


def _is_live_sessions_schema_error(exc: Exception) -> bool:
    detail = _error_payload_text(exc)
    if _is_missing_table_error(exc, "live_sessions"):
        return True
    if "live_sessions" not in detail:
        return False
    return any(marker in detail for marker in ("pgrst204", "pgrst205", "42p01", "42501", "permission denied", "schema cache", "could not find", "does not exist", "undefined table", "column", "relation"))


def _is_public_live_durability_schema_error(exc: Exception) -> bool:
    detail = _error_payload_text(exc)
    return "live_sessions" in detail and any(
        marker in detail
        for marker in (
            "'version'", '"version"', "live_sessions.version",
            "'edit_token_hash'", '"edit_token_hash"', "live_sessions.edit_token_hash",
            "'completed_at'", '"completed_at"', "live_sessions.completed_at",
        )
    ) and any(marker in detail for marker in ("column", "schema cache", "could not find", "pgrst204"))


def _live_sessions_backend_error_detail(exc: Exception) -> str:
    raw_detail = _error_payload_text(exc) or exc.__class__.__name__
    if len(raw_detail) > 500:
        raw_detail = raw_detail[:500] + "..."
    return f"{LIVE_SESSIONS_SETUP_ERROR} Supabase host: {_supabase_host_for_diagnostics()}. Backend error: {raw_detail}"


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
    fallback = PUBLIC_CLUB_FALLBACKS.get(str(club_slug or "").strip())
    return dict(fallback) if fallback else None


def _public_club_payload(club: dict[str, Any], club_slug: str) -> dict[str, str]:
    return {"id": str(club.get("id") or club.get("club_id") or club_slug), "slug": str(club.get("slug") or club.get("club_slug") or club_slug), "name": str(club.get("name") or club.get("club_name") or club.get("display_name") or club_slug)}


def _normalize_public_leaderboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        clean = {key: row.get(key) for key in PUBLIC_LEADERBOARD_ENTRY_FIELDS if key in row}
        clean["badges"] = [
            {key: badge.get(key) for key in PUBLIC_LEADERBOARD_BADGE_FIELDS if key in badge}
            for badge in (row.get("badges") or [])
            if isinstance(badge, dict)
        ][:3]
        if clean.get("rank") is None:
            clean["rank"] = clean.get("rank_position") if clean.get("rank_position") is not None else idx
        normalized.append(clean)
    return normalized


def _normalize_public_leaderboard_projection(payload: dict[str, Any]) -> dict[str, Any]:
    scopes = [
        {
            "name": str(scope.get("name") or ""),
            "label": str(scope.get("label") or scope.get("name") or ""),
            "min_games": max(0, int(scope.get("min_games") or 0)),
        }
        for scope in (payload.get("scopes") or [])
        if isinstance(scope, dict) and str(scope.get("name") or "").strip()
    ]
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    pagination = payload.get("pagination") if isinstance(payload.get("pagination"), dict) else {}
    snapshot_rows = _normalize_public_leaderboard_rows([payload["snapshot"]]) if isinstance(payload.get("snapshot"), dict) else []
    highlights = payload.get("highlights") if isinstance(payload.get("highlights"), dict) else {}
    return {
        "scopes": scopes,
        "selected_scope": str(payload.get("selected_scope") or "OVERALL"),
        "scope": {
            "name": str(scope.get("name") or payload.get("selected_scope") or "OVERALL"),
            "label": str(scope.get("label") or scope.get("name") or payload.get("selected_scope") or "Overall"),
            "min_games": max(0, int(scope.get("min_games") or 0)),
        },
        "filters": {
            "status": str(filters.get("status") or "active"),
            "search": str(filters.get("search") or ""),
            "sort": str(filters.get("sort") or "rank"),
        },
        "summary": {
            "ranked_players": max(0, int(summary.get("ranked_players") or 0)),
            "active_players": max(0, int(summary.get("active_players") or 0)),
            "inactive_players": max(0, int(summary.get("inactive_players") or 0)),
            "leaderboard_scopes": max(0, int(summary.get("leaderboard_scopes") or len(scopes))),
            "filtered_players": max(0, int(summary.get("filtered_players") or 0)),
        },
        "leaderboard": _normalize_public_leaderboard_rows(payload.get("leaderboard") or []),
        "snapshot": snapshot_rows[0] if snapshot_rows else None,
        "highlights": {
            key: _normalize_public_leaderboard_rows(highlights.get(key) or [])
            for key in ("highest_rating", "most_improved", "best_win_pct", "most_wins")
        },
        "pagination": {
            "total": max(0, int(pagination.get("total") or 0)),
            "offset": max(0, int(pagination.get("offset") or 0)),
            "limit": max(1, min(int(pagination.get("limit") or 50), 100)),
            "has_more": bool(pagination.get("has_more")),
        },
    }


def _build_leaderboard_response(
    club_slug: str,
    league_name: str | None,
    *,
    status: str = "active",
    search: str | None = None,
    sort: str = "rank",
    player_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    try:
        projection = build_public_leaderboard(
            supabase,
            club_id=club_id,
            league_name=league_name,
            status=status,
            search=search,
            sort=sort,
            player_id=player_id,
            limit=limit,
            offset=offset,
        )
    except LeaderboardDataUnavailable as exc:
        raise HTTPException(status_code=503, detail="Leaderboard data is temporarily unavailable.") from exc
    return {"club": _public_club_payload(club, club_slug), **_normalize_public_leaderboard_projection(projection)}


def _build_live_sessions_response(club_slug: str, limit: int) -> dict[str, Any]:
    _require_live_sessions_service_role()
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    safe_limit = max(1, min(int(limit or 20), 50))
    supabase = get_supabase_client()
    durability_ready = True
    try:
        rows = supabase.table("live_sessions").select(PUBLIC_LIVE_SESSION_SUMMARY_SELECT).eq("club_id", club_id).order("updated_at", desc=True).limit(safe_limit * 3).execute().data or []
    except Exception as exc:
        if _is_public_live_durability_schema_error(exc):
            durability_ready = False
            try:
                rows = supabase.table("live_sessions").select(PUBLIC_LIVE_SESSION_LEGACY_SELECT).eq("club_id", club_id).order("updated_at", desc=True).limit(safe_limit * 3).execute().data or []
            except Exception as legacy_exc:
                if _is_live_sessions_schema_error(legacy_exc):
                    _raise_live_sessions_setup_error(legacy_exc)
                _raise_live_sessions_backend_error(legacy_exc)
        elif _is_live_sessions_schema_error(exc):
            _raise_live_sessions_setup_error(exc)
        else:
            _raise_live_sessions_backend_error(exc)
    return {
        "club": _public_club_payload(club, club_slug),
        "sessions": public_live_sessions_from_rows(rows, limit=safe_limit),
        "write_enabled": bool(durability_ready and is_public_live_write_ready()),
        "write_fallback_url": os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "https://juprtrespalapas.streamlit.app").strip(),
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
        rows = supabase.table("live_sessions").select(PUBLIC_LIVE_SESSION_DETAIL_SELECT).eq("club_id", club_id).eq("session_key", clean_session_key).limit(1).execute().data or []
    except Exception as exc:
        if _is_public_live_durability_schema_error(exc):
            try:
                rows = supabase.table("live_sessions").select(PUBLIC_LIVE_SESSION_LEGACY_SELECT).eq("club_id", club_id).eq("session_key", clean_session_key).limit(1).execute().data or []
            except Exception as legacy_exc:
                if _is_live_sessions_schema_error(legacy_exc):
                    _raise_live_sessions_setup_error(legacy_exc)
                _raise_live_sessions_backend_error(legacy_exc)
        elif _is_live_sessions_schema_error(exc):
            _raise_live_sessions_setup_error(exc)
        else:
            _raise_live_sessions_backend_error(exc)
    if not rows or not is_public_live_session_row(rows[0]):
        raise HTTPException(status_code=404, detail="live session not found")
    return {"club": _public_club_payload(club, club_slug), "session": public_live_session_detail(rows[0])}


def _score_entry_player_ids(matches: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for match in matches or []:
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            try:
                pid = int(match.get(key))
            except Exception:
                continue
            if pid not in ids:
                ids.append(pid)
    return ids


def _validate_score_entry_match(matches: list[dict[str, Any]]) -> None:
    if len(matches or []) != 1:
        raise HTTPException(status_code=400, detail="Score Entry accepts exactly one match. Use Match Uploader for batches.")
    match = matches[0]

    def whole_number(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("boolean is not a score or player id")
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError("fractional value")
        return int(numeric)

    try:
        players = [whole_number(match[key]) for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        score_t1 = whole_number(match["score_t1"])
        score_t2 = whole_number(match["score_t2"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Select four players and enter both whole-number scores.") from exc
    if any(player_id <= 0 for player_id in players) or len(set(players)) != 4:
        raise HTTPException(status_code=400, detail="Select four different players.")
    if score_t1 < 0 or score_t2 < 0 or score_t1 + score_t2 <= 0:
        raise HTTPException(status_code=400, detail="Scores must be non-negative and the match score must be non-zero.")
    if score_t1 == score_t2:
        raise HTTPException(status_code=400, detail="Match scores cannot be tied.")


def _normalize_score_entry_match(
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Allowlist the score-entry contract before fingerprinting or auditing."""

    match = dict(matches[0])

    def text_value(key: str, default: str = "", limit: int = 120) -> str:
        return (
            str(match.get(key) or default)
            .replace("<", "")
            .replace(">", "")
            .strip()[:limit]
        )

    return [
        {
            "date": text_value("date", limit=80) or None,
            "league": text_value("league", "Open") or "Open",
            "week_tag": text_value("week_tag", limit=80),
            "match_type": text_value(
                "match_type",
                "Web Score Entry",
                limit=80,
            )
            or "Web Score Entry",
            "rating_scope": text_value("rating_scope", limit=40),
            "t1_p1": int(float(match["t1_p1"])),
            "t1_p2": int(float(match["t1_p2"])),
            "t2_p1": int(float(match["t2_p1"])),
            "t2_p2": int(float(match["t2_p2"])),
            "score_t1": int(float(match["score_t1"])),
            "score_t2": int(float(match["score_t2"])),
        }
    ]


def _fetch_score_entry_players(supabase, *, club_id: str, player_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not player_ids:
        return {}
    try:
        rows = supabase.table("players").select("id,name,rating,wins,losses,matches_played").eq("club_id", club_id).execute().data or []
    except Exception:
        return {}
    result: dict[int, dict[str, Any]] = {}
    allowed = {int(pid) for pid in player_ids}
    for row in rows:
        try:
            pid = int(row.get("id"))
        except Exception:
            continue
        if pid in allowed:
            result[pid] = dict(row)
    return result


def _score_entry_feedback(*, before: dict[int, dict[str, Any]], after: dict[int, dict[str, Any]], player_ids: list[int], latest_match_id: Any = None) -> dict[str, Any]:
    affected = []
    ratings_updated = False
    for pid in player_ids:
        b = before.get(pid, {})
        a = after.get(pid, {})
        rb = b.get("rating")
        ra = a.get("rating")
        try:
            delta = None if rb is None or ra is None else float(ra) - float(rb)
        except Exception:
            delta = None
        if delta not in (None, 0):
            ratings_updated = True
        affected.append({"id": pid, "name": a.get("name") or b.get("name") or f"Player {pid}", "rating_before": rb, "rating_after": ra, "rating_delta": delta, "matches_played_before": b.get("matches_played"), "matches_played_after": a.get("matches_played")})
    return {"ratings_updated": ratings_updated, "affected_players": affected, "latest_match_id": latest_match_id}


def _latest_score_entry_match_id(supabase, *, club_id: str, matches: list[dict[str, Any]]) -> Any:
    try:
        rows = supabase.table("matches").select("id").eq("club_id", club_id).order("date", desc=True).limit(1).execute().data or []
        return rows[0].get("id") if rows else None
    except Exception:
        return None


@app.on_event("startup")
def startup_checks() -> None:
    _log_runtime_guardrails()


@app.get("/health")
def health() -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": True, "service": "jupr-api"}
    environment = get_jupr_env()
    if environment in {"staging", "production"}:
        supabase_host = (urlparse(os.getenv("SUPABASE_URL", "")).hostname or "").lower()
        controlled_write_flags = {
            name: os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}
            for name in ALL_STAGING_WRITE_FLAGS
        }
        feature_flags = {
            name: os.getenv(name, "").strip().lower()
            in {"1", "true", "yes", "y", "on"}
            for name in PRODUCTION_FEATURE_FLAGS
        }
        write_wave = os.getenv("JUPR_STAGING_WRITE_WAVE", "").strip() or None
        payload.update(
            {
                "environment": environment,
                "git_commit_sha": (
                    os.getenv("JUPR_IMAGE_BUILD_GIT_SHA", "").strip().lower()
                    or os.getenv("JUPR_DEPLOYMENT_GIT_SHA", "").strip().lower()
                    or None
                ),
                "image_build_git_sha": os.getenv(
                    "JUPR_IMAGE_BUILD_GIT_SHA", ""
                ).strip().lower()
                or None,
                "fly_app_name": os.getenv("FLY_APP_NAME", "").strip() or None,
                "fly_image_ref": os.getenv("FLY_IMAGE_REF", "").strip() or None,
                "fly_machine_version": os.getenv("FLY_MACHINE_VERSION", "").strip() or None,
                "web_origin": _canonical_https_origin(os.getenv("JUPR_WEB_BASE_URL")),
                "write_wave": write_wave,
                "staging_write_wave": write_wave,
                "business_data_write_wave_active": write_wave not in {None, "none"},
                "production_business_write_policy": os.getenv(
                    "JUPR_PRODUCTION_WRITE_POLICY", ""
                ).strip()
                or None,
                "security_denial_audit_logging_required": is_api_audit_log_required(),
                "jwt_verification_configured": jwt_verification_configured(),
                "jwt_verification_mode": jwt_verification_mode(),
                "jwt_verification_project_ref": jwt_verification_project_ref(),
                "feature_flags": feature_flags,
                "feature_flag_fingerprint": feature_flag_fingerprint(feature_flags),
                "controlled_write_flags": controlled_write_flags,
                "controlled_write_flag_fingerprint": feature_flag_fingerprint(
                    controlled_write_flags
                ),
                "public_live_writes_enabled": is_public_live_write_enabled(),
                "public_live_production_override_enabled": os.getenv(
                    "JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION", ""
                ).strip().lower()
                in {"1", "true", "yes"},
                "registration_edit_secret_configured": len(
                    os.getenv("JUPR_REGISTRATION_EDIT_SECRET", "").strip()
                )
                >= 32,
                "registration_confirmation_secret_configured": len(
                    os.getenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "").strip()
                )
                >= 32,
                "write_prerequisites": {
                    "service_role_configured": bool(
                        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
                    ),
                    "api_audit_required": os.getenv(
                        "JUPR_REQUIRE_API_AUDIT_LOG", ""
                    ).strip().lower()
                    in {"1", "true", "yes", "y", "on"},
                    "worker_run_log_required": os.getenv(
                        "JUPR_REQUIRE_WORKER_RUN_LOG", ""
                    ).strip().lower()
                    in {"1", "true", "yes", "y", "on"},
                    "email_mode": os.getenv("JUPR_EMAIL_MODE", "").strip().lower() or None,
                    "live_player_update_email_enabled": os.getenv(
                        "JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL", ""
                    ).strip().lower()
                    in {"1", "true", "yes", "y", "on"},
                },
                "expected_migration_head": os.getenv(
                    "JUPR_EXPECTED_MIGRATION_HEAD", ""
                ).strip()
                or None,
                "expected_migration_contract": os.getenv(
                    "JUPR_EXPECTED_MIGRATION_CONTRACT", ""
                ).strip().lower()
                or None,
                "expected_migration_profile": os.getenv(
                    "JUPR_EXPECTED_MIGRATION_PROFILE", ""
                ).strip()
                or None,
                "cors_allowed_origins": get_cors_allowed_origins(),
                "cors_allowed_origin_regex": get_cors_allowed_origin_regex(),
                "supabase_project_ref": (
                    supabase_host.split(".", 1)[0]
                    if supabase_host.endswith(".supabase.co")
                    else None
                ),
            }
        )
    return payload


@app.get("/health/live-sessions")
def health_live_sessions() -> dict[str, Any]:
    host = _supabase_host_for_diagnostics()
    if not _has_supabase_service_role_key():
        return {"ok": False, "service": "jupr-api", "supabase_host": host, "service_role_configured": False, "detail": LIVE_SESSIONS_SETUP_ERROR}
    supabase = None
    try:
        supabase = get_supabase_client()
        rows = supabase.table("live_sessions").select("club_id,session_key,status,version,edit_token_hash,updated_at").limit(1).execute().data or []
    except Exception as exc:
        if supabase is not None and _is_public_live_durability_schema_error(exc):
            try:
                legacy_rows = supabase.table("live_sessions").select(PUBLIC_LIVE_SESSION_LEGACY_SELECT).limit(1).execute().data or []
            except Exception as legacy_exc:
                return {"ok": False, "service": "jupr-api", "supabase_host": host, "service_role_configured": True, "live_sessions_query_ok": False, "detail": _error_payload_text(legacy_exc) or legacy_exc.__class__.__name__}
            return {
                "ok": False,
                "service": "jupr-api",
                "supabase_host": host,
                "service_role_configured": True,
                "live_sessions_query_ok": True,
                "operation_ledger_query_ok": False,
                "durability_schema_ready": False,
                "sample_count": len(legacy_rows),
                "detail": "Apply the public live durability migration before enabling writes.",
            }
        return {"ok": False, "service": "jupr-api", "supabase_host": host, "service_role_configured": True, "live_sessions_query_ok": False, "detail": _error_payload_text(exc) or exc.__class__.__name__}
    try:
        supabase.table("public_live_operations").select("operation_key,status,executor_token,lease_expires_at").limit(1).execute()
    except Exception:
        return {
            "ok": False,
            "service": "jupr-api",
            "supabase_host": host,
            "service_role_configured": True,
            "live_sessions_query_ok": True,
            "operation_ledger_query_ok": False,
            "durability_schema_ready": False,
            "sample_count": len(rows),
            "detail": "Apply the public live durability migration before enabling writes.",
        }
    token_secret_ready = len(os.getenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "").strip()) >= 32
    rate_secret_ready = len(
        (
            os.getenv("JUPR_PUBLIC_LIVE_RATE_LIMIT_SECRET", "").strip()
            or os.getenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "").strip()
        )
    ) >= 32
    return {
        "ok": bool(token_secret_ready and rate_secret_ready),
        "service": "jupr-api",
        "supabase_host": host,
        "service_role_configured": True,
        "live_sessions_query_ok": True,
        "operation_ledger_query_ok": True,
        "durability_schema_ready": True,
        "token_secret_configured": token_secret_ready,
        "rate_limit_secret_configured": rate_secret_ready,
        "sample_count": len(rows),
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
        rows = supabase.table("clubs").select(club_fields).eq("slug", slug).limit(1).execute().data or []
        if not rows:
            for club_id in _club_lookup_candidates(slug):
                rows = supabase.table("clubs").select(club_fields).eq("id", club_id).limit(1).execute().data or []
                if rows:
                    break
    except Exception:
        try:
            rows = supabase.table("clubs").select(club_minimal_fields).eq("slug", slug).limit(1).execute().data or []
            if not rows:
                for club_id in _club_lookup_candidates(slug):
                    rows = supabase.table("clubs").select(club_minimal_fields).eq("id", club_id).limit(1).execute().data or []
                    if rows:
                        break
        except Exception:
            rows = []
    if rows:
        row = rows[0] or {}
        return {"id": row.get("id"), "slug": row.get("slug") or slug, "name": row.get("name") or _display_name_from_slug(slug), "tagline": row.get("tagline"), "support_email": row.get("support_email"), "public_base_url": row.get("public_base_url"), "logo_url": row.get("logo_url"), "primary_color": row.get("primary_color"), "is_active": row.get("is_active", True)}
    for club_id in _club_lookup_candidates(slug):
        try:
            fallback = supabase.table("players").select("club_id").eq("club_id", club_id).limit(1).execute().data or []
        except Exception:
            fallback = []
        if fallback:
            return {"id": str(fallback[0].get("club_id") or club_id), "slug": slug, "name": _display_name_from_slug(slug), "tagline": None, "support_email": None, "public_base_url": None, "logo_url": None, "primary_color": None, "is_active": True}
    if known_fallback:
        return known_fallback
    raise HTTPException(status_code=404, detail="club not found")


install_public_match_explorer_routes(app, get_club=get_club, get_supabase_client=get_supabase_client, public_club_payload=_public_club_payload)
install_public_league_results_routes(app, get_club=get_club, get_supabase_client=get_supabase_client, public_club_payload=_public_club_payload)
install_public_badge_codex_routes(app, get_club=get_club, get_supabase_client=get_supabase_client, public_club_payload=_public_club_payload)
install_public_challenge_ladder_routes(app, get_club=get_club, get_supabase_client=get_supabase_client, public_club_payload=_public_club_payload)
install_public_weekly_recap_routes(app, get_club=get_club, get_supabase_client=get_supabase_client, public_club_payload=_public_club_payload)
install_public_team_league_routes(app, get_club=get_club, get_supabase_client=get_supabase_client, public_club_payload=_public_club_payload)
install_public_tournament_team_routes(app, get_club=get_club, get_supabase_client=get_supabase_client, public_club_payload=_public_club_payload)
install_admin_team_league_routes(app, get_supabase_client=get_supabase_client)
install_admin_tournament_team_competition_routes(app, get_supabase_client=get_supabase_client)


@app.get("/clubs/{club_slug}/leaderboards")
def get_club_leaderboard(
    club_slug: str,
    league_name: str | None = Query(default=None),
    status: str = Query(default="active"),
    q: str | None = Query(default=None, max_length=120),
    sort: str = Query(default="rank"),
    player_id: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return _build_leaderboard_response(
        club_slug,
        league_name,
        status=status,
        search=q,
        sort=sort,
        player_id=player_id,
        limit=limit,
        offset=offset,
    )


@app.get("/clubs/{club_slug}/leaderboards/public")
def get_club_leaderboard_compat(
    club_slug: str,
    league_name: str | None = Query(default=None),
    status: str = Query(default="active"),
    q: str | None = Query(default=None, max_length=120),
    sort: str = Query(default="rank"),
    player_id: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return _build_leaderboard_response(
        club_slug,
        league_name,
        status=status,
        search=q,
        sort=sort,
        player_id=player_id,
        limit=limit,
        offset=offset,
    )


@app.get("/clubs/{club_slug}/players")
def get_club_players(
    club_slug: str,
    q: str | None = Query(default=None, max_length=80),
    status: str = Query(default="active", max_length=16),
    sort: str = Query(default="rating", max_length=16),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    directory = build_public_player_directory(
        supabase,
        club_id=club_id,
        search=q,
        status=status,
        sort=sort,
        limit=limit,
        offset=offset,
    )
    return {"club": _public_club_payload(club, club_slug), **directory}


@app.get("/clubs/{club_slug}/players/{player_id}")
def get_club_player_profile(
    club_slug: str,
    player_id: str,
    recent_limit: int = Query(default=12, ge=1, le=25),
    history_limit: int = Query(default=500, ge=12, le=500),
) -> dict[str, Any]:
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    profile = get_public_player_profile(
        supabase,
        club_id=club_id,
        player_id=player_id,
        recent_match_limit=recent_limit,
        history_limit=history_limit,
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="player not found")
    return {"club": _public_club_payload(club, club_slug), **profile}


@app.get("/clubs/{club_slug}/matches")
def get_club_matches(club_slug: str, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    return {"club": _public_club_payload(club, club_slug), "matches": get_public_matches(supabase, club_id=club_id, limit=limit)}


@app.get("/clubs/{club_slug}/matches/{match_id}")
def get_club_match_detail(club_slug: str, match_id: str) -> dict[str, Any]:
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    match = get_public_match_detail(supabase, club_id=club_id, match_id=match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="match not found")
    return {"club": _public_club_payload(club, club_slug), "match": match}


@app.get("/clubs/{club_slug}/players/{player_id}/matches")
def get_club_player_matches(club_slug: str, player_id: str, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    return {"club": _public_club_payload(club, club_slug), "matches": get_public_matches(supabase, club_id=club_id, player_id=player_id, limit=limit)}


@app.get("/clubs/{club_slug}/live-sessions")
def get_club_live_sessions(club_slug: str, limit: int = Query(default=20, ge=1, le=50)) -> dict[str, Any]:
    return _build_live_sessions_response(club_slug, limit)


@app.post("/clubs/{club_slug}/live-sessions")
def create_club_live_session(club_slug: str, payload: PublicLiveSessionCreateRequest, request: Request) -> dict[str, Any]:
    _require_public_live_writes()
    _require_live_sessions_service_role()
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    try:
        result = create_public_live_session(
            supabase,
            club_id=club_id,
            event_name=payload.event_name,
            event_type=payload.event_type,
            participant_names=payload.participant_names,
            live_mode=payload.live_mode,
            total_rounds=payload.total_rounds,
            court_sizes=payload.court_sizes,
            host_name=payload.host_name,
            skill_levels=payload.skill_levels,
            participant_player_ids=payload.participant_player_ids,
            idempotency_key=payload.idempotency_key,
            requester_hash=_public_live_requester_hash(request),
        )
    except Exception as exc:
        _raise_public_live_write_error(exc)
    return {"club": _public_club_payload(club, club_slug), **result}


@app.get("/clubs/{club_slug}/live-sessions/{session_key}")
def get_club_live_session(club_slug: str, session_key: str) -> dict[str, Any]:
    return _build_live_session_detail_response(club_slug, session_key)


@app.patch("/clubs/{club_slug}/live-sessions/{session_key}/scores")
def update_club_live_session_scores(club_slug: str, session_key: str, payload: PublicLiveScoreUpdateRequest, request: Request) -> dict[str, Any]:
    _require_public_live_writes()
    _require_live_sessions_service_role()
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    supabase = get_supabase_client()
    try:
        result = update_public_live_scores(
            supabase,
            club_id=club_id,
            session_key=session_key,
            edit_token=payload.edit_token,
            expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key,
            requester_hash=_public_live_requester_hash(request),
            scores=[score.model_dump() for score in payload.scores],
        )
    except Exception as exc:
        _raise_public_live_write_error(exc)
    return {"club": _public_club_payload(club, club_slug), **result}


@app.post("/clubs/{club_slug}/live-sessions/{session_key}/advance")
def advance_club_live_session(club_slug: str, session_key: str, payload: PublicLiveMutationRequest, request: Request) -> dict[str, Any]:
    _require_public_live_writes()
    _require_live_sessions_service_role()
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    try:
        result = advance_public_live_session(
            get_supabase_client(),
            club_id=club_id,
            session_key=session_key,
            edit_token=payload.edit_token,
            expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key,
            requester_hash=_public_live_requester_hash(request),
        )
    except Exception as exc:
        _raise_public_live_write_error(exc)
    return {"club": _public_club_payload(club, club_slug), **result}


@app.post("/clubs/{club_slug}/live-sessions/{session_key}/substitutions")
def substitute_club_live_session(club_slug: str, session_key: str, payload: PublicLiveSubstitutionRequest, request: Request) -> dict[str, Any]:
    _require_public_live_writes()
    _require_live_sessions_service_role()
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    try:
        result = substitute_public_live_participant(
            get_supabase_client(),
            club_id=club_id,
            session_key=session_key,
            edit_token=payload.edit_token,
            expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key,
            requester_hash=_public_live_requester_hash(request),
            scope=payload.scope,
            round_number=payload.round_number,
            original_participant_id=payload.original_participant_id,
            substitute_name=payload.substitute_name,
            substitute_player_id=payload.substitute_player_id,
            match_id=payload.match_id,
            note=payload.note,
        )
    except Exception as exc:
        _raise_public_live_write_error(exc)
    return {"club": _public_club_payload(club, club_slug), **result}


@app.post("/clubs/{club_slug}/live-sessions/{session_key}/complete")
def complete_club_live_session(club_slug: str, session_key: str, payload: PublicLiveMutationRequest, request: Request) -> dict[str, Any]:
    _require_public_live_writes()
    _require_live_sessions_service_role()
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    try:
        result = complete_public_live_session(
            get_supabase_client(),
            club_id=club_id,
            session_key=session_key,
            edit_token=payload.edit_token,
            expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key,
            requester_hash=_public_live_requester_hash(request),
        )
    except Exception as exc:
        _raise_public_live_write_error(exc)
    return {"club": _public_club_payload(club, club_slug), **result}


@app.get("/clubs/{club_slug}/live-sessions/{session_key}/export")
def export_club_live_session(
    club_slug: str,
    session_key: str,
    format: str = Query(default="csv", pattern="^(csv|json)$"),
) -> Response:
    _require_live_sessions_service_role()
    club = get_club(club_slug)
    club_id = str(club.get("id") or club.get("club_id") or club_slug)
    try:
        export = build_public_live_export(
            get_supabase_client(),
            club_id=club_id,
            session_key=session_key,
            export_format=format,
        )
    except Exception as exc:
        _raise_public_live_write_error(exc)
    return Response(
        content=str(export["content"]),
        media_type=str(export["media_type"]),
        headers={"Content-Disposition": f'attachment; filename="{export["filename"]}"'},
    )


@app.get("/admin/clubs/{club_id}/score-entry/status")
def get_admin_score_entry_status(club_id: str) -> dict[str, Any]:
    flag_enabled = is_next_admin_score_entry_enabled()
    service_role_configured = _has_supabase_service_role_key()
    ready = flag_enabled and service_role_configured
    return {
        "enabled": flag_enabled,
        "ready": ready,
        "status": "ready" if ready else "fallback_required",
        "service_role_configured": service_role_configured,
        "submit_endpoint": "/admin/clubs/{club_id}/matches/batch" if ready else None,
        "max_matches": 1,
        "fallback": {
            "match_uploader_route": "/admin/match-uploader",
            "match_log_route": "/admin/match-log",
            "streamlit_url": os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "https://juprtrespalapas.streamlit.app").strip(),
        },
        "warnings": [] if ready else [SCORE_ENTRY_SETUP_ERROR],
    }


@app.post("/admin/clubs/{club_id}/matches/batch")
def submit_admin_match_batch(club_id: str, payload: MatchBatchRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
    if not is_next_admin_score_entry_enabled():
        raise HTTPException(status_code=403, detail="Next admin score entry is disabled. Use Match Uploader or the Streamlit fallback.")
    if not _has_supabase_service_role_key():
        raise HTTPException(status_code=503, detail=SCORE_ENTRY_SETUP_ERROR)
    user = authenticate_bearer(authorization)
    supabase = get_supabase_client()
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied_payload = build_activity_payload(club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="submit_match_batch_denied", entity_type="matches", entity_id="batch", after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"}, source_page=payload.source, flagged_for_review=True)
        write_admin_activity_log(supabase, denied_payload)
        raise HTTPException(status_code=403, detail="insufficient permission")
    _validate_score_entry_match(payload.matches)
    clean_matches = _normalize_score_entry_match(payload.matches)
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
    try:
        result = submit_atomic_direct_matches(
            supabase,
            club_id=str(club_id),
            matches=clean_matches,
            match_format="doubles",
            idempotency_key=payload.idempotency_key,
            actor_email=user.email,
            actor_role=role_resolution.role,
            source=payload.source,
            name_to_id=name_to_id,
            df_players_all=df_players_all,
            df_leagues=df_leagues,
            df_meta=df_meta,
        )
    except DirectMatchConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DirectMatchRecoveryRequiredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **result,
        "ok": True,
        "auth_mode": "supabase_jwt",
        "required_permission": PERMISSION_ENTER_SCORES,
        "match_write_committed": True,
        "recovery": {
            "match_log_route": "/admin/match-log",
            "match_uploader_route": "/admin/match-uploader",
            "replay_history_route": "/admin/replay-history",
            "operator_rule": "Retry the exact unchanged request after an interrupted response; the same idempotency key cannot create a duplicate.",
        },
    }
