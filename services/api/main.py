from __future__ import annotations

import os
from typing import Any

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



def get_supabase_client() -> Client:
    url, key = _get_supabase_credentials()
    return create_client(url, key)



def _is_missing_table_error(exc: Exception, table_name: str) -> bool:
    detail = str(exc).lower()
    table = table_name.lower()
    return table in detail and (
        "does not exist" in detail
        or "undefined table" in detail
        or "relation" in detail
        or "not found" in detail
        or "could not find" in detail
        or "schema cache" in detail
    )



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
        "club": {
            "id": str(club.get("id") or club.get("club_id") or club_id),
            "slug": str(club.get("slug") or club.get("club_slug") or club_slug),
            "name": str(club.get("name") or club.get("club_name") or club.get("display_name") or club_slug),
        },
        "leaderboard": _normalize_public_leaderboard_rows(rows),
    }


@app.on_event("startup")
def startup_checks() -> None:
    _log_runtime_guardrails()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "jupr-api"}


@app.get("/clubs/{club_slug}")
def get_club(club_slug: str) -> dict[str, Any]:
    slug = str(club_slug).strip()
    if not slug:
        raise HTTPException(status_code=400, detail="club_slug is required")

    supabase = get_supabase_client()
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
    except Exception as exc:
        if not _is_missing_table_error(exc, "clubs"):
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
            except Exception as fallback_exc:
                if not _is_missing_table_error(fallback_exc, "clubs"):
                    raise

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
        fallback = (
            supabase.table("players")
            .select("club_id")
            .eq("club_id", club_id)
            .limit(1)
            .execute()
            .data
            or []
        )
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

    raise HTTPException(status_code=404, detail="club not found")


@app.get("/clubs/{club_slug}/leaderboards")
def get_club_leaderboard(club_slug: str, league_name: str | None = Query(default=None)) -> dict[str, Any]:
    return _build_leaderboard_response(club_slug, league_name)


@app.get("/clubs/{club_slug}/leaderboards/public")
def get_club_leaderboard_compat(club_slug: str, league_name: str | None = Query(default=None)) -> dict[str, Any]:
    # Temporary compatibility alias for Next.js clients still calling /leaderboards/public.
    # Remove this route after the web app fully migrates to /leaderboards.
    return _build_leaderboard_response(club_slug, league_name)


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
    df_players_all, _, df_leagues, _, df_meta, _, _, _, _, name_to_id = load_data(supabase, club_id)

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
