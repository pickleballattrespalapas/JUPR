from __future__ import annotations

from fastapi import HTTPException, Response

from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import auth_header
from services.api.club_site_models import SiteAction, SiteDocument, SiteSave, default_site


SITE_FIELDS = "club_id,revision,draft,published,published_at,updated_at"


def site_administrator(db_factory, authorization, club_id):
    user, assignments = require_admin_assignments(get_supabase_client=db_factory,
        authorization=authorization, requested_club_id=club_id)
    if not any(a["role"] in {"super_admin", "administrator", "club_owner"} for a in assignments):
        raise HTTPException(403, "Club administrator access required.")
    return user


def site_rpc(db, function, params):
    try:
        return db.rpc(function, params).execute().data
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        if code in {"40001", "PT409"}: raise HTTPException(409, "This draft changed. Reload before saving or publishing.") from exc
        if code == "42501": raise HTTPException(403, "Verified club administrator access required.") from exc
        if code == "23505": raise HTTPException(409, "That club address is already in use. Choose another address.") from exc
        if code in {"22023", "23514"}: raise HTTPException(422, "Check the website details and try again.") from exc
        if code == "P0002": raise HTTPException(404, "Club or season unavailable.") from exc
        if code == "54000": raise HTTPException(429, "Club creation limit reached. Contact PCS for help.") from exc
        raise HTTPException(503, "Could not confirm the save. Reload before trying again.") from exc


def published_site(db, slug):
    clubs = db.table("clubs").select("id,slug,is_active,public_site_status").eq("slug", slug).limit(1).execute().data or []
    if not clubs or not clubs[0].get("is_active") or clubs[0].get("public_site_status") != "published":
        raise HTTPException(404, "Club website is not published.")
    club = clubs[0]
    rows = db.table("pcs_club_sites").select("published,published_at").eq("club_id", club["id"]).limit(1).execute().data or []
    if not rows or not rows[0].get("published"):
        raise HTTPException(404, "Club website is not published.")
    return {"club_id": club["id"], "slug": club["slug"], "document": SiteDocument.model_validate(rows[0]["published"]).model_dump(), "published_at": rows[0]["published_at"]}


def install_club_site_routes(app, *, get_supabase_client):
    @app.get("/public/clubs/{slug}/site")
    def public_site(slug: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        site = published_site(get_supabase_client(), slug)
        if site["document"]["visibility"] == "unlisted": response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return site

    @app.get("/admin/clubs/{club_id}/site")
    def admin_site(club_id: str, response: Response, authorization: str | None = auth_header()):
        response.headers["Cache-Control"] = "no-store"
        site_administrator(get_supabase_client, authorization, club_id)
        db = get_supabase_client()
        clubs = db.table("clubs").select("id,slug,name,tagline,logo_url,is_active").eq("id", club_id).limit(1).execute().data or []
        if not clubs: raise HTTPException(404, "Club unavailable.")
        club = clubs[0]
        rows = db.table("pcs_club_sites").select(SITE_FIELDS).eq("club_id", club_id).limit(1).execute().data or []
        site = rows[0] if rows else {"club_id": club_id, "revision": 0, "draft": default_site(club), "published": None, "published_at": None}
        return {**site, "slug": club["slug"], "club_active": club["is_active"]}

    @app.put("/admin/clubs/{club_id}/site")
    def save_site(club_id: str, body: SiteSave, authorization: str | None = auth_header()):
        user = site_administrator(get_supabase_client, authorization, club_id)
        return site_rpc(get_supabase_client(), "save_club_website_settings", {"p_actor_id": user.user_id,
            "p_actor_email": user.email, "p_club_id": club_id, "p_revision": body.revision,
            "p_action": "save", "p_document": body.document.model_dump(mode="json")})

    @app.post("/admin/clubs/{club_id}/site/{action}")
    def publish_site(club_id: str, action: str, body: SiteAction, authorization: str | None = auth_header()):
        if action not in {"publish", "unpublish", "discard"}: raise HTTPException(404, "Unknown action.")
        user = site_administrator(get_supabase_client, authorization, club_id)
        return site_rpc(get_supabase_client(), "save_club_website_settings", {"p_actor_id": user.user_id,
            "p_actor_email": user.email, "p_club_id": club_id, "p_revision": body.revision,
            "p_action": action, "p_document": None})
