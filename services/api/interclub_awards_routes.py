from __future__ import annotations

import hashlib
import json
from uuid import UUID

from fastapi import HTTPException, Response
from pydantic import Field

from jupr_app.domain.interclub_awards import season_awards
from jupr_app.services.interclub_awards_service import public_interclub_trophies
from services.api.auth import auth_header
from services.api.club_site_models import StrictModel
from services.api.club_site_routes import published_site, site_administrator, site_rpc
from services.api.interclub_public_routes import publication_response, reviewed_publication, season_context


class AwardSeason(StrictModel):
    revision: int = Field(ge=0)
    publication_revision: int = Field(ge=0)
    preview_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


def award_preview(db, season, clubs, meets):
    document, sources, source_fingerprint = reviewed_publication(db, season, clubs, meets)
    batches = db.table("pcs_interclub_competition_batches").select("meet_id,state,revision,approved_revision,ratings_status").eq("season_id", season["id"]).execute().data or []
    by_meet = {batch["meet_id"]: batch for batch in batches}
    problems = []
    if any(meet["id"] not in by_meet or by_meet[meet["id"]]["state"] != "approved"
           or by_meet[meet["id"]]["approved_revision"] != by_meet[meet["id"]]["revision"] for meet in meets):
        problems.append("Approve every scheduled meet and any open score corrections first.")
    if any(batch["ratings_status"] != "completed" for batch in batches):
        problems.append("Finish the meet rating updates before awarding trophies.")
    try:
        awards = season_awards(season["id"], document)
    except ValueError as exc:
        problems.append(str(exc)); awards = []
    sets = db.table("pcs_interclub_award_sets").select("revision,preview_fingerprint,issued_at").eq("season_id", season["id"]).limit(1).execute().data or []
    publications = db.table("pcs_interclub_publications").select("revision,published").eq("season_id", season["id"]).limit(1).execute().data or []
    current = sets[0] if sets else None
    fingerprint = hashlib.sha256(json.dumps({"source": source_fingerprint, "awards": awards}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    publication = publications[0] if publications else None
    return {"season_id": season["id"], "preview": {"id": season["id"], **publication_response(document)},
            "awards": awards, "problems": problems, "ready": not problems,
            "revision": current["revision"] if current else 0,
            "publication_revision": publication["revision"] if publication else 0,
            "issued_at": current["issued_at"] if current else None,
            "current": bool(current and current["preview_fingerprint"] == fingerprint and publication and publication["published"] == document),
            "preview_fingerprint": fingerprint}, document, sources


def install_interclub_awards_routes(app, *, get_supabase_client):
    def organizer(club_id, season_id, authorization):
        user = site_administrator(get_supabase_client, authorization, club_id)
        db = get_supabase_client()
        season, clubs, meets = season_context(db, str(season_id))
        if season["organizer_club_id"] != club_id:
            raise HTTPException(403, "Only the season organizer can award interclub trophies.")
        return user, db, season, clubs, meets

    @app.get("/admin/clubs/{club_id}/interclub/{season_id}/awards")
    def preview(club_id: str, season_id: UUID, response: Response, authorization: str | None = auth_header()):
        response.headers["Cache-Control"] = "no-store"
        _, db, season, clubs, meets = organizer(club_id, season_id, authorization)
        return award_preview(db, season, clubs, meets)[0]

    @app.post("/admin/clubs/{club_id}/interclub/{season_id}/awards")
    def issue(club_id: str, season_id: UUID, body: AwardSeason, authorization: str | None = auth_header()):
        user, db, season, clubs, meets = organizer(club_id, season_id, authorization)
        preview, document, sources = award_preview(db, season, clubs, meets)
        if not preview["ready"]:
            raise HTTPException(409, " ".join(preview["problems"]))
        if preview["preview_fingerprint"] != body.preview_fingerprint:
            raise HTTPException(409, "Season results or recipients changed. Reload and review the awards again.")
        return site_rpc(db, "pcs_award_interclub_season", {
            "p_actor_id": user.user_id, "p_actor_email": user.email, "p_club_id": club_id, "p_season_id": str(season_id),
            "p_revision": body.revision, "p_publication_revision": body.publication_revision,
            "p_preview_fingerprint": body.preview_fingerprint, "p_document": document, "p_sources": sources, "p_awards": preview["awards"]})

    @app.get("/public/clubs/{slug}/trophies")
    def club_trophies(slug: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        db = get_supabase_client(); site = published_site(db, slug)
        return {"club_name": site["document"]["name"], "trophies": public_interclub_trophies(db, club_id=site["club_id"])}

    @app.get("/public/interclub/{season_id}/awards")
    def awards(season_id: UUID, response: Response):
        response.headers["Cache-Control"] = "no-store"
        db = get_supabase_client()
        publications = db.table("pcs_interclub_publications").select("published").eq("season_id", str(season_id)).limit(1).execute().data or []
        if not publications or not publications[0].get("published"):
            raise HTTPException(404, "League website is not published.")
        return {"trophies": public_interclub_trophies(db, season_id=str(season_id))}
