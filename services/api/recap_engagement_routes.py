"""Public recap activities and their separate, authenticated moderation queue."""
from datetime import date, datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import HTTPException, Request, Response
from postgrest.exceptions import APIError
from pydantic import BaseModel, Field

from jupr_app.services.admin_weekly_recap_service import is_admin_weekly_recap_enabled
from services.api.admin_weekly_recap_routes import (
    _require_communications_mutations, _require_service_role, _resolve_weekly_recap_role_or_403,
)
from services.api.auth import auth_header
from services.api.staging_write_guard import require_public_intake_or_403


class ReaderAction(BaseModel):
    visitor: UUID
    source: Literal["website", "email", "whatsapp"] = "website"
    action: Literal["visit", "vote", "shoutout"]
    question: str = Field(default="", max_length=200)
    answer: str = Field(default="", max_length=80)
    name: str = Field(default="", max_length=80)
    story: str = Field(default="", max_length=400)


class Review(BaseModel):
    decision: Literal["featured", "dismissed"]
    permission: bool = False


def edition(db, club_id, week, public=False):
    fields = "recap_id,scope_key,config,closes_at,followup" + (",weekly_recaps!inner(status)" if public else "")
    q = db.table("recap_engagement_editions").select(fields).eq("club_id", club_id).eq("week_start", str(week))
    if public:
        q = q.eq("weekly_recaps.status", "published")
    rows = q.limit(1).execute().data
    return rows[0] if rows else None


def install_recap_engagement_routes(app, *, get_club, get_supabase_client, requester_hash):
    def admin(db, club, token):
        if not is_admin_weekly_recap_enabled():
            raise HTTPException(403, "Weekly Recap Admin is disabled.")
        _require_service_role()
        return _resolve_weekly_recap_role_or_403(supabase=db, club_id=club, authorization=token, source="recap_engagement")[0]

    def club_key(slug):
        club = get_club(slug)
        return str(club.get("id") or club.get("club_id") or slug)

    @app.get("/clubs/{club_slug}/weekly-recaps/{week}/engagement")
    def public_activity(club_slug: str, week: date, response: Response):
        response.headers["Cache-Control"] = "no-store"
        db = get_supabase_client()
        row = edition(db, club_key(club_slug), week, public=True)
        if not row:
            raise HTTPException(404, "This edition is unavailable.")
        cfg = row["config"]
        on = cfg["mode"] != "off" and (cfg["mode"] != "review" or cfg["approved"])
        feature = cfg["featured"] if on else "none"
        nominations = feature == "shoutout" or (feature == "poll" and cfg["shoutouts"])
        stories = (db.table("recap_engagement_readers").select("nominee,story").eq("recap_id", row["recap_id"])
                   .eq("moderation", "featured").order("created_at", desc=True).limit(6).execute().data) if nominations else []
        return {"featured": feature, "open": datetime.fromisoformat(row["closes_at"].replace("Z", "+00:00")) > datetime.now(timezone.utc),
                "poll": {k: cfg["poll"][k] for k in ("question", "options")} if feature == "poll" else None,
                "shoutouts": nominations, "stories": stories, "followup": row.get("followup")}

    @app.post("/clubs/{club_slug}/weekly-recaps/{week}/engagement")
    def public_action(club_slug: str, week: date, payload: ReaderAction, request: Request, response: Response):
        require_public_intake_or_403()
        response.headers["Cache-Control"] = "no-store"
        if payload.action == "shoutout" and not (payload.name.strip() and payload.story.strip()):
            raise HTTPException(422, "Enter a member’s name and one sentence.")
        db = get_supabase_client()
        club_id = club_key(club_slug)
        try:
            receipt = db.rpc("record_recap_engagement", {
                "p_club": club_id, "p_week": str(week), "p_visitor": str(payload.visitor),
                "p_source": payload.source, "p_hash": requester_hash(request), "p_action": payload.action,
                "p_question": payload.question, "p_answer": payload.answer, "p_name": payload.name, "p_story": payload.story,
            }).execute().data
        except APIError as exc:
            if exc.code == "P0002":
                raise HTTPException(404, "This edition is unavailable.") from exc
            if exc.message == "Too many readers":
                raise HTTPException(429, "Please try again later.") from exc
            if exc.message in {"Voting is closed", "Nominations are closed", "Poll changed"}:
                raise HTTPException(409, f"{exc.message}. Refresh the recap.") from exc
            raise

        if receipt["choice"] is not None:
            row = edition(db, club_id, week, public=True)
            if row:
                totals = db.rpc("recap_engagement_counts", {"p_id": row["recap_id"]}).execute().data
                receipt["counts"] = [totals["first"], totals["second"]]
        return receipt

    @app.get("/admin/clubs/{club_id}/weekly-recap/recaps/{week}/engagement")
    def activity_stats(club_id: str, week: date, authorization: str | None = auth_header()):
        db = get_supabase_client()
        admin(db, club_id, authorization)
        row = edition(db, club_id, week)
        if row:
            scope = row["scope_key"]
        else:
            draft = (db.table("weekly_recaps").select("scope:final_json->report->>scope_label")
                     .eq("club_id", club_id).eq("week_start", str(week)).limit(1).execute().data or [])
            scope = (draft[0].get("scope") if draft else None) or "Whole club"
        previous_rows = (db.table("recap_engagement_editions").select("recap_id,week_start,config,weekly_recaps!inner(status)")
                         .eq("club_id", club_id).eq("scope_key", scope).eq("weekly_recaps.status", "published")
                         .lt("week_start", str(week)).order("week_start", desc=True).limit(1).execute().data or [])
        previous = None
        if previous_rows:
            last = previous_rows[0]
            totals = db.rpc("recap_engagement_counts", {"p_id": last["recap_id"]}).execute().data
            previous = {"week": last["week_start"], "question": last["config"]["poll"]["question"],
                        "options": last["config"]["poll"]["options"], "counts": [totals["first"], totals["second"]]}
        if not row:
            return {"counts": None, "pending": [], "published_poll": None, "previous": previous}
        rid = row["recap_id"]
        pending = (db.table("recap_engagement_readers").select("id,nominee,story").eq("recap_id", rid)
                   .eq("moderation", "pending").order("created_at").limit(50).execute().data)
        return {"counts": db.rpc("recap_engagement_counts", {"p_id": rid}).execute().data,
                "pending": pending, "published_poll": row["config"]["poll"], "previous": previous}

    @app.post("/admin/clubs/{club_id}/weekly-recap/recaps/{week}/engagement/nominations/{nomination}")
    def review_nomination(club_id: str, week: date, nomination: UUID, payload: Review, authorization: str | None = auth_header()):
        _require_communications_mutations()
        db = get_supabase_client()
        actor = admin(db, club_id, authorization)
        if payload.decision == "featured" and not payload.permission:
            raise HTTPException(422, "Confirm permission before featuring this story.")
        row = edition(db, club_id, week)
        if not row:
            raise HTTPException(404, "Edition unavailable.")
        changed = (db.table("recap_engagement_readers").update({"moderation": payload.decision, "reviewed_by": actor})
                   .eq("id", str(nomination)).eq("recap_id", row["recap_id"]).eq("moderation", "pending").execute().data)
        if not changed:
            raise HTTPException(409, "This nomination is no longer pending.")
        return {"ok": True}
