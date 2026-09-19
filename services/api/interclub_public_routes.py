from __future__ import annotations

import hashlib
import json
from uuid import UUID
from fastapi import HTTPException, Response
from pydantic import Field, model_validator

from services.api.auth import auth_header
from services.api.club_site_models import StrictModel, SiteAction
from services.api.club_site_routes import published_site, site_administrator, site_rpc
from services.api.interclub_competition_routes import approved_documents
from jupr_app.domain import interclub_competition as competition


class GameScore(StrictModel):
    a: int = Field(ge=0, le=100)
    b: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def completed(self):
        high, low = max(self.a, self.b), min(self.a, self.b)
        if high < 11 or high - low < 2 or (high > 11 and high - low != 2):
            raise ValueError("A completed game is to 11, win by two, no cap.")
        return self


class Encounter(StrictModel):
    id: UUID
    meet_id: UUID
    division: str = Field(min_length=1, max_length=20)
    club_a: str = Field(min_length=1, max_length=100)
    club_b: str = Field(min_length=1, max_length=100)
    games: list[GameScore] = Field(min_length=3, max_length=3)


class ResultsDocument(StrictModel):
    results: list[Encounter] = Field(default_factory=list, max_length=2000)


class ResultsSave(StrictModel):
    revision: int = Field(ge=0)
    document: ResultsDocument


class PublicationAction(SiteAction):
    preview_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


def season_context(db, season_id):
    seasons = db.table("pcs_interclub_seasons").select("id,organizer_club_id,details").eq("id", season_id).limit(1).execute().data or []
    if not seasons: raise HTTPException(404, "Season unavailable.")
    season = seasons[0]
    participants = db.table("pcs_interclub_participations").select("club_id,status").eq("season_id", season_id).execute().data or []
    ids = [p["club_id"] for p in participants if p["status"] == "accepted"]
    clubs = db.table("clubs").select("id,name").in_("id", ids).execute().data if ids else []
    meets = db.table("pcs_interclub_meets").select("id,host_club_id,club_ids,starts_at,duration_minutes,courts,revision").eq("season_id", season_id).order("starts_at").execute().data or []
    return season, clubs or [], meets


def validate_results(document, season, clubs, meets):
    allowed = {c["id"] for c in clubs}
    scheduled = {str(m["id"]): m for m in meets}
    seen = set()
    ids = set()
    for row in document.results:
        meet = scheduled.get(str(row.meet_id))
        key = (str(row.meet_id), row.division, tuple(sorted([row.club_a, row.club_b])))
        if not meet or row.division not in season["details"]["divisions"] or row.club_a == row.club_b or not {row.club_a, row.club_b}.issubset(allowed & set(meet["club_ids"])):
            raise HTTPException(422, "Results must use accepted clubs and a scheduled meet in this season.")
        if key in seen or row.id in ids:
            raise HTTPException(422, "An encounter can be recorded only once per meet, division and club pairing.")
        seen.add(key); ids.add(row.id)


def league_standings(document):
    result = []
    for division in document["divisions"]:
        rows = {c["id"]: {"club_id": c["id"], "name": c["name"], "played": 0, "wins": 0, "losses": 0,
            "games_won": 0, "games_lost": 0, "point_difference": 0} for c in document["clubs"]}
        for match in document["results"]:
            if match["division"] != division: continue
            if match["club_a"] not in rows or match["club_b"] not in rows: continue
            a, b = rows[match["club_a"]], rows[match["club_b"]]
            wins_a = sum(g["a"] > g["b"] for g in match["games"])
            diff = sum(g["a"] - g["b"] for g in match["games"])
            for own, won, delta in [(a, wins_a, diff), (b, 3-wins_a, -diff)]:
                own["played"] += 1; own["wins"] += int(won >= 2); own["losses"] += int(won < 2)
                own["games_won"] += won; own["games_lost"] += 3-won; own["point_difference"] += delta
        ordered = sorted(rows.values(), key=lambda r: (-r["wins"], -(r["games_won"]-r["games_lost"]), -r["point_difference"], r["name"].casefold()))
        result.append({"division": division, "rows": ordered})
    return result


def public_document(season, clubs, meets, results):
    doc = {k: season["details"][k] for k in ["name", "start_date", "end_date", "timezone", "divisions"]}
    accepted = {c["id"] for c in clubs}
    public_meets = [{**{key: value for key, value in m.items() if key != "revision"}, "club_ids": [cid for cid in m["club_ids"] if cid in accepted],
                     "host_club_id": m["host_club_id"] if m["host_club_id"] in accepted else None} for m in meets]
    doc.update({"clubs": clubs, "meets": public_meets, **results})
    return doc


def competition_publication(db, season, clubs, meets, *, documents=None):
    """Freeze official competition totals, never private pool or rating-job records."""
    if documents is None:
        documents = approved_documents(db, season["id"])
    tables = competition.league_standings(documents, clubs=clubs)
    cup = competition.club_cup(documents, clubs=clubs)
    summaries = []
    for document in documents:
        for encounter in document["encounters"]:
            summaries.append({
                "id": encounter["id"], "meet_id": document["meet_id"],
                "phase": document["phase"], "weather": document["weather"],
                "division": encounter["division"],
                "club_a": encounter["club_a"], "club_b": encounter["club_b"],
                "pairings": [{"kind": pairing["kind"], "games": [
                    {key: game.get(key) for key in ("status", "a", "b", "winner")}
                    for game in pairing["games"]
                ]} for pairing in encounter["pairings"]],
                "tiebreak": ({key: encounter["tiebreak"].get(key) for key in ("status", "a", "b")}
                             if encounter.get("tiebreak") else None),
            })
    doc = public_document(season, clubs, meets, {"results": []})
    doc.update({"scoring_version": 1, "competition_results": summaries,
                "competition_standings": [{"division": division, "rows": tables["divisions"].get(division, [])}
                                          for division in doc["divisions"]],
                "club_cup": cup, "qualification": tables["qualification"]})
    return doc


def reviewed_publication(db, season, clubs, meets):
    """One canonical read binds the preview to exact official result revisions."""
    rows = db.table("pcs_interclub_competition_batches").select("id,approved_revision,approved_document").eq("season_id", season["id"]).execute().data or []
    official = sorted([row for row in rows if row.get("approved_document") is not None], key=lambda row: row["id"])
    clubs = sorted(clubs, key=lambda row: row["id"])
    meets = sorted(meets, key=lambda row: (row["starts_at"], row["id"]))
    sources = {
        "approved": sorted([{"id": row["id"], "revision": row["approved_revision"]} for row in official], key=lambda row: row["id"]),
        "season": season["details"],
        "clubs": sorted(clubs, key=lambda row: row["id"]),
        "meets": sorted([{"id": meet["id"], "revision": meet["revision"]} for meet in meets], key=lambda row: row["id"]),
    }
    document = competition_publication(db, season, clubs, meets, documents=[row["approved_document"] for row in official])
    encoded = json.dumps({"document": document, "sources": sources}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return document, sources, hashlib.sha256(encoded.encode()).hexdigest()


def publication_response(document):
    if document.get("scoring_version") == 1:
        return {"document": document, "standings": document["competition_standings"],
                "club_cup": document["club_cup"], "qualification": document["qualification"]}
    # Retain old published snapshots for reference. Their one-pairing results
    # cannot be inferred into the Southern BCS two-pairing scoring model.
    return {"document": document, "standings": league_standings(document)}


def install_interclub_public_routes(app, *, get_supabase_client):
    @app.get("/public/clubs/{slug}/interclub")
    def club_leagues(slug: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        db = get_supabase_client(); site = published_site(db, slug)
        memberships = db.table("pcs_interclub_participations").select("season_id").eq("club_id", site["club_id"]).eq("status", "accepted").execute().data or []
        ids = [m["season_id"] for m in memberships]
        rows = db.table("pcs_interclub_publications").select("season_id,published,published_at").in_("season_id", ids).execute().data if ids else []
        return {"leagues": [{"id": r["season_id"], "name": r["published"]["name"], "start_date": r["published"]["start_date"],
            "end_date": r["published"]["end_date"]} for r in rows or [] if r.get("published") and any(c["id"] == site["club_id"] for c in r["published"]["clubs"])]}

    @app.get("/public/interclub/{season_id}")
    def league(season_id: UUID, response: Response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        rows = get_supabase_client().table("pcs_interclub_publications").select("season_id,published,published_at").eq("season_id", str(season_id)).limit(1).execute().data or []
        if not rows or not rows[0].get("published"): raise HTTPException(404, "League website is not published.")
        doc = rows[0]["published"]
        return {"id": str(season_id), **publication_response(doc), "published_at": rows[0]["published_at"]}

    def organizer(club_id, season_id, authorization):
        user = site_administrator(get_supabase_client, authorization, club_id)
        db = get_supabase_client(); season, clubs, meets = season_context(db, str(season_id))
        if season["organizer_club_id"] != club_id: raise HTTPException(403, "Season organizer access required.")
        return user, db, season, clubs, meets

    @app.get("/admin/clubs/{club_id}/interclub/{season_id}/publication")
    def admin_publication(club_id: str, season_id: UUID, authorization: str | None = auth_header()):
        _, db, season, clubs, meets = organizer(club_id, season_id, authorization)
        rows = db.table("pcs_interclub_publications").select("*").eq("season_id", str(season_id)).limit(1).execute().data or []
        publication = rows[0] if rows else {"revision": 0, "draft": {"results": []}, "published": None}
        preview, _, fingerprint = reviewed_publication(db, season, clubs, meets)
        return {"season": season, "clubs": clubs, "meets": meets, "publication": publication,
                "preview": publication_response(preview), "preview_fingerprint": fingerprint}

    @app.put("/admin/clubs/{club_id}/interclub/{season_id}/publication")
    def save_publication(club_id: str, season_id: UUID, body: ResultsSave, authorization: str | None = auth_header()):
        user, db, season, clubs, meets = organizer(club_id, season_id, authorization)
        saved = db.table("pcs_interclub_publications").select("draft").eq("season_id", str(season_id)).limit(1).execute().data or []
        previous = saved[0]["draft"] if saved else {"results": []}
        if body.document.model_dump(mode="json") != previous:
            raise HTTPException(422, "Enter scores in the meet workspace, then submit the full meet for organizer approval.")
        return site_rpc(db, "pcs_write_interclub_publication", {"p_actor_id": user.user_id, "p_actor_email": user.email,
            "p_club_id": club_id, "p_season_id": str(season_id), "p_revision": body.revision, "p_action": "save", "p_document": body.document.model_dump(mode="json")})

    @app.post("/admin/clubs/{club_id}/interclub/{season_id}/publication/{action}")
    def publish(club_id: str, season_id: UUID, action: str, body: PublicationAction, authorization: str | None = auth_header()):
        if action not in {"publish", "unpublish"}: raise HTTPException(404, "Unknown action.")
        user, db, season, clubs, meets = organizer(club_id, season_id, authorization)
        doc = None
        if action == "publish":
            rows = db.table("pcs_interclub_publications").select("revision,draft").eq("season_id", str(season_id)).limit(1).execute().data or []
            if not rows or rows[0]["revision"] != body.revision: raise HTTPException(409, "Save or reload the draft before publishing.")
            if len(clubs) < 2: raise HTTPException(422, "At least two clubs must accept before publication.")
            doc, sources, fingerprint = reviewed_publication(db, season, clubs, meets)
            if not body.preview_fingerprint or body.preview_fingerprint != fingerprint:
                raise HTTPException(409, "Official results or the schedule changed. Reload and review the league preview before publishing.")
            return site_rpc(db, "pcs_publish_reviewed_interclub_publication", {"p_actor_id": user.user_id, "p_actor_email": user.email,
                "p_club_id": club_id, "p_season_id": str(season_id), "p_revision": body.revision,
                "p_document": doc, "p_sources": sources})
        return site_rpc(db, "pcs_write_interclub_publication", {"p_actor_id": user.user_id, "p_actor_email": user.email,
            "p_club_id": club_id, "p_season_id": str(season_id), "p_revision": body.revision, "p_action": action, "p_document": doc})
