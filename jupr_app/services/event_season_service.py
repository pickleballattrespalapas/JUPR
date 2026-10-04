"""Shared history for recurring interclub leagues, club leagues and tournaments."""
from __future__ import annotations

from urllib.parse import quote, urlencode

from jupr_app.domain.event_seasons import retained_champion_seasons


def _rows(query):
    result = []
    while True:
        page = query.range(len(result), len(result) + 999).execute().data or []
        result.extend(page)
        if len(page) < 1000:
            return result


def load_source(db, club_id, kind, source_id):
    return db.rpc("pcs_event_season_source", {"p_club_id": club_id, "p_kind": kind, "p_source_id": source_id}).execute().data


def admin_event_href(kind, source_id, *, draft=False):
    if kind == "interclub":
        return "/admin/interclub?" + urlencode({"season": source_id}) if draft else "/admin/interclub/season?" + urlencode({"season": source_id})
    if kind == "league":
        return "/admin/league-manager/league?" + urlencode({"league": source_id, "league_name": source_id})
    return "/admin/tournaments/setup?" + urlencode({"tournament": source_id}) if draft else "/admin/tournaments/tournament?" + urlencode({"tournament": source_id})


def source_summary(kind, source_id, source, slug=""):
    event = source["event"]
    setup = source.get("setup") or event.get("schedule_config") or event
    name = setup.get("name") if kind == "interclub" else event.get("league_name") if kind == "league" else event.get("name")
    start = setup.get("start_date") or event.get("started_at")
    end = setup.get("end_date") or event.get("ended_at")
    completed = bool(source.get("complete"))
    if kind == "interclub":
        results_href = f"/interclub/{source_id}" + ("/final-results" if completed else "")
        draft = "opened_at" not in event
    elif kind == "league":
        results_href = (f"/clubs/{quote(slug, safe='')}/team-leagues/{quote(source_id, safe='')}" if event.get("league_type") == "Team" else
                        f"/clubs/{quote(slug, safe='')}/leagues/{quote(source_id, safe='')}/standings")
        draft = event.get("status") in {"draft", "planned", "inactive"}
    else:
        results_href = f"/clubs/{quote(slug, safe='')}/tournament-results?" + urlencode({"tournament_id": source_id})
        draft = str(event.get("status", "")).upper() == "DRAFT"
    return {"source_id": source_id, "name": name or source_id, "start_date": str(start)[:10] if start else None,
            "end_date": str(end)[:10] if end else None, "complete": completed, "public": bool(source.get("public")),
            "status": "Completed" if completed else "Draft" if draft else "In progress", "league_type": event.get("league_type") if kind == "league" else None,
            "results_href": results_href if source.get("public") and (kind == "interclub" or slug) else None,
            "admin_href": admin_event_href(kind, source_id, draft=draft), "fingerprint": source["fingerprint"]}


def _honors(db, club_id, kind, source_id, source):
    if not source.get("complete"):
        return []
    if kind == "interclub":
        from jupr_app.services.interclub_awards_service import public_interclub_trophies
        return [{"id": row["id"], "title": "League Champion" if row["award_key"] == "club_cup_champion" else f"{row['division']} Division Champion",
                 "recipient": row["recipient_name"], "placement": 1}
                for row in public_interclub_trophies(db, season_id=source_id)
                if row["recipient_type"] == "club" and row["award_key"] in {"club_cup_champion", "division_champion"}]
    if kind == "league":
        # Every saved workflow step has its own result set. Read the latest
        # revision so an earlier preview cannot hide or replace published awards.
        sets = (db.table("league_award_result_sets").select("workflow_revision,result_fingerprint,finalized_at")
                .eq("club_id", club_id).eq("league_name", source_id)
                .order("workflow_revision", desc=True).limit(1).execute().data or [])
        if not sets or not sets[0].get("finalized_at"):
            return []
        ledger = sets[0]
        records = _rows(db.table("league_award_result_records").select("id,category_label,recipient_name,placement,metric_display")
                        .eq("club_id", club_id).eq("league_name", source_id).eq("public_visible", True)
                        .eq("workflow_revision", ledger["workflow_revision"]).eq("result_fingerprint", ledger["result_fingerprint"]).order("award_key"))
        return [{"id": row["id"], "title": row["category_label"], "recipient": row["recipient_name"],
                 "placement": row["placement"], "record": row.get("metric_display")} for row in records]
    if not source.get("public"):
        return []
    # Reuse the official public projections, including event visibility and
    # published podium rules for both standard and four-player team draws.
    from jupr_app.services.public_tournament_results_service import build_public_tournament_results
    from jupr_app.services.public_tournament_team_service import build_public_team_tournament_results
    result = build_public_tournament_results(db, club_id=club_id, tournament_id=source_id)
    honors = []
    for index, draw in enumerate(result.get("draws", [])):
        # Public podiums omit team IDs, and recipients may share a placement.
        for podium_index, podium in enumerate(draw.get("podium", [])):
            honors.append({"id": f"standard:{index}:{podium_index}",
                           "title": draw.get("division_name") or draw.get("name") or "Tournament podium",
                           "recipient": podium["team_name"], "placement": podium["placement"]})
    draws = _rows(db.table("tournament_event_draws").select("id,name").eq("tournament_id", source_id).eq("draw_kind", "TEAM_PARENT").eq("status", "published").order("id"))
    for draw in draws:
        team_results = build_public_team_tournament_results(db, club_id=club_id, tournament_id=source_id, draw_id=draw["id"])
        for podium in team_results.get("podium", []):
            honors.append({"id": f"team:{draw['id']}:{podium['team_id']}", "title": draw["name"],
                           "recipient": podium["team_name"], "placement": podium["placement"]})
    return honors


def _history_is_public(kind, source, honors):
    # Archiving removes a league from public results discovery, but its finalized,
    # public award records remain part of the series' permanent history.
    return bool(source.get("public") or (
        kind == "league" and source.get("complete")
        and str(source["event"].get("status", "")).strip().lower() == "archived"
        and honors
    ))


def event_history(db, *, club_id, kind, source_id, slug="", admin=False):
    if admin and not slug and kind != "interclub":
        clubs = db.table("clubs").select("slug").eq("id", club_id).limit(1).execute().data or []
        slug = clubs[0]["slug"] if clubs else ""
    source = load_source(db, club_id, kind, source_id)
    if not source:
        raise LookupError("This event history is not available.")
    current_honors = _honors(db, club_id, kind, source_id, source)
    if not admin and not _history_is_public(kind, source, current_honors):
        raise LookupError("This event history is not available.")
    current = source_summary(kind, source_id, source, slug)
    membership = db.table("pcs_event_editions").select("*").eq("club_id", club_id).eq("event_kind", kind).eq("source_id", source_id).limit(1).execute().data or []
    series = None
    if membership:
        series = db.table("pcs_event_series").select("id,name").eq("id", membership[0]["series_id"]).eq("club_id", club_id).limit(1).execute().data[0]
        editions = _rows(db.table("pcs_event_editions").select("source_id,label,position").eq("series_id", series["id"]).order("position", desc=True))
    else:
        editions = [{"source_id": source_id, "label": current["start_date"][:4] if current["start_date"] else current["name"], "position": 1}]
    seasons = []
    for edition in editions:
        item = source if edition["source_id"] == source_id else load_source(db, club_id, kind, edition["source_id"])
        if not item:
            continue
        honors = current_honors if edition["source_id"] == source_id else _honors(db, club_id, kind, edition["source_id"], item)
        if not admin and not _history_is_public(kind, item, honors):
            continue
        season = {**source_summary(kind, edition["source_id"], item, slug), "label": edition["label"],
                  "position": edition["position"], "selected": edition["source_id"] == source_id,
                  "honors": honors}
        if not admin:
            season.pop("admin_href"); season.pop("fingerprint")
        seasons.append(season)
    result = {"kind": kind, "series_id": series["id"] if series else None, "series_name": series["name"] if series else current["name"], "seasons": seasons}
    if admin:
        result.update(current=current, current_label=next(row["label"] for row in editions if row["source_id"] == source_id),
                      can_start=current["complete"] and editions[0]["source_id"] == source_id,
                      reason="Continue the later season from History." if editions[0]["source_id"] != source_id else
                             "Finish and publish this season before starting the next one." if not current["complete"] else "",
                      past_candidates=past_candidates(db, club_id, kind, source_id))
    return result


def past_candidates(db, club_id, kind, source_id):
    linked = {row["source_id"] for row in _rows(db.table("pcs_event_editions").select("source_id").eq("club_id", club_id).eq("event_kind", kind).order("source_id"))}
    if kind == "interclub":
        rows = _rows(db.table("pcs_interclub_seasons").select("id,details").eq("organizer_club_id", club_id).order("opened_at", desc=True))
        candidates = [{"source_id": row["id"], "name": row["details"]["name"]} for row in rows]
    elif kind == "league":
        rows = _rows(db.table("leagues_metadata").select("league_name,status").eq("club_id", club_id).in_("status", ["ended", "completed", "complete", "done", "archived"]).order("league_name"))
        candidates = [{"source_id": row["league_name"], "name": row["league_name"]} for row in rows]
    else:
        rows = _rows(db.table("tournaments").select("id,name").eq("club_id", club_id).in_("status", ["COMPLETED", "ARCHIVED"]).order("created_at", desc=True))
        candidates = [{"source_id": row["id"], "name": row["name"]} for row in rows]
    return [row for row in candidates if row["source_id"] != source_id and row["source_id"] not in linked]


def current_club_championships(db, trophies):
    season_ids = sorted({row["season_id"] for row in trophies})
    if not season_ids:
        return []
    linked = _rows(db.table("pcs_event_editions").select("series_id,source_id,position").eq("event_kind", "interclub").in_("source_id", season_ids).order("source_id"))
    series_ids = sorted({row["series_id"] for row in linked})
    if not series_ids:
        return [{**row, "is_current_champion": True} for row in trophies]
    editions = _rows(db.table("pcs_event_editions").select("series_id,source_id,position").in_("series_id", series_ids).order("series_id").order("position"))
    publications = _rows(db.table("pcs_interclub_publications").select("season_id,published").in_("season_id", [row["source_id"] for row in editions]).order("season_id"))
    completed = {row["season_id"] for row in publications if (row.get("published") or {}).get("season_complete")
                 and ((row.get("published") or {}).get("club_cup") or {}).get("status") == "complete"}
    defending = retained_champion_seasons(editions, completed)
    linked_ids = {row["source_id"] for row in linked}
    return [{**row, "is_current_champion": row["season_id"] not in linked_ids or row["season_id"] in defending} for row in trophies]
