"""Public trophy projection. Only honors matching the published results are visible."""
from __future__ import annotations

from jupr_app.domain.interclub_awards import PERFORMANCE_BADGES, award_title, performance_awards


def _all(query):
    rows = []
    while True:
        page = query.range(len(rows), len(rows) + 999).execute().data or []
        rows.extend(page)
        if len(page) < 1000:
            return rows


def public_interclub_trophies(db, *, club_id: str | None = None, player_id=None, season_id: str | None = None):
    query = db.table("pcs_public_interclub_awards").select(
        "id,season_id,club_id,entry_id,award_key,division,title,recipient_type,recipient_name,season_name,earned_at")
    if club_id is not None:
        query = query.eq("club_id", club_id)
    if player_id is not None:
        query = query.eq("player_id", player_id)
    elif season_id is None:
        query = query.eq("recipient_type", "club")
    if season_id is not None:
        query = query.eq("season_id", season_id)
    query = query.order("earned_at", desc=True).order("award_key").order("id")
    rows = _all(query)
    return [{**{key: row[key] for key in ("id", "season_id", "club_id", "award_key", "division", "title",
                                         "recipient_type", "recipient_name", "season_name", "earned_at")},
             "title": award_title(row["award_key"], row["season_name"], row["division"]),
             "recipient_key": row.get("entry_id") or f"club:{row['club_id']}",
             "results_href": f"/interclub/{row['season_id']}/final-results"} for row in rows
            if row["recipient_type"] != "club" or row["award_key"] != "participation"]


def public_interclub_achievements(db, *, club_id=None, player_id=None, season_id=None, publication=None):
    """The published score snapshot is the ledger for repeatable achievements.

    This also covers previously published meets without rewriting their results.
    Private corrections cannot change badges; a republish reconciles them at once.
    """
    entries = None
    if season_id is not None:
        publications = ([publication] if publication is not None else
                        db.table("pcs_interclub_publications").select("season_id,published,published_at")
                        .eq("season_id", season_id).limit(1).execute().data or [])
    elif player_id is not None and club_id is not None:
        rows = _all(db.table("pcs_interclub_entries").select("id,season_id").eq("club_id", club_id)
                    .eq("player_id", player_id).order("id"))
        entries = {row["id"] for row in rows}
        seasons = sorted({row["season_id"] for row in rows})
        publications = []
        for offset in range(0, len(seasons), 200):
            publications.extend(_all(db.table("pcs_interclub_publications").select("season_id,published,published_at")
                                     .in_("season_id", seasons[offset:offset+200]).order("season_id")))
    else:
        return []
    result = []
    for row in publications:
        document = row.get("published")
        if not document:
            continue
        sid = row["season_id"]
        for award in performance_awards(sid, document, entry_ids=entries):
            if (entries is not None and award["entry_id"] not in entries) or (club_id is not None and award["club_id"] != club_id):
                continue
            entry = award.pop("entry_id")
            result.append({**award, "season_id": sid, "season_name": document.get("name", "Inter-Club"),
                           "earned_at": award.get("earned_at") or row.get("published_at"), "recipient_key": entry,
                           "results_href": (f"/interclub/{sid}/final-results" if document.get("season_complete")
                                            and award["award_key"] == "participation" else f"/interclub/{sid}?view=results")})
    return result


def public_interclub_honors(db, **scope):
    publication = scope.pop("publication", None)
    trophies = public_interclub_trophies(db, **scope)
    achievements = public_interclub_achievements(db, **scope, publication=publication)
    return sorted({row["id"]: row for row in [*trophies, *achievements]}.values(), key=lambda row: row["id"])


def player_interclub_awards(db, *, club_id: str, player_id):
    rows = public_interclub_honors(db, club_id=club_id, player_id=player_id)
    badges = []
    for kind, definition in PERFORMANCE_BADGES.items():
        earned = [row for row in rows if row["award_key"] == kind]
        if not earned:
            continue
        earned.sort(key=lambda row: (row.get("earned_at") or "", row["id"]), reverse=True)
        badges.append({"badge_id": f"interclub_{kind}", "name": definition["title"], "category": "Inter-Club",
                       "prestige": 0, "rarity": None, "icon_key": "medal", "description": definition["requirement"],
                       "requirements": definition["requirement"], "count": len(earned), "last_earned_at": earned[0]["earned_at"],
                       "achievements": [{"id": row["id"], "earned_at": row["earned_at"],
                                         "detail": f"{row['season_name']} · {row['detail']}", "results_href": row["results_href"]}
                                        for row in earned]})
    return {"badges": badges, "trophies": _player_trophies(rows)}


def player_interclub_trophies(db, *, club_id: str, player_id):
    return _player_trophies(public_interclub_honors(db, club_id=club_id, player_id=player_id))


def _player_trophies(rows):
    return [{"badge_id": row["id"], "title": row["title"], "placement": 1 if row["award_key"] != "participation" else None,
             "context_type": "interclub", "context_label": row["season_name"], "earned_at": row["earned_at"],
             "results_href": row["results_href"], "award_key": row["award_key"]}
            for row in rows if row["award_key"] not in PERFORMANCE_BADGES]
