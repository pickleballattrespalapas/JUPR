"""Public trophy projection. Only honors matching the published results are visible."""
from __future__ import annotations


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
    rows = []
    while True:
        page = query.range(len(rows), len(rows) + 999).execute().data or []
        rows.extend(page)
        if len(page) < 1000:
            break
    return [{**{key: row[key] for key in ("id", "season_id", "club_id", "award_key", "division", "title",
                                         "recipient_type", "recipient_name", "season_name", "earned_at")},
             "recipient_key": row.get("entry_id") or f"club:{row['club_id']}",
             "results_href": f"/interclub/{row['season_id']}/final-results"} for row in rows]


def player_interclub_trophies(db, *, club_id: str, player_id):
    return [{"badge_id": row["id"], "title": row["title"], "placement": 1 if row["award_key"] != "participation" else None,
             "context_type": "interclub", "context_label": row["season_name"], "earned_at": row["earned_at"],
             "results_href": row["results_href"], "award_key": row["award_key"]}
            for row in public_interclub_trophies(db, club_id=club_id, player_id=player_id)]
