"""Recent gold medalists from closed, publicly visible club tournaments."""
from __future__ import annotations

from calendar import monthrange
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlencode

from jupr_app.domain.tournament_registration_repo import get_public_tournament_bundle
from jupr_app.services.public_tournament_results_service import build_public_tournament_results
from jupr_app.services.public_tournament_team_service import build_public_team_tournament_results


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (TypeError, ValueError):
        return None


def _month_after(value):
    year, month = (value.year + 1, 1) if value.month == 12 else (value.year, value.month + 1)
    return value.replace(year=year, month=month, day=min(value.day, monthrange(year, month)[1]))


def _rows(query):
    rows = []
    seen = set()
    while True:
        page = query.range(len(rows), len(rows) + 999).execute().data or []
        if not page:
            return rows
        ids = {str(row["id"]) for row in page}
        if len(ids) != len(page) or seen.intersection(ids):
            raise RuntimeError("Tournament highlights could not load a complete result set.")
        seen.update(ids)
        rows.extend(page)


def _division(draw):
    family = str(draw.get("event_family_label") or "").strip()
    division = str(draw.get("division_name") or "").strip()
    if family and division and family.casefold() != division.casefold():
        return f"{family} · {division}"
    return division or family or draw.get("name") or "Tournament gold"


def public_tournament_gold_highlights(db, *, club_id, slug, now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    # Every calendar month is at most 31 days. Only immutable completion
    # receipts start the window; edits and archive/unarchive cannot extend it.
    receipts = _rows(
        db.table("tournament_lifecycle_receipts").select("id,tournament_id,created_at")
        .eq("club_id", club_id).eq("action", "complete").eq("to_status", "COMPLETED")
        .gte("created_at", (now - timedelta(days=31)).isoformat()).order("created_at").order("id")
    )
    completed = {}
    for receipt in receipts:
        closed_at = _timestamp(receipt.get("created_at"))
        if closed_at is not None and closed_at <= now < _month_after(closed_at):
            tournament_id = str(receipt["tournament_id"])
            completed[tournament_id] = min(completed.get(tournament_id, closed_at), closed_at)

    highlights = []
    for tournament_id, closed_at in completed.items():
        tournament, _settings, _days, events = get_public_tournament_bundle(
            db, club_id=club_id, tournament_id=tournament_id,
        )
        if not tournament or str(tournament.get("status") or "").upper() != "COMPLETED":
            continue
        common = {
            "tournament_name": tournament.get("name") or "Tournament",
            "completed_at": closed_at.isoformat(),
            "expires_at": _month_after(closed_at).isoformat(),
        }
        prefix = f"/clubs/{quote(slug, safe='')}"
        result = build_public_tournament_results(db, club_id=club_id, tournament_id=tournament_id)
        for draw in result.get("draws", []):
            if draw.get("state") != "COMPLETE":
                continue
            href = prefix + "/tournament-results?" + urlencode({
                "tournament_id": tournament_id, "view": "past", "tab": "completed",
                "draw": draw["public_draw_key"],
            })
            for index, podium in enumerate(draw.get("podium", [])):
                if podium.get("placement") == 1:
                    highlights.append({
                        **common, "id": f"{tournament_id}:{draw['public_draw_key']}:{index}",
                        "division": _division(draw), "recipient": podium["team_name"],
                        "players": [], "results_href": href,
                    })

        public_events = {str(event["id"]) for event in events if event.get("enabled", True)
                         and str(event.get("status") or "").lower() not in {"cancelled", "canceled", "disabled"}}
        team_draws = _rows(
            db.table("tournament_event_draws").select("id,event_option_id")
            .eq("tournament_id", tournament_id).eq("draw_kind", "TEAM_PARENT")
            .eq("status", "published").order("id")
        )
        for draw in team_draws:
            if str(draw.get("event_option_id") or "") not in public_events:
                continue
            result = build_public_team_tournament_results(
                db, club_id=club_id, tournament_id=tournament_id, draw_id=draw["id"],
            )
            teams = {str(team["id"]): team for team in result.get("teams", [])}
            for podium in result.get("podium", []):
                if podium.get("placement") != 1:
                    continue
                team_id = str(podium["team_id"])
                highlights.append({
                    **common, "id": f"{tournament_id}:{draw['id']}:{team_id}",
                    "division": _division(result["draw"]), "recipient": podium["team_name"],
                    "players": [member["display_name"] for member in teams.get(team_id, {}).get("members", [])],
                    "results_href": f"{prefix}/tournament-team-results/{quote(tournament_id, safe='')}/{quote(str(draw['id']), safe='')}",
                })
    # Stable alphabetical order within each tournament, newest closeout first.
    highlights.sort(key=lambda row: (row["tournament_name"].casefold(), row["division"].casefold(), row["recipient"].casefold(), row["id"]))
    highlights.sort(key=lambda row: row["completed_at"], reverse=True)
    return highlights
