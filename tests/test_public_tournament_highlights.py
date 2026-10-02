from copy import deepcopy
from datetime import datetime, timezone
import json
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.services.public_tournament_highlights_service import public_tournament_gold_highlights
from tests.test_public_tournament_registration_service import FakeQuery, FakeSupabase
from tests.test_public_tournament_results_service import _results_storage


class Query(FakeQuery):
    def gte(self, key, value):
        self.filters.append(("gte", key, value))
        return self

    def _apply_filters(self, rows):
        rows = super()._apply_filters(rows)
        for op, key, value in self.filters:
            if op == "gte":
                rows = [row for row in rows if str(row.get(key) or "") >= value]
        return rows


class DB(FakeSupabase):
    def table(self, name):
        return Query(self.storage, name)


@pytest.fixture
def completed():
    tables = _results_storage()
    tables["tournaments"][0].update(status="COMPLETED", updated_at="2026-10-20T12:00:00Z")
    tables["tournament_lifecycle_receipts"] = [{
        "id": "closeout", "club_id": "club-1", "tournament_id": "t1", "action": "complete",
        "to_status": "COMPLETED", "created_at": "2026-09-30T18:00:00+00:00",
        "evidence_json": {"private": "do not publish"},
    }]
    tables["tournament_teams"][0]["player2_id"] = 12
    tables["players"].append({"id": 12, "club_id": "club-1", "name": "Casey Counter", "email": "private@example.test"})
    return DB(tables), tables


def highlights(db, at="2026-10-02T12:00:00+00:00", **kwargs):
    return public_tournament_gold_highlights(db, club_id="club-1", slug="club one", now=datetime.fromisoformat(at), **kwargs)


def test_closed_tournament_gold_links_to_its_public_completed_draw(completed):
    db, tables = completed
    before = deepcopy(tables)
    rows = highlights(db)
    assert len(rows) == 1
    assert rows[0]["recipient"] == "Alex Ace / Casey Counter"
    assert rows[0]["completed_at"] == "2026-09-30T18:00:00+00:00"
    assert rows[0]["expires_at"] == "2026-10-30T18:00:00+00:00"
    link = urlparse(rows[0]["results_href"])
    assert link.path == "/clubs/club%20one/tournament-results"
    assert parse_qs(link.query) == {"tournament_id": ["t1"], "view": ["past"], "tab": ["completed"], "draw": [rows[0]["id"].split(":")[1]]}
    assert "Blair Backhand" not in json.dumps(rows)  # Silver is not a homepage gold.
    assert not any(secret in json.dumps(rows) for secret in ["private@example", "evidence_json", "team-private", "podium-private", "admin_notes"])
    assert not db.rpc_calls
    assert all(tables[key] == value for key, value in before.items())


@pytest.mark.parametrize("closed,expires", [
    ("2026-09-30T18:00:00+00:00", "2026-10-30T18:00:00+00:00"),
    ("2026-01-31T18:00:00+00:00", "2026-02-28T18:00:00+00:00"),
    ("2028-01-31T18:00:00+00:00", "2028-02-29T18:00:00+00:00"),
    ("2026-12-31T18:00:00+00:00", "2027-01-31T18:00:00+00:00"),
])
def test_one_calendar_month_including_exact_expiry(completed, closed, expires):
    db, tables = completed
    tables["tournament_lifecycle_receipts"][0]["created_at"] = closed
    assert highlights(db, closed)[0]["expires_at"] == expires
    before_expiry = datetime.fromisoformat(expires).replace(second=0, microsecond=0)
    from datetime import timedelta
    assert highlights(db, (before_expiry - timedelta(microseconds=1)).isoformat())
    assert highlights(db, expires) == []


@pytest.mark.parametrize("hidden", ["active", "draft", "archived", "unpublished", "missing_receipt", "other_club", "future", "invalid_time"])
def test_nonpublic_or_unclosed_tournaments_do_not_appear(completed, hidden):
    db, tables = completed
    if hidden in {"active", "draft", "archived"}:
        tables["tournaments"][0]["status"] = hidden.upper()
    elif hidden == "unpublished":
        tables["tournament_registration_settings"][0]["builder_draft_json"] = {}
    elif hidden == "missing_receipt":
        tables["tournament_lifecycle_receipts"] = []
    elif hidden == "other_club":
        tables["tournaments"][0]["club_id"] = "other-club"
    else:
        tables["tournament_lifecycle_receipts"][0]["created_at"] = "2026-11-01T18:00:00+00:00" if hidden == "future" else "not-a-date"
    assert highlights(db) == []


def test_archiving_and_unarchiving_do_not_restart_the_window(completed):
    db, tables = completed
    original = tables["tournament_lifecycle_receipts"][0]
    tables["tournament_lifecycle_receipts"] += [
        {**original, "id": "archive", "action": "archive", "to_status": "ARCHIVED", "created_at": "2026-10-29T18:00:00+00:00"},
        {**original, "id": "unarchive", "action": "unarchive", "created_at": "2026-10-30T18:00:00+00:00"},
    ]
    assert highlights(db, "2026-10-31T12:00:00+00:00") == []


def test_singles_shared_gold_and_four_player_winners(completed):
    db, tables = completed
    tables["tournament_teams"][0].pop("player2_id")
    tables["tournament_podium"][1]["placement"] = 1
    tables["tournament_event_draws"].append({"id": "team-draw", "tournament_id": "t1", "event_option_id": "team-event",
        "draw_kind": "TEAM_PARENT", "status": "published", "name": "Team Cup"})
    tables["tournament_event_options"].append({**tables["tournament_event_options"][0], "id": "team-event", "competition_format": "FOUR_PLAYER_TEAM"})
    tables["tournament_four_player_teams"] = [{"id": "team4", "tournament_id": "t1", "draw_id": "team-draw", "name": "Team champions",
        "status": "CONFIRMED", "eligibility_state": "ELIGIBLE"}]
    tables["tournament_four_player_team_members"] = [{"id": str(i), "team_id": "team4", "tournament_id": "t1", "slot": str(i),
        "status": "ACCEPTED", "display_name_snapshot": name, "invited_email": "private@example.test"}
        for i, name in enumerate(["Alex Ace", "Blair Backhand", "Casey Counter", "Drew Dink"])]
    tables["tournament_four_player_team_members"].append({"id": "removed", "team_id": "team4", "tournament_id": "t1", "status": "REMOVED", "display_name_snapshot": "Removed player"})
    tables["tournament_four_player_podium"] = [{"tournament_id": "t1", "draw_id": "team-draw", "team_id": "team4", "placement": 1, "published_at": "2026-09-30"}]
    rows = highlights(db)
    assert {row["recipient"] for row in rows} == {"Alex Ace", "Blair Backhand", "Team champions"}
    assert len({row["id"] for row in rows}) == 3
    team = next(row for row in rows if row["recipient"] == "Team champions")
    assert team["players"] == ["Alex Ace", "Blair Backhand", "Casey Counter", "Drew Dink"]
    assert team["results_href"] == "/clubs/club%20one/tournament-team-results/t1/team-draw"
    tables["tournament_four_player_podium"][0]["published_at"] = None
    assert "Team champions" not in json.dumps(highlights(db))
    tables["tournament_four_player_podium"][0]["published_at"] = "2026-09-30"
    tables["tournament_event_options"][-1]["enabled"] = False
    assert "Team champions" not in json.dumps(highlights(db))


def test_receipts_are_club_scoped_and_paginate_through_capped_pages(completed, monkeypatch):
    db, tables = completed
    receipt = tables["tournament_lifecycle_receipts"][0]
    tables["tournament_lifecycle_receipts"] = [{**receipt, "id": f"{i:03}", "tournament_id": "missing"} for i in range(5)] + [receipt]
    tables["tournament_lifecycle_receipts"].append({**receipt, "id": "other", "club_id": "other", "created_at": "2026-10-01T18:00:00+00:00"})
    execute = Query.execute
    def capped(query):
        response = execute(query)
        if query.table_name == "tournament_lifecycle_receipts" and query.limit_count != 1:
            response.data = response.data[:2]
        return response
    monkeypatch.setattr(Query, "execute", capped)
    assert highlights(db)[0]["completed_at"] == receipt["created_at"]


def test_highlights_api_uses_published_club_visibility_and_no_cache(completed, monkeypatch):
    from services.api import public_tournament_results_routes as routes
    from services.api.club_site_models import SiteDocument
    db, tables = completed
    tables["clubs"] = [{"id": "club-1", "slug": "club", "is_active": True, "public_site_status": "published"}]
    document = SiteDocument(name="Test club").model_dump()
    tables["pcs_club_sites"] = [{"club_id": "club-1", "published": document, "published_at": "2026-09-01"}]
    calls = []
    def current(db, **scope):
        calls.append(scope)
        return public_tournament_gold_highlights(db, now=datetime(2026, 10, 2, tzinfo=timezone.utc), **scope)
    monkeypatch.setattr(routes, "public_tournament_gold_highlights", current)
    app = FastAPI()
    routes.install_public_tournament_results_routes(app, get_club=lambda slug: {}, get_supabase_client=lambda: db, public_club_payload=lambda *_: {})
    client = TestClient(app)
    result = client.get("/public/clubs/club/tournament-highlights")
    assert result.status_code == 200 and len(result.json()["highlights"]) == 1
    assert result.headers["cache-control"] == "no-store"
    assert calls == [{"club_id": "club-1", "slug": "club"}]
    document["page_visibility"] = {"tournaments": "private"}
    assert client.get("/public/clubs/club/tournament-highlights").json() == {"highlights": []}
    assert len(calls) == 1
    assert client.get("/public/clubs/missing/tournament-highlights").status_code == 404
    tables["clubs"][0]["public_site_status"] = "draft"
    assert client.get("/public/clubs/club/tournament-highlights").status_code == 404
