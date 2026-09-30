from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.domain.event_seasons import new_season_template, next_event_id, retained_champion_seasons
from jupr_app.services.event_season_service import current_club_championships, event_history
from services.api import event_season_routes as routes


def test_interclub_new_season_keeps_rules_but_has_fresh_signups_and_meets():
    source = {"event": {"rules": {"3.5": {"max_rating": 3.75}}},
              "setup": {"name": "2026", "club_ids": ["a", "b"], "divisions": ["3.5"],
               "registration_rules": {"3.5": {"max_rating": 3.999}}, "meets": [{"starts_at": "2026-01-01"}]}}
    before = deepcopy(source)
    result = new_season_template("interclub", source, name="2027", start_date="2027-01-01", end_date="2027-03-01", new_id=str(uuid4()))
    assert result["club_ids"] == ["a", "b"] and result["registration_rules"] == source["event"]["rules"]
    assert result["meets"] == [] and result["setup_step"] == 0 and result["name"] == "2027"
    assert source == before


def test_league_template_preserves_schedule_rules_not_old_sessions():
    source = {"event": {"schedule_config": {"start_date": "2026-01-01", "end_date": "2026-03-01", "weeks": 8,
                        "weekday": "Sunday", "sessions": [{"score": 11}], "skip_dates": ["2026-02-01"]}}}
    result = new_season_template("league", source, name="2027", start_date="2027-01-01", end_date="2027-03-01", new_id="unused")
    assert result == {"schedule_config": {"start_date": "2027-01-01", "end_date": "2027-03-01", "weeks": 8, "weekday": "Sunday"}}


def test_tournament_rollover_remaps_all_references_and_resets_publication_and_sponsors():
    source = {"settings": {"location_name": "Club", "sponsors_json": [{"logo": "old-private-asset"}],
              "builder_draft_json": {"published_at": "2026-01-01", "published_event_families": [{"id": "family", "day_ids": ["day"], "label": "Mixed"}]}},
              "days": [{"id": "day", "tournament_id": "old", "event_date": "2026-01-01"}, {"id": "day2", "event_date": "2026-01-03"}],
              "divisions": [{"id": "division", "tournament_id": "old", "family_id": "family", "registration_day_id": "day", "scheduled_day_ids": ["day", "day2"], "skill_label": "3.5"}]}
    result = new_season_template("tournament", source, name="Classic 2027", start_date="2027-02-01", end_date="2027-02-03", new_id="new")
    assert [day["event_date"] for day in result["days"]] == ["2027-02-01", "2027-02-03"]
    assert result["divisions"][0]["registration_day_id"] == result["days"][0]["id"] != "day"
    assert result["divisions"][0]["scheduled_day_ids"] == [day["id"] for day in result["days"]]
    assert result["divisions"][0]["family_id"] == result["event_families"][0]["id"] != "family"
    assert result["published_at"] is None and result["published_event_families"] == []
    assert result["basics"]["sponsors_json"] == [] and result["settings"]["registration_status"] == "draft"
    assert result["divisions"][0]["tournament_id"] == "new"
    with pytest.raises(ValueError, match="enough days"):
        new_season_template("tournament", source, name="Classic", start_date="2027-02-01", end_date="2027-02-02", new_id="new")


def test_defending_championships_only_expire_after_a_later_edition_finishes():
    editions = [{"series_id": "coastal", "source_id": str(year), "position": i} for i, year in enumerate([2025, 2026, 2027])]
    editions += [{"series_id": "other", "source_id": "separate", "position": 0}]
    assert retained_champion_seasons(editions, {"2025", "separate"}) == {"2025", "separate"}
    assert retained_champion_seasons(editions, {"2025", "2026", "separate"}) == {"2026", "separate"}
    assert retained_champion_seasons(editions, {"2025", "2026", "2027", "separate"}) == {"2027", "separate"}


class Query:
    def __init__(self, rows): self.rows = deepcopy(rows); self.bounds = None
    def select(self, *args, **kwargs): return self
    def eq(self, key, value): self.rows = [r for r in self.rows if r.get(key) == value]; return self
    def in_(self, key, values): self.rows = [r for r in self.rows if r.get(key) in values]; return self
    def order(self, key, desc=False): self.rows.sort(key=lambda row: row.get(key, ""), reverse=desc); return self
    def limit(self, value): self.rows = self.rows[:value]; return self
    def range(self, start, end): self.bounds = (start, end); return self
    def execute(self): return SimpleNamespace(data=self.rows if self.bounds is None else self.rows[self.bounds[0]:self.bounds[1]+1])


def test_retiring_home_cards_preserves_history_and_is_independent_of_which_club_won():
    tables = {"pcs_event_editions": [{"event_kind": "interclub", "series_id": "series", "source_id": sid, "position": i} for i, sid in enumerate(["old", "next"])],
              "pcs_interclub_publications": [{"season_id": "old", "published": {"season_complete": True, "club_cup": {"status": "complete"}}},
                                            {"season_id": "next", "published": {"season_complete": False}}]}
    db = SimpleNamespace(table=lambda name: Query(tables[name]))
    trophies = [{"season_id": "old", "club_id": "old-winner"}, {"season_id": "unlinked", "club_id": "old-winner"}]
    assert all(row["is_current_champion"] for row in current_club_championships(db, trophies))
    tables["pcs_interclub_publications"][1]["published"].update(season_complete=True, club_cup={"status": "complete", "champions": ["another-club"]})
    results = current_club_championships(db, trophies)
    assert len(results) == 2 and not results[0]["is_current_champion"] and results[1]["is_current_champion"]
    assert trophies == [{"season_id": "old", "club_id": "old-winner"}, {"season_id": "unlinked", "club_id": "old-winner"}]


def test_public_history_hides_drafts_and_private_seasons(monkeypatch):
    import jupr_app.services.event_season_service as service
    tables = {"pcs_event_editions": [{"series_id": "series", "club_id": "club", "event_kind": "league", "source_id": sid, "label": sid, "position": i} for i, sid in enumerate(["2026", "secret 2027"])],
              "pcs_event_series": [{"id": "series", "club_id": "club", "name": "Club ladder"}]}
    source = lambda sid: {"event": {"league_name": sid, "status": "ended" if sid == "2026" else "draft"}, "fingerprint": "a"*32,
                          "complete": sid == "2026", "public": sid == "2026"}
    monkeypatch.setattr(service, "load_source", lambda db, club_id, kind, sid: source(sid) if club_id == "club" else None)
    monkeypatch.setattr(service, "_honors", lambda *args: [])
    db = SimpleNamespace(table=lambda name: Query(tables[name]))
    history = event_history(db, club_id="club", kind="league", source_id="2026", slug="club")
    assert [row["name"] for row in history["seasons"]] == ["2026"]
    assert "fingerprint" not in history["seasons"][0] and "admin_href" not in history["seasons"][0]
    with pytest.raises(LookupError): event_history(db, club_id="club", kind="league", source_id="secret 2027")
    with pytest.raises(LookupError): event_history(db, club_id="other", kind="league", source_id="2026")


def test_api_rollover_uses_scoped_source_and_a_reviewed_atomic_rpc(monkeypatch):
    calls = []
    source = {"event": {"league_name": "2026", "league_type": "Individual", "schedule_config": {}}, "fingerprint": "a"*32}
    monkeypatch.setattr(routes, "site_administrator", lambda *args: SimpleNamespace(user_id=str(uuid4()), email="qa@example.invalid"))
    monkeypatch.setattr(routes, "load_source", lambda db, club, kind, source_id: source if club == "club" and source_id == "2026" else None)
    monkeypatch.setattr(routes, "_write_guard", lambda *args: None)
    db = SimpleNamespace(rpc=lambda fn, args: calls.append((fn, args)) or SimpleNamespace(execute=lambda: SimpleNamespace(data={"source_id": "2027", "kind": "league"})))
    app = FastAPI(); routes.install_event_season_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    body = {"request_id": str(uuid4()), "fingerprint": "a"*32, "series_name": "Sunday Ladder", "current_label": "2026", "label": "2027", "name": "2027", "start_date": "2027-01-01", "end_date": "2027-03-01"}
    path = "/admin/clubs/club/event-seasons/start?kind=league&event=2026"
    response = client.post(path, json=body)
    assert response.status_code == 200 and "league=2027" in response.json()["admin_href"]
    fn, args = calls[-1]
    assert fn == "pcs_start_event_season" and args["p_club_id"] == "club"
    assert args["p_input"]["fingerprint"] == body["fingerprint"]
    assert args["p_new_id"] == next_event_id("club", body["request_id"])
    assert client.post(path.replace("/club/", "/other/"), json=body).status_code == 404
    assert client.post(path, json={**body, "end_date": "2026-01-01"}).status_code == 422
    assert client.post(path, json={**body, "copied_scores": []}).status_code == 422
    assert len(calls) == 1
