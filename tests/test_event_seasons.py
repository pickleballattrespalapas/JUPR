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


@pytest.fixture
def league_award_history(monkeypatch):
    import jupr_app.services.event_season_service as service

    sources = {sid: {"event": {"league_name": sid, "league_type": "Individual", "status": status},
                     "fingerprint": "a" * 32, "complete": status == "archived", "public": status == "active"}
               for sid, status in [("private", "archived"), ("2026", "archived"), ("2027", "active")]}
    scope = {"club_id": "club", "league_name": "2026"}
    tables = {
        "pcs_event_editions": [{"series_id": "series", "club_id": "club", "event_kind": "league",
                                "source_id": sid, "label": sid, "position": position}
                               for position, sid in enumerate(sources)],
        "pcs_event_series": [{"id": "series", "club_id": "club", "name": "Club ladder"}],
        "leagues_metadata": [],
        # Storage order must not decide which workflow revision is published.
        "league_award_result_sets": [{**scope, "workflow_revision": revision, "result_fingerprint": str(revision),
                                      "finalized_at": "2026-09-30" if revision >= 4 else None}
                                     for revision in [3, 2, 1, 5, 4]],
        "league_award_result_records": [{**scope, "id": str(revision), "workflow_revision": revision,
                                         "result_fingerprint": str(revision), "public_visible": True,
                                         "award_key": "most_wins:1", "category_label": "Most Wins",
                                         "recipient_name": "Avery Ace", "placement": 1, "metric_display": "17 wins"}
                                        for revision in range(1, 6)],
    }
    published = tables["league_award_result_records"][-1]
    tables["league_award_result_records"] += [
        {**published, "id": "private-record", "public_visible": False},
        {**published, "id": "wrong-fingerprint", "result_fingerprint": "stale"},
        {**published, "id": "other-club", "club_id": "other"},
        {**published, "id": "other-league", "league_name": "other"},
    ]
    tables["league_award_result_sets"].append({**tables["league_award_result_sets"][3], "club_id": "other", "workflow_revision": 99})
    monkeypatch.setattr(service, "load_source", lambda db, club_id, kind, sid: sources.get(sid) if club_id == "club" else None)
    return SimpleNamespace(table=lambda name: Query(tables[name])), tables, sources


@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize("selected", ["2026", "2027"])
@pytest.mark.parametrize("league_type", ["Individual", "Team"])
def test_finalized_awards_survive_archival_in_both_season_histories(league_award_history, admin, selected, league_type):
    db, _tables, sources = league_award_history
    sources["2026"]["event"]["league_type"] = league_type
    history = event_history(db, club_id="club", kind="league", source_id=selected, slug="club", admin=admin)
    old = next(season for season in history["seasons"] if season["source_id"] == "2026")
    assert old["honors"] == [{"id": "5", "title": "Most Wins", "recipient": "Avery Ace", "placement": 1, "record": "17 wins"}]
    assert old["results_href"] is None  # Archival does not reopen the public results routes.
    assert old["selected"] == (selected == "2026")
    assert next(season for season in history["seasons"] if season["source_id"] == "2027")["honors"] == []
    if not admin:
        assert [season["source_id"] for season in history["seasons"]] == ["2027", "2026"]
        assert "admin_href" not in old and "fingerprint" not in old


@pytest.mark.parametrize("unpublished", ["preview", "private_records", "no_records", "draft"])
def test_history_does_not_publish_unfinished_or_private_awards(league_award_history, unpublished):
    db, tables, sources = league_award_history
    if unpublished == "preview":
        tables["league_award_result_sets"][3]["finalized_at"] = None
    elif unpublished == "private_records":
        for record in tables["league_award_result_records"]:
            record["public_visible"] = False
    elif unpublished == "no_records":
        tables["league_award_result_records"] = []
    else:
        sources["2026"].update(complete=False)
        sources["2026"]["event"]["status"] = "draft"
    with pytest.raises(LookupError):
        event_history(db, club_id="club", kind="league", source_id="2026", slug="club")
    history = event_history(db, club_id="club", kind="league", source_id="2027", slug="club")
    assert [season["source_id"] for season in history["seasons"]] == ["2027"]
    assert history["seasons"][0]["honors"] == []


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
