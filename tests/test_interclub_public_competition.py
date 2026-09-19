"""Public league snapshots cannot expose private data or unofficial scores."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.api import interclub_public_routes as publication
from jupr_app.domain import interclub_competition as engine


def official_document():
    entries = []
    for club in ("alpha", "beta"):
        entries.append({"division": "3.5", "club_id": club, "roster": [
            {"entry_id": f"{club}-{index}", "name": f"Player {index}", "gender": "female" if index < 2 else "male"}
            for index in range(4)
        ]})
    document = engine.generate_round_robin("meet", entries, played_at="2027-01-16T17:00:00Z", courts=2)
    for encounter in document["encounters"]:
        for pairing in encounter["pairings"]:
            for game in pairing["games"]:
                game.update(status="completed", a=11, b=7)
    return document


def test_publication_uses_full_approved_pairings_and_strips_private_fields(monkeypatch):
    document = official_document()
    document["internal_email"] = "private@example.invalid"
    # The route reads canonical approved documents, then whitelists public data.
    canonical = deepcopy(document)
    canonical.pop("internal_email")
    monkeypatch.setattr(publication, "approved_documents", lambda db, sid: [canonical])
    season = {"id": "season", "details": {"name": "Southern BCS", "start_date": "2027-01-01", "end_date": "2027-03-31", "timezone": "America/Mazatlan", "divisions": ["3.5"]}}
    clubs = [{"id": "alpha", "name": "Alpha"}, {"id": "beta", "name": "Beta"}]
    meets = [{"id": "meet", "host_club_id": "alpha", "club_ids": ["alpha", "beta"]}]
    snapshot = publication.competition_publication(None, season, clubs, meets)
    response = publication.publication_response(snapshot)
    assert snapshot["scoring_version"] == 1
    assert snapshot["results"] == []  # Never infer legacy one-pairing results.
    assert len(snapshot["competition_results"][0]["pairings"]) == 2
    leader = response["standings"][0]["rows"][0]
    assert leader["club_id"] == "alpha" and leader["points"] == 3 and leader["pairings_won"] == 2
    assert leader["games_won"] == 6
    assert {row["club_id"] for row in response["standings"][0]["rows"]} == {"alpha", "beta"}
    assert {row["club_id"] for row in response["club_cup"]["standings"]} == {"alpha", "beta"}
    assert "private@example.invalid" not in str(response)
    assert "alpha-0" not in str(response) and "players_a" not in str(response)
    assert "injury_reason" not in str(response) and "ratings_error" not in str(response)
    canonical["encounters"][0]["pairings"][0]["games"][0]["a"] = 99
    assert snapshot["competition_results"][0]["pairings"][0]["games"][0]["a"] == 11


@pytest.fixture
def publication_client(monkeypatch):
    sid = str(uuid4())
    season = {"id": sid, "organizer_club_id": "alpha", "details": {"name": "Southern BCS", "start_date": "2027-01-01", "end_date": "2027-03-31", "timezone": "America/Mazatlan", "divisions": ["3.5"]}}
    clubs = [{"id": "alpha", "name": "Alpha"}, {"id": "beta", "name": "Beta"}]
    meets = [{"id": "meet", "host_club_id": "alpha", "club_ids": ["alpha", "beta"], "starts_at": "2027-01-16T17:00:00Z", "revision": 2}]
    approved = {"id": str(uuid4()), "approved_revision": 4, "approved_document": official_document()}
    state = {"batches": [approved], "publication": {"revision": 1, "draft": {"results": []}, "published": None}, "calls": [], "error": None}

    class Query:
        def __init__(self, rows): self.rows = rows
        def select(self, *args, **kwargs): return self
        def eq(self, *args): return self
        def limit(self, *args): return self
        def execute(self): return SimpleNamespace(data=deepcopy(self.rows))

    def rpc(name, args):
        state["calls"].append((name, args))
        def execute():
            if state["error"]:
                error = RuntimeError("private database detail"); error.code = state["error"]; raise error
            return SimpleNamespace(data={"revision": 2, "draft": {"results": []}, "published": args.get("p_document")})
        return SimpleNamespace(execute=execute)

    db = SimpleNamespace(table=lambda name: Query(state["batches"] if name == "pcs_interclub_competition_batches" else [state["publication"]]), rpc=rpc)
    monkeypatch.setattr(publication, "season_context", lambda db, sid: (season, clubs, meets))
    monkeypatch.setattr(publication, "site_administrator", lambda *args: SimpleNamespace(user_id=str(uuid4()), email="organizer@example.invalid"))
    app = FastAPI(); publication.install_interclub_public_routes(app, get_supabase_client=lambda: db)
    state.update(season=season, clubs=clubs, meets=meets, db=db)
    return TestClient(app), f"/admin/clubs/alpha/interclub/{sid}/publication", state


def test_publish_uses_reviewed_sources_and_does_not_send_private_metadata(publication_client):
    client, path, state = publication_client
    preview = client.get(path).json()
    result = client.post(path + "/publish", json={"revision": 1, "preview_fingerprint": preview["preview_fingerprint"]})
    assert result.status_code == 200
    name, params = state["calls"][-1]
    assert name == "pcs_publish_reviewed_interclub_publication"
    assert params["p_sources"]["approved"] == [{"id": state["batches"][0]["id"], "revision": 4}]
    assert params["p_document"] == preview["preview"]["document"]
    assert "revision" not in params["p_document"]["meets"][0]


@pytest.mark.parametrize("change", ["approval", "meet", "club", "season", "new_batch"])
def test_changed_preview_requires_review_before_any_publish_rpc(publication_client, change):
    client, path, state = publication_client
    fingerprint = client.get(path).json()["preview_fingerprint"]
    if change == "approval": state["batches"][0]["approved_revision"] += 1
    elif change == "meet": state["meets"][0]["revision"] += 1
    elif change == "club": state["clubs"][0]["name"] = "Renamed club"
    elif change == "season": state["season"]["details"]["name"] = "Renamed season"
    else:
        added = deepcopy(state["batches"][0]); added["id"] = str(uuid4())
        added["approved_document"]["meet_id"] = "second-meet"
        state["batches"].append(added)
    response = client.post(path + "/publish", json={"revision": 1, "preview_fingerprint": fingerprint})
    assert response.status_code == 409 and "review" in response.text
    assert state["calls"] == []


def test_missing_review_token_rejected_but_unpublish_remains_available(publication_client):
    client, path, state = publication_client
    assert client.post(path + "/publish", json={"revision": 1}).status_code == 409
    assert state["calls"] == []
    assert client.post(path + "/unpublish", json={"revision": 1}).status_code == 200
    assert state["calls"][-1][0] == "pcs_write_interclub_publication"


def test_change_between_api_read_and_database_commit_fails_closed(publication_client):
    client, path, state = publication_client
    preview = client.get(path).json()
    state["error"] = "40001"
    response = client.post(path + "/publish", json={"revision": 1, "preview_fingerprint": preview["preview_fingerprint"]})
    assert response.status_code == 409
    assert "private database detail" not in response.text
