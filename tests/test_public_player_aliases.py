from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import Response

from jupr_app.domain.player_aliases import public_alias_projection
from jupr_app.services.public_player_service import build_public_player_directory
from services.api.public_alias_middleware import install_public_alias_middleware
from tests.test_public_player_service import FakeSupabase


def test_alias_search_and_profile_id_preserve_canonical_identity():
    rows = {"players": [{"id": 7, "club_id": "club", "name": "Carol Becker",
                         "public_alias": "Ann Johnson", "rating": 1400, "matches_played": 3}]}
    db = FakeSupabase()
    db.rows_by_table = rows
    result = build_public_player_directory(db, club_id="club", search="Ann Johnson")
    assert [(p["id"], p["name"]) for p in result["players"]] == [(7, "Ann Johnson")]
    assert build_public_player_directory(db, club_id="club", search="Carol Becker")["players"] == []
    assert rows["players"][0]["name"] == "Carol Becker"


def test_saved_names_teams_and_recap_text_are_projected_without_mutating_history():
    source = {"participants": [{"id": "player:7", "name": "Carol Becker"}],
              "winners": "Carol Becker / Jane Doe", "text": "CAROL  BECKER won.",
              "ratings": {"Carol Becker": 3.5}, "other": "Carol Beckerson"}
    saved = deepcopy(source)
    output = public_alias_projection(source, {"Carol Becker": "Ann Johnson", "Ann Johnson": "Other"})
    assert output["participants"] == [{"id": "player:7", "name": "Ann Johnson"}]
    assert output["winners"] == "Ann Johnson / Jane Doe"
    assert output["text"] == "Ann Johnson won."
    assert output["ratings"] == {"Ann Johnson": 3.5}
    assert output["other"] == "Carol Beckerson"
    assert source == saved


@pytest.mark.parametrize("path", ["players/7", "round-robins/test", "play-generators/sessions/test",
                                  "leaderboards", "league-results", "matches/7", "weekly-recaps", "tournament-roster"])
def test_public_boundary_masks_snapshots_even_with_staff_bearer(monkeypatch, path):
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_PLAYER_ALIASES", "true")
    monkeypatch.setattr("services.api.public_alias_middleware.load_player_aliases",
                        lambda db, club_id: {"Carol Becker": "Ann Johnson"} if club_id == "club" else {})
    app = FastAPI()
    @app.get("/clubs/{slug}/{rest:path}")
    def public(slug, rest):
        return {"name": "Carol Becker", "player_id": 7}
    @app.get("/admin/clubs/club/players/roster")
    def staff():
        return {"name": "Carol Becker", "player_id": 7}
    install_public_alias_middleware(app, get_club=lambda slug: {"id": slug}, get_supabase_client=lambda: object())
    client = TestClient(app)
    response = client.get(f"/clubs/club/{path}", headers={"Authorization": "Bearer does-not-grant-public-names"})
    assert response.status_code == 200
    assert response.json() == {"name": "Ann Johnson", "player_id": 7}
    assert client.get("/admin/clubs/club/players/roster").json()["name"] == "Carol Becker"
    assert client.get(f"/clubs/other/{path}").json()["name"] == "Carol Becker"


def test_alias_lookup_failure_does_not_expose_real_name(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_PLAYER_ALIASES", "true")
    def unavailable(*args):
        raise RuntimeError("offline")
    monkeypatch.setattr("services.api.public_alias_middleware.load_player_aliases", unavailable)
    app = FastAPI()
    @app.get("/clubs/club/players")
    def public():
        return {"name": "Carol Becker"}
    install_public_alias_middleware(app, get_club=lambda slug: {"id": slug}, get_supabase_client=lambda: object())
    response = TestClient(app).get("/clubs/club/players")
    assert response.status_code == 503
    assert "Carol" not in response.text


def test_public_export_escapes_alias_and_keeps_results(monkeypatch):
    import csv
    import io
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_PLAYER_ALIASES", "true")
    monkeypatch.setattr("services.api.public_alias_middleware.load_player_aliases",
                        lambda *_: {"Carol Becker": "Johnson, Ann"})
    app = FastAPI()
    @app.get("/clubs/club/play-generators/sessions/test/export")
    def export():
        return Response("name,score\r\nCarol Becker,11\r\n", media_type="text/csv")
    install_public_alias_middleware(app, get_club=lambda slug: {"id": slug}, get_supabase_client=lambda: object())
    response = TestClient(app).get("/clubs/club/play-generators/sessions/test/export")
    assert response.status_code == 200
    assert list(csv.reader(io.StringIO(response.text))) == [["name", "score"], ["Johnson, Ann", "11"]]
