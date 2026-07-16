from jupr_app.domain.tournament_partner_service import create_partner_request
from jupr_app.services import public_tournament_partner_request_service as svc
from tests.test_tournament_partner_requests import _FakeSupabase, _storage


def _install_verified_bundle(monkeypatch):
    def fake_verified_bundle(_supabase, *, club_id, edit_token, tournament_id=None):
        return (
            {"tournament_id": "tour-1", "registration_id": "reg_elizabeth", "email": "elizabeth@example.com"},
            {
                "registration": {"id": "reg_elizabeth", "email": "elizabeth@example.com", "display_name": "Elizabeth Whelan"},
                "settings": {"registration_slug": "spring"},
                "selections": [
                    {
                        "id": "sel_elizabeth",
                        "tournament_id": "tour-1",
                        "registration_id": "reg_elizabeth",
                        "event_option_id": "event-wd-35",
                    }
                ],
            },
        )

    monkeypatch.setattr(svc, "_verified_bundle", fake_verified_bundle)


def test_public_partner_request_accept_creates_confirmed_team(monkeypatch):
    storage = _storage()
    supabase = _FakeSupabase(storage)
    _install_verified_bundle(monkeypatch)
    request = create_partner_request(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        source="NEEDS_PARTNER_LIST",
    )

    result = svc.accept_public_tournament_partner_request(
        supabase,
        club_id="club",
        edit_token="token",
        tournament_id="tour-1",
        partner_request_id=request["id"],
    )

    assert result["ok"] is True
    assert result["status"] == "ACCEPTED"
    assert len(storage["tournament_registration_team_links"]) == 1
    assert {row["selection_id"] for row in storage["tournament_registration_team_members"]} == {"sel_mary", "sel_elizabeth"}
    assert storage["tournament_registration_partner_requests"][0]["status"] == "ACCEPTED"


def test_public_partner_request_list_shows_incoming(monkeypatch):
    storage = _storage()
    supabase = _FakeSupabase(storage)
    _install_verified_bundle(monkeypatch)
    request = create_partner_request(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        source="NEEDS_PARTNER_LIST",
    )

    result = svc.list_public_tournament_partner_requests(
        supabase,
        club_id="club",
        edit_token="token",
        tournament_id="tour-1",
    )

    assert result["ok"] is True
    assert result["summary"]["pending_incoming"] == 1
    assert result["incoming"][0]["id"] == request["id"]
    assert result["incoming"][0]["direction"] == "incoming"
